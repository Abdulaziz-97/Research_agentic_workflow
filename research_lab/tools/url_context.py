"""URL context tool for extracting and processing web content.

Supports Gemini's native url_context tool when available,
with fallback to web scraping + LLM processing for other providers (DeepSeek, OpenAI, etc.).
"""

from typing import Optional, Dict, Any, List
import httpx
from urllib.parse import urlparse

try:
    from google import genai
    from google.genai.types import GenerateContentConfig
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None
    GenerateContentConfig = None

try:
    from bs4 import BeautifulSoup
    BEAUTIFUL_SOUP_AVAILABLE = True
except ImportError:
    BEAUTIFUL_SOUP_AVAILABLE = False
    BeautifulSoup = None

try:
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    ChatOpenAI = None

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    webdriver = None
    Options = None
    Service = None
    ChromeDriverManager = None

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

from config.settings import settings


class URLContextInput(BaseModel):
    """Input schema for URL context extraction."""
    url: str = Field(..., description="URL to extract content from")
    query: Optional[str] = Field(default=None, description="Optional query/question about the URL content")
    use_gemini: bool = Field(default=True, description="Use Gemini's native url_context if available")


class URLContextTool:
    """
    Tool for extracting and processing content from URLs.
    
    When Gemini API is available, uses native url_context tool for
    intelligent content extraction. Otherwise falls back to web scraping.
    
    This is particularly useful for:
    - Reading research paper pages
    - Extracting information from academic websites
    - Processing content from Semantic Scholar, PubMed, etc.
    - Getting context from any web page
    """
    
    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        use_llm_processing: bool = True
    ):
        """
        Initialize URL context tool.
        
        Args:
            gemini_api_key: Optional Gemini API key (uses GEMINI_API_KEY env var if not provided)
            use_llm_processing: Whether to use LLM (DeepSeek/OpenAI) to process scraped content
        """
        # Use proper user agent and headers to avoid blocking
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
        self._client = httpx.Client(timeout=30.0, follow_redirects=True, headers=headers)
        self.gemini_api_key = gemini_api_key
        self._gemini_client = None
        self.use_llm_processing = use_llm_processing
        self._llm = None
        
        # Initialize Gemini client if available
        if GEMINI_AVAILABLE:
            try:
                self._gemini_client = genai.Client(api_key=self.gemini_api_key)
            except Exception:
                # Gemini not configured, will use fallback
                self._gemini_client = None
        
        # Initialize LLM for processing scraped content (DeepSeek/OpenAI)
        if self.use_llm_processing and LANGCHAIN_AVAILABLE:
            try:
                llm_kwargs = {
                    "model": settings.openai_model,
                    "openai_api_key": settings.openai_api_key,
                    "temperature": 0.3
                }
                if settings.openai_base_url:
                    llm_kwargs["openai_api_base"] = settings.openai_base_url
                self._llm = ChatOpenAI(**llm_kwargs)
            except Exception:
                # LLM not configured
                self._llm = None
    
    def extract_with_selenium(
        self,
        url: str,
        query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract content using Selenium (Google Chrome).
        
        Args:
            url: URL to extract content from
            query: Optional question about the content
            
        Returns:
            Dictionary with extracted content and metadata
        """
        if not SELENIUM_AVAILABLE:
            return {
                "success": False,
                "error": "Selenium not available. Please install selenium and webdriver-manager.",
                "method": "selenium"
            }
            
        try:
            # Configure Chrome options
            chrome_options = Options()
            chrome_options.add_argument("--headless")  # Run in headless mode
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            
            # Anti-detection settings
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            # Initialize driver
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            
            # Additional stealth: override navigator.webdriver
            driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
                "source": """
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined
                    })
                """
            })
            
            try:
                driver.get(url)
                
                # Wait for body to be present
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                
                # Get page source
                html_content = driver.page_source
                
                # Use BeautifulSoup if available to clean up
                if BEAUTIFUL_SOUP_AVAILABLE:
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    # Remove script and style elements
                    for script in soup(["script", "style", "nav", "footer", "header"]):
                        script.decompose()
                    
                    # Extract text
                    text = soup.get_text(separator='\n', strip=True)
                    
                    # Clean up whitespace
                    lines = [line.strip() for line in text.split('\n') if line.strip()]
                    text = '\n'.join(lines)
                else:
                    # Basic extraction
                    text = driver.find_element(By.TAG_NAME, "body").text
                
                # Process with LLM
                llm_result = self._process_content_with_llm(text, query, url)
                
                if "error" in llm_result:
                    return {
                        "success": False,
                        "error": llm_result["error"],
                        "url": url,
                        "method": f"selenium+{llm_result.get('method', 'unknown')}"
                    }
                
                return {
                    "success": True,
                    "content": llm_result["content"],
                    "method": f"selenium+{llm_result['method']}",
                    "url": url
                }
                
            finally:
                driver.quit()
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "method": "selenium"
            }
    
    def _try_wikipedia_api(self, page_title: str) -> Dict[str, Any]:
        """
        Try to get Wikipedia content using their API (more reliable than scraping).
        
        Args:
            page_title: Wikipedia page title
            
        Returns:
            Dictionary with content or error
        """
        try:
            # Use Wikipedia's REST API
            api_url = "https://en.wikipedia.org/api/rest_v1/page/html/" + page_title.replace(" ", "_")
            response = self._client.get(api_url)
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "content": response.text
                }
            else:
                return {
                    "success": False,
                    "error": f"Wikipedia API returned {response.status_code}"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def extract_with_gemini(
        self,
        url: str,
        query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract content using Gemini's native url_context tool.
        
        Args:
            url: URL to process
            query: Optional question about the content
            
        Returns:
            Dictionary with extracted content and metadata
        """
        if not GEMINI_AVAILABLE or not self._gemini_client:
            return {
                "success": False,
                "error": "Gemini API not available",
                "method": "gemini"
            }
        
        try:
            # Build prompt
            if query:
                prompt = f"{query}\n\nURL: {url}"
            else:
                prompt = f"Extract and summarize the key information from: {url}"
            
            # Use Gemini's url_context tool
            response = self._gemini_client.models.generate_content(
                model="gemini-2.5-flash",  # Fast and cost-effective
                contents=prompt,
                config=GenerateContentConfig(tools=[{"url_context": {}}])
            )
            
            # Extract text from response
            text_parts = []
            for candidate in response.candidates:
                for part in candidate.content.parts:
                    if hasattr(part, 'text') and part.text:
                        text_parts.append(part.text)
            
            content = "\n".join(text_parts)
            
            return {
                "success": True,
                "content": content,
                "method": "gemini_url_context",
                "url": url
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "method": "gemini"
            }
    
            return {
                "success": False,
                "error": str(e),
                "method": "gemini"
            }
    
    def _process_content_with_llm(
        self,
        text: str,
        query: Optional[str],
        url: str
    ) -> Dict[str, Any]:
        """
        Process extracted text with LLM (DeepSeek/OpenAI).
        
        Args:
            text: Raw extracted text
            query: User query (optional)
            url: Source URL
            
        Returns:
            Dictionary with processed content and method
        """
        if not self._llm:
            return {
                "content": text,
                "method": "raw_extraction (no_llm)"
            }
            
        try:
            # Check if text is too short or mostly non-readable
            if len(text.strip()) < 50:
                return {
                    "error": "Extracted content is too short or empty.",
                    "method": "llm_failed"
                }
            
            if query:
                # Use LLM to extract relevant information based on query
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are a research assistant that extracts and summarizes relevant information from web content.
                    Focus on answering the user's question and extracting key information.
                    Be concise but comprehensive.
                    If the content is corrupted, unreadable, or doesn't contain relevant information, clearly state this."""),
                    ("human", """Question: {query}

Web content from {url}:
{content}

Extract and summarize the information relevant to the question. If the content is corrupted, unreadable, or doesn't contain relevant information, clearly state this.""")
                ])
                
                chain = prompt | self._llm | StrOutputParser()
                processed_text = chain.invoke({
                    "query": query,
                    "url": url,
                    "content": text
                })
                method = "llm_query_processing"
                
            else:
                # No query, but use LLM to clean up and summarize
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are a research assistant that extracts key information from web content.
                    Summarize the main points, remove noise, and present the information clearly."""),
                    ("human", """Extract and summarize the key information from this web content:

URL: {url}

Content:
{content}

Provide a clear, structured summary of the main information.""")
                ])
                
                chain = prompt | self._llm | StrOutputParser()
                processed_text = chain.invoke({
                    "url": url,
                    "content": text[:15000]  # Limit for LLM context
                })
                method = "llm_summary"
            
            # Check if LLM detected corruption
            if processed_text and any(keyword in processed_text.lower() for keyword in ["corrupted", "unreadable", "cannot extract", "no meaningful"]):
                return {
                    "error": f"LLM detected corrupted or unreadable content: {processed_text[:200]}",
                    "method": method
                }
                
            return {
                "content": processed_text,
                "method": method
            }
            
        except Exception as e:
            # LLM processing failed, use raw content if substantial
            if len(text.strip()) > 100:
                return {
                    "content": text,
                    "method": "raw_extraction (llm_failed)"
                }
            else:
                return {
                    "error": f"LLM processing failed and extracted content is insufficient: {str(e)}",
                    "method": "llm_failed"
                }
    
    def extract_with_scraping(
        self,
        url: str,
        query: Optional[str] = None,
        max_length: int = 15000
    ) -> Dict[str, Any]:
        """
        Extract content using web scraping (fallback method).
        Optionally processes with LLM (DeepSeek/OpenAI) for better understanding.
        
        Args:
            url: URL to scrape
            query: Optional question about the content
            max_length: Maximum content length to extract before LLM processing
            
        Returns:
            Dictionary with extracted content and metadata
        """
        try:
            html_content = None
            
            # For Wikipedia, try using their API first (more reliable)
            if "wikipedia.org" in url and "/wiki/" in url:
                # Extract page title from URL
                page_title = url.split("/wiki/")[-1].replace("_", " ")
                api_result = self._try_wikipedia_api(page_title)
                if api_result["success"]:
                    html_content = api_result["content"]
            
            # If Wikipedia API didn't work or not Wikipedia, try direct scraping
            if html_content is None:
                # For Wikipedia, add a small delay to be respectful
                if "wikipedia.org" in url:
                    import time
                    time.sleep(0.5)
                
                response = self._client.get(url)
                response.raise_for_status()
                html_content = response.text
            
            if BEAUTIFUL_SOUP_AVAILABLE:
                # Use BeautifulSoup for better text extraction
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.decompose()
                
                # Extract text
                text = soup.get_text(separator='\n', strip=True)
                
                # Clean up whitespace
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                text = '\n'.join(lines)
            else:
                # Basic text extraction without BeautifulSoup
                # This is a simple fallback - not ideal but works
                import re
                text = re.sub(r'<[^>]+>', '', html_content)
                text = re.sub(r'\s+', ' ', text)
            
            # Truncate if too long (before LLM processing)
            original_length = len(text)
            if len(text) > max_length:
                text = text[:max_length] + "... [truncated]"
            
            # Check for corrupted or unreadable content before LLM processing
            # Detect binary data, encoding issues, or heavily corrupted text
            if text:
                # Check for high ratio of non-printable characters (indicates binary/corrupted data)
                printable_chars = sum(1 for c in text if c.isprintable() or c.isspace())
                total_chars = len(text)
                if total_chars > 0:
                    printable_ratio = printable_chars / total_chars
                    # If less than 70% printable, likely corrupted
                    if printable_ratio < 0.7:
                        return {
                            "success": False,
                            "error": "Content appears corrupted or contains binary data. Cannot extract meaningful text.",
                            "url": url,
                            "method": "scraping",
                            "printable_ratio": printable_ratio
                        }
            
            # Process with LLM
            llm_result = self._process_content_with_llm(text, query, url)
            
            if "error" in llm_result:
                return {
                    "success": False,
                    "error": llm_result["error"],
                    "url": url,
                    "method": f"scraping+{llm_result.get('method', 'unknown')}"
                }
            
            return {
                "success": True,
                "content": llm_result["content"],
                "method": f"scraping+{llm_result['method']}",
                "url": url,
                "original_length": original_length
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "method": "scraping"
            }
    
    def extract_content(
        self,
        url: str,
        query: Optional[str] = None,
        use_gemini: bool = True,
        use_selenium: bool = False,
        fallback_to_scraping: bool = True
    ) -> Dict[str, Any]:
        """
        Extract content from URL using best available method.
        
        Args:
            url: URL to extract content from
            query: Optional question about the content
            use_gemini: Try Gemini's url_context first if available
            use_selenium: Try Selenium (Chrome) if available
            fallback_to_scraping: Fall back to scraping if other methods fail
            
        Returns:
            Dictionary with extracted content and metadata
        """
        # Validate URL
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return {
                "success": False,
                "error": f"Invalid URL: {url}",
                "method": "validation"
            }
        
        # Try Gemini first if requested and available
        if use_gemini and GEMINI_AVAILABLE and self._gemini_client:
            result = self.extract_with_gemini(url, query)
            if result["success"]:
                return result
        
        # Try Selenium if requested
        if use_selenium and SELENIUM_AVAILABLE:
            result = self.extract_with_selenium(url, query)
            if result["success"]:
                return result
        
        # Fall back to scraping if requested
        if fallback_to_scraping:
            result = self.extract_with_scraping(url, query)
            if result["success"]:
                return result
            else:
                # If scraping also failed, return error with details
                errors = []
                if use_gemini:
                    gemini_result = self.extract_with_gemini(url, query)
                    if not gemini_result["success"]:
                        errors.append(f"Gemini: {gemini_result.get('error')}")
                
                if use_selenium:
                    selenium_result = self.extract_with_selenium(url, query)
                    if not selenium_result["success"]:
                        errors.append(f"Selenium: {selenium_result.get('error')}")
                        
                errors.append(f"Scraping: {result.get('error')}")
                
                return {
                    "success": False,
                    "error": "All methods failed. " + "; ".join(errors),
                    "method": "all_failed"
                }
        
        return result
    
    def as_langchain_tool(self):
        """Return as a LangChain tool."""
        @tool
        def extract_url_content(
            url: str,
            query: Optional[str] = None,
            use_gemini: bool = True,
            use_selenium: bool = False
        ) -> str:
            """
            Extract and process content from a web URL.
            
            This tool can intelligently extract information from web pages,
            research paper pages, academic websites, and more. When Gemini API
            is available, it uses advanced content understanding. Otherwise,
            it falls back to web scraping.
            
            Args:
                url: The URL to extract content from
                query: Optional question or instruction about what to extract
                       (e.g., "What are the main findings?" or "Extract the abstract")
                use_gemini: Whether to use Gemini's native url_context tool
                           if available (default: True)
                use_selenium: Whether to use Selenium (Chrome) for extraction
                             (useful for JavaScript-heavy sites)
                
            Returns:
                Extracted and processed content from the URL
            """
            result = self.extract_content(url, query, use_gemini, use_selenium)
            
            if not result["success"]:
                return f"Error extracting content from {url}: {result.get('error', 'Unknown error')}"
            
            content = result["content"]
            method = result.get("method", "unknown")
            
            header = f"Content extracted from {url} (method: {method})"
            if query:
                header += f"\nQuery: {query}"
            
            return f"{header}\n\n{content}"
        
        return extract_url_content
    
    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, '_client'):
            self._client.close()

