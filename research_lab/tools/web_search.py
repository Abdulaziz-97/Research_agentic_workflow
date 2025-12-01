"""Web search tool using Tavily for research queries."""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from langchain_core.tools import tool

from config.settings import settings


@dataclass
class WebSearchResult:
    """Represents a web search result."""
    title: str
    url: str
    snippet: str
    source: str = "tavily"
    score: float = 0.0


class WebSearchTool:
    """
    Web search tool using Tavily API.
    
    Provides high-quality search capability with support for
    academic and research queries.
    """
    
    TAVILY_API_URL = "https://api.tavily.com/search"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Tavily web search tool.
        
        Args:
            api_key: Tavily API key (defaults to settings)
        """
        self.api_key = api_key or settings.tavily_api_key
        self._client = httpx.Client(timeout=30.0)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def search(
        self,
        query: str,
        max_results: int = 10,
        search_depth: str = "basic",
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None
    ) -> List[WebSearchResult]:
        """
        Perform a web search using Tavily API.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            search_depth: "basic" or "advanced" (advanced is more thorough)
            include_domains: List of domains to include
            exclude_domains: List of domains to exclude
            
        Returns:
            List of WebSearchResult objects
        """
        if not self.api_key:
            print("Warning: Tavily API key not set")
            return []
        
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "include_answer": True,
            "include_raw_content": False
        }
        
        if include_domains:
            payload["include_domains"] = include_domains
        if exclude_domains:
            payload["exclude_domains"] = exclude_domains
        
        try:
            response = self._client.post(self.TAVILY_API_URL, json=payload)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            # Parse results
            for item in data.get("results", []):
                results.append(WebSearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                    source="tavily",
                    score=item.get("score", 0.0)
                ))
            
            return results[:max_results]
            
        except httpx.HTTPError as e:
            print(f"Tavily search error: {e}")
            return []
    
    def search_academic(
        self,
        query: str,
        max_results: int = 10
    ) -> List[WebSearchResult]:
        """
        Search with academic focus.
        
        Args:
            query: Search query
            max_results: Maximum results
            
        Returns:
            List of WebSearchResult objects
        """
        # Include academic domains
        academic_domains = [
            "arxiv.org",
            "scholar.google.com",
            "pubmed.ncbi.nlm.nih.gov",
            "nature.com",
            "science.org",
            "ieee.org",
            "acm.org",
            "springer.com",
            "sciencedirect.com"
        ]
        
        return self.search(
            query,
            max_results=max_results,
            search_depth="advanced",
            include_domains=academic_domains
        )
    
    def search_with_site(
        self,
        query: str,
        site: str,
        max_results: int = 10
    ) -> List[WebSearchResult]:
        """
        Search within a specific site.
        
        Args:
            query: Search query
            site: Site domain (e.g., "nature.com")
            max_results: Maximum results
            
        Returns:
            List of WebSearchResult objects
        """
        return self.search(
            query,
            max_results=max_results,
            include_domains=[site]
        )
    
    def as_langchain_tool(self):
        """Return as a LangChain tool."""
        @tool
        def web_search(query: str, max_results: int = 10) -> str:
            """
            Search the web for information using Tavily.
            
            Use this when academic databases don't have the information needed,
            or for general knowledge queries.
            
            Args:
                query: Search query
                max_results: Maximum number of results (default 10)
                
            Returns:
                Formatted string with search results
            """
            results = self.search(query, max_results)
            
            if not results:
                return "No results found for the given query."
            
            output = f"Found {len(results)} results:\n\n"
            for i, result in enumerate(results, 1):
                output += f"{i}. **{result.title}**\n"
                output += f"   URL: {result.url}\n"
                output += f"   Score: {result.score:.2f}\n"
                output += f"   {result.snippet[:300]}...\n\n"
            
            return output
        
        return web_search
    
    def __del__(self):
        """Cleanup HTTP client."""
        if hasattr(self, '_client'):
            self._client.close()


class ResearchToolkit:
    """
    Combined toolkit with all research tools.
    
    Provides a unified interface for all research search capabilities.
    """
    
    def __init__(self):
        """Initialize all research tools."""
        from .arxiv_tool import ArxivSearchTool
        from .semantic_scholar import SemanticScholarTool
        from .pubmed_tool import PubMedSearchTool
        from .pdf_reader import PDFReaderTool
        from .url_context import URLContextTool
        
        self.arxiv = ArxivSearchTool()
        self.semantic_scholar = SemanticScholarTool()
        self.pubmed = PubMedSearchTool()
        self.web = WebSearchTool()
        try:
            self.pdf_reader = PDFReaderTool()
        except ImportError:
            # PDF reader not available if PyMuPDF not installed
            self.pdf_reader = None
        
        try:
            self.url_context = URLContextTool(gemini_api_key=settings.gemini_api_key)
        except Exception:
            # URL context tool not available
            self.url_context = None
    
    def get_tools_for_field(self, field: str) -> List:
        """
        Get appropriate tools for a research field.
        
        Args:
            field: Research field
            
        Returns:
            List of LangChain tools
        """
        tools = []
        
        # All fields get web search, PDF reader, and URL context
        tools.append(self.web.as_langchain_tool())
        if self.pdf_reader:
            tools.append(self.pdf_reader.as_langchain_tool())
        if self.url_context:
            tools.append(self.url_context.as_langchain_tool())
        
        # Field-specific tools
        if field in ["ai_ml", "physics", "mathematics", "computer_science"]:
            tools.append(self.arxiv.as_langchain_tool())
            tools.append(self.semantic_scholar.as_langchain_tool())
        
        if field in ["biology", "medicine", "neuroscience", "chemistry"]:
            tools.append(self.pubmed.as_langchain_tool())
            tools.append(self.semantic_scholar.as_langchain_tool())
        
        return tools
    
    def get_all_tools(self) -> List:
        """Get all available tools."""
        tools = [
            self.arxiv.as_langchain_tool(),
            self.semantic_scholar.as_langchain_tool(),
            self.pubmed.as_langchain_tool(),
            self.web.as_langchain_tool()
        ]
        if self.pdf_reader:
            tools.append(self.pdf_reader.as_langchain_tool())
        if self.url_context:
            tools.append(self.url_context.as_langchain_tool())
        return tools
