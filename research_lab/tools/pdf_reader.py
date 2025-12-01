"""PDF reading tool for extracting text from research papers."""

from typing import Optional, Dict, Any
import io
import httpx
from pathlib import Path

try:
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    fitz = None

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from config.settings import settings


class PDFReaderInput(BaseModel):
    """Input schema for PDF reading."""
    url: str = Field(..., description="URL or file path to PDF document")
    max_pages: Optional[int] = Field(default=50, description="Maximum number of pages to extract (default 50)")
    start_page: Optional[int] = Field(default=1, description="Starting page number (1-indexed)")


class PDFReaderTool:
    """
    Tool for reading and extracting text from PDF documents.
    
    Supports both local files and remote URLs (Arxiv, etc.).
    Extracts full text content for RAG indexing and analysis.
    """
    
    def __init__(self):
        """Initialize PDF reader tool."""
        if not PDF_AVAILABLE:
            raise ImportError(
                "PyMuPDF is not installed. Install it with: pip install pymupdf"
            )
        self._client = httpx.Client(timeout=60.0, follow_redirects=True)
    
    def read_pdf(
        self,
        url: str,
        max_pages: Optional[int] = 50,
        start_page: int = 1
    ) -> Dict[str, Any]:
        """
        Read and extract text from a PDF document.
        
        Args:
            url: URL or file path to PDF
            max_pages: Maximum number of pages to extract
            start_page: Starting page number (1-indexed)
            
        Returns:
            Dictionary with extracted text and metadata
        """
        if not PDF_AVAILABLE:
            return {
                "success": False,
                "error": "PyMuPDF is not installed. Install it with: pip install pymupdf",
                "text": "",
                "pages": 0
            }
        
        try:
            # Check if it's a local file path
            pdf_path = Path(url)
            if pdf_path.exists() and pdf_path.is_file():
                # Read from local file
                pdf_bytes = pdf_path.read_bytes()
            else:
                # Download from URL
                response = self._client.get(url)
                response.raise_for_status()
                pdf_bytes = response.content
            
            # Open PDF with PyMuPDF
            pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            total_pages = len(pdf_doc)
            
            # Calculate page range
            end_page = min(start_page + max_pages - 1, total_pages)
            actual_pages = end_page - start_page + 1
            
            # Extract text from pages
            text_parts = []
            for page_num in range(start_page - 1, end_page):  # PyMuPDF uses 0-indexed
                page = pdf_doc[page_num]
                text = page.get_text()
                if text.strip():
                    text_parts.append(f"--- Page {page_num + 1} ---\n{text}\n")
            
            pdf_doc.close()
            
            full_text = "\n".join(text_parts)
            
            return {
                "success": True,
                "text": full_text,
                "total_pages": total_pages,
                "extracted_pages": actual_pages,
                "start_page": start_page,
                "end_page": end_page,
                "url": url
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "text": "",
                "pages": 0
            }
    
    def extract_abstract(self, url: str) -> Optional[str]:
        """
        Extract abstract from PDF (typically first few pages).
        
        Args:
            url: URL or file path to PDF
            
        Returns:
            Abstract text or None
        """
        result = self.read_pdf(url, max_pages=3, start_page=1)
        if result["success"]:
            text = result["text"]
            # Try to find abstract section
            abstract_keywords = ["abstract", "summary", "introduction"]
            lines = text.split("\n")
            abstract_start = None
            
            for i, line in enumerate(lines):
                if any(keyword in line.lower() for keyword in abstract_keywords):
                    abstract_start = i
                    break
            
            if abstract_start is not None:
                # Extract next 20-30 lines as abstract
                abstract_lines = lines[abstract_start:abstract_start + 30]
                return "\n".join(abstract_lines)
            
            # Fallback: return first 500 characters
            return text[:500]
        
        return None
    
    def as_langchain_tool(self):
        """Return as a LangChain tool."""
        @tool
        def read_pdf(url: str, max_pages: int = 50, start_page: int = 1) -> str:
            """
            Read and extract text from a PDF document (research papers, articles, etc.).
            
            This tool can read PDFs from:
            - URLs (e.g., Arxiv PDF links)
            - Local file paths
            
            Args:
                url: URL or file path to the PDF document
                max_pages: Maximum number of pages to extract (default: 50)
                start_page: Starting page number, 1-indexed (default: 1)
                
            Returns:
                Extracted text content from the PDF with page markers
            """
            if not PDF_AVAILABLE:
                return "Error: PyMuPDF is not installed. Install it with: pip install pymupdf"
            
            result = self.read_pdf(url, max_pages, start_page)
            
            if not result["success"]:
                return f"Error reading PDF: {result.get('error', 'Unknown error')}"
            
            text = result["text"]
            metadata = (
                f"Extracted {result['extracted_pages']} pages "
                f"(pages {result['start_page']}-{result['end_page']} of {result['total_pages']} total)"
            )
            
            if not text.strip():
                return f"{metadata}\n\nNo text content found in PDF."
            
            return f"{metadata}\n\n{text}"
        
        return read_pdf
    
    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, '_client'):
            self._client.close()

