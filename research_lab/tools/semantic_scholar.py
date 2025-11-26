"""Semantic Scholar paper search tool."""

from typing import List, Optional, Dict, Any
from datetime import datetime
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from states.agent_state import Paper
from config.settings import settings


class SemanticScholarTool:
    """
    Tool for searching research papers via Semantic Scholar API.
    
    Provides access to a large corpus of academic papers with citation
    information and paper influence metrics.
    """
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    def __init__(self, max_results: Optional[int] = None):
        """
        Initialize Semantic Scholar tool.
        
        Args:
            max_results: Default maximum results per search
        """
        self.max_results = max_results or settings.semantic_scholar_max_results
        self._client = httpx.Client(timeout=30.0)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        year_range: Optional[tuple] = None,
        fields_of_study: Optional[List[str]] = None
    ) -> List[Paper]:
        """
        Search for papers on Semantic Scholar.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            year_range: Optional tuple of (start_year, end_year)
            fields_of_study: Optional list of fields to filter
            
        Returns:
            List of Paper objects
        """
        max_results = max_results or self.max_results
        
        # Build request parameters
        params = {
            "query": query,
            "limit": max_results,
            "fields": "paperId,title,authors,abstract,url,venue,year,citationCount,fieldsOfStudy"
        }
        
        if year_range:
            params["year"] = f"{year_range[0]}-{year_range[1]}"
        
        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)
        
        # Execute search
        response = self._client.get(
            f"{self.BASE_URL}/paper/search",
            params=params
        )
        response.raise_for_status()
        
        data = response.json()
        papers = []
        
        for item in data.get("data", []):
            # Parse authors
            authors = [
                author.get("name", "") 
                for author in item.get("authors", [])
            ]
            
            # Determine field
            fields = item.get("fieldsOfStudy", [])
            field = self._map_field(fields)
            
            # Parse publication date
            pub_date = None
            if item.get("year"):
                pub_date = datetime(item["year"], 1, 1)
            
            paper = Paper(
                id=item.get("paperId", ""),
                title=item.get("title", ""),
                authors=authors,
                abstract=item.get("abstract", "") or "",
                url=item.get("url", "") or f"https://www.semanticscholar.org/paper/{item.get('paperId', '')}",
                source="semantic_scholar",
                published_date=pub_date,
                citations=item.get("citationCount", 0) or 0,
                field=field
            )
            papers.append(paper)
        
        return papers
    
    def search_by_field(
        self,
        query: str,
        field: str,
        max_results: Optional[int] = None
    ) -> List[Paper]:
        """
        Search papers filtered by research field.
        
        Args:
            query: Search query
            field: Research field
            max_results: Maximum results
            
        Returns:
            List of Paper objects
        """
        # Map internal fields to Semantic Scholar fields
        field_mapping = {
            "ai_ml": ["Computer Science", "Artificial Intelligence"],
            "physics": ["Physics"],
            "biology": ["Biology"],
            "chemistry": ["Chemistry"],
            "mathematics": ["Mathematics"],
            "neuroscience": ["Neuroscience", "Psychology"],
            "medicine": ["Medicine"],
            "computer_science": ["Computer Science"]
        }
        
        ss_fields = field_mapping.get(field, [])
        return self.search(query, max_results, fields_of_study=ss_fields)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def get_paper_by_id(self, paper_id: str) -> Optional[Paper]:
        """
        Get a specific paper by Semantic Scholar ID.
        
        Args:
            paper_id: Semantic Scholar paper ID
            
        Returns:
            Paper object or None
        """
        params = {
            "fields": "paperId,title,authors,abstract,url,venue,year,citationCount,fieldsOfStudy,references,citations"
        }
        
        response = self._client.get(
            f"{self.BASE_URL}/paper/{paper_id}",
            params=params
        )
        
        if response.status_code == 404:
            return None
        
        response.raise_for_status()
        item = response.json()
        
        authors = [
            author.get("name", "") 
            for author in item.get("authors", [])
        ]
        
        fields = item.get("fieldsOfStudy", [])
        field = self._map_field(fields)
        
        pub_date = None
        if item.get("year"):
            pub_date = datetime(item["year"], 1, 1)
        
        return Paper(
            id=item.get("paperId", ""),
            title=item.get("title", ""),
            authors=authors,
            abstract=item.get("abstract", "") or "",
            url=item.get("url", "") or f"https://www.semanticscholar.org/paper/{paper_id}",
            source="semantic_scholar",
            published_date=pub_date,
            citations=item.get("citationCount", 0) or 0,
            field=field
        )
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def get_citations(self, paper_id: str, max_results: int = 10) -> List[Paper]:
        """
        Get papers that cite a given paper.
        
        Args:
            paper_id: Semantic Scholar paper ID
            max_results: Maximum number of citing papers
            
        Returns:
            List of citing Paper objects
        """
        params = {
            "fields": "paperId,title,authors,abstract,url,year,citationCount",
            "limit": max_results
        }
        
        response = self._client.get(
            f"{self.BASE_URL}/paper/{paper_id}/citations",
            params=params
        )
        response.raise_for_status()
        
        data = response.json()
        papers = []
        
        for item in data.get("data", []):
            citing_paper = item.get("citingPaper", {})
            
            authors = [
                author.get("name", "") 
                for author in citing_paper.get("authors", [])
            ]
            
            pub_date = None
            if citing_paper.get("year"):
                pub_date = datetime(citing_paper["year"], 1, 1)
            
            paper = Paper(
                id=citing_paper.get("paperId", ""),
                title=citing_paper.get("title", ""),
                authors=authors,
                abstract=citing_paper.get("abstract", "") or "",
                url=f"https://www.semanticscholar.org/paper/{citing_paper.get('paperId', '')}",
                source="semantic_scholar",
                published_date=pub_date,
                citations=citing_paper.get("citationCount", 0) or 0
            )
            papers.append(paper)
        
        return papers
    
    def _map_field(self, fields: List[str]) -> str:
        """Map Semantic Scholar fields to internal fields."""
        if not fields:
            return "general"
        
        fields_lower = [f.lower() for f in fields]
        
        if any("artificial intelligence" in f or "machine learning" in f for f in fields_lower):
            return "ai_ml"
        elif any("physics" in f for f in fields_lower):
            return "physics"
        elif any("biology" in f for f in fields_lower):
            return "biology"
        elif any("chemistry" in f for f in fields_lower):
            return "chemistry"
        elif any("mathematics" in f for f in fields_lower):
            return "mathematics"
        elif any("neuroscience" in f or "psychology" in f for f in fields_lower):
            return "neuroscience"
        elif any("medicine" in f for f in fields_lower):
            return "medicine"
        elif any("computer science" in f for f in fields_lower):
            return "computer_science"
        
        return "general"
    
    def as_langchain_tool(self):
        """Return as a LangChain tool."""
        @tool
        def semantic_scholar_search(query: str, max_results: int = 10) -> str:
            """
            Search for research papers on Semantic Scholar.
            
            Args:
                query: Search query for papers
                max_results: Maximum number of results (default 10)
                
            Returns:
                Formatted string with paper information including citations
            """
            papers = self.search(query, max_results)
            
            if not papers:
                return "No papers found for the given query."
            
            result = f"Found {len(papers)} papers:\n\n"
            for i, paper in enumerate(papers, 1):
                result += f"{i}. **{paper.title}**\n"
                result += f"   Authors: {', '.join(paper.authors[:3])}"
                if len(paper.authors) > 3:
                    result += f" et al."
                result += f"\n   Citations: {paper.citations}\n"
                result += f"   Field: {paper.field}\n"
                result += f"   URL: {paper.url}\n"
                if paper.abstract:
                    result += f"   Abstract: {paper.abstract[:300]}...\n"
                result += "\n"
            
            return result
        
        return semantic_scholar_search
    
    def __del__(self):
        """Cleanup HTTP client."""
        if hasattr(self, '_client'):
            self._client.close()

