"""Arxiv paper search tool."""

from typing import List, Optional
from datetime import datetime
import arxiv

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from states.agent_state import Paper
from config.settings import settings


class ArxivSearchInput(BaseModel):
    """Input schema for Arxiv search."""
    query: str = Field(..., description="Search query for Arxiv papers")
    max_results: int = Field(default=10, description="Maximum number of results")
    sort_by: str = Field(default="relevance", description="Sort by: relevance, lastUpdatedDate, submittedDate")


class ArxivSearchTool:
    """
    Tool for searching research papers on Arxiv.
    
    Provides access to the Arxiv repository of preprints in physics,
    mathematics, computer science, and related fields.
    """
    
    def __init__(self, max_results: Optional[int] = None):
        """
        Initialize Arxiv search tool.
        
        Args:
            max_results: Default maximum results per search
        """
        self.max_results = max_results or settings.arxiv_max_results
        self.client = arxiv.Client()
    
    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        sort_by: str = "relevance",
        categories: Optional[List[str]] = None
    ) -> List[Paper]:
        """
        Search for papers on Arxiv.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            sort_by: Sort criteria (relevance, lastUpdatedDate, submittedDate)
            categories: Optional list of Arxiv categories to filter
            
        Returns:
            List of Paper objects
        """
        max_results = max_results or self.max_results
        
        # Map sort options
        sort_mapping = {
            "relevance": arxiv.SortCriterion.Relevance,
            "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
            "submittedDate": arxiv.SortCriterion.SubmittedDate
        }
        sort_criterion = sort_mapping.get(sort_by, arxiv.SortCriterion.Relevance)
        
        # Build query with category filter
        if categories:
            category_filter = " OR ".join([f"cat:{cat}" for cat in categories])
            query = f"({query}) AND ({category_filter})"
        
        # Execute search
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_criterion
        )
        
        papers = []
        for result in self.client.results(search):
            paper = Paper(
                id=result.entry_id,
                title=result.title,
                authors=[author.name for author in result.authors],
                abstract=result.summary,
                url=result.pdf_url or result.entry_id,
                source="arxiv",
                published_date=result.published,
                field=self._categorize_paper(result.categories),
                citations=0  # Arxiv doesn't provide citation count
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
            field: Research field (ai_ml, physics, mathematics, cs)
            max_results: Maximum results
            
        Returns:
            List of Paper objects
        """
        # Map fields to Arxiv categories
        field_categories = {
            "ai_ml": ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "stat.ML"],
            "physics": ["physics", "hep-th", "hep-ph", "quant-ph", "cond-mat"],
            "mathematics": ["math"],
            "computer_science": ["cs"],
            "biology": ["q-bio"],
            "chemistry": ["physics.chem-ph"]
        }
        
        categories = field_categories.get(field)
        return self.search(query, max_results, categories=categories)
    
    def get_paper_by_id(self, arxiv_id: str) -> Optional[Paper]:
        """
        Get a specific paper by Arxiv ID.
        
        Args:
            arxiv_id: Arxiv paper ID (e.g., "2301.00001")
            
        Returns:
            Paper object or None
        """
        search = arxiv.Search(id_list=[arxiv_id])
        
        results = list(self.client.results(search))
        if results:
            result = results[0]
            return Paper(
                id=result.entry_id,
                title=result.title,
                authors=[author.name for author in result.authors],
                abstract=result.summary,
                url=result.pdf_url or result.entry_id,
                source="arxiv",
                published_date=result.published,
                field=self._categorize_paper(result.categories)
            )
        return None
    
    def _categorize_paper(self, categories: List[str]) -> str:
        """Determine primary field from Arxiv categories."""
        category_str = " ".join(categories).lower()
        
        if any(cat in category_str for cat in ["cs.ai", "cs.lg", "cs.cl", "cs.cv", "stat.ml"]):
            return "ai_ml"
        elif any(cat in category_str for cat in ["physics", "hep", "quant", "cond-mat"]):
            return "physics"
        elif "math" in category_str:
            return "mathematics"
        elif "q-bio" in category_str:
            return "biology"
        elif "cs" in category_str:
            return "computer_science"
        
        return "general"
    
    def as_langchain_tool(self):
        """Return as a LangChain tool."""
        @tool
        def arxiv_search(query: str, max_results: int = 10) -> str:
            """
            Search for research papers on Arxiv.
            
            Args:
                query: Search query for papers
                max_results: Maximum number of results (default 10)
                
            Returns:
                Formatted string with paper information
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
                result += f"\n   Field: {paper.field}\n"
                result += f"   URL: {paper.url}\n"
                result += f"   Abstract: {paper.abstract[:300]}...\n\n"
            
            return result
        
        return arxiv_search

