"""PubMed paper search tool for biomedical literature."""

from typing import List, Optional
from datetime import datetime
import xml.etree.ElementTree as ET

from Bio import Entrez
from langchain_core.tools import tool

from states.agent_state import Paper
from config.settings import settings


class PubMedSearchTool:
    """
    Tool for searching biomedical literature on PubMed.
    
    Provides access to MEDLINE and other life sciences journals
    via the NCBI Entrez API.
    """
    
    def __init__(
        self, 
        email: str = "research_lab@example.com",
        max_results: Optional[int] = None
    ):
        """
        Initialize PubMed search tool.
        
        Args:
            email: Email for NCBI API (required by their terms)
            max_results: Default maximum results per search
        """
        self.max_results = max_results or settings.pubmed_max_results
        Entrez.email = email
    
    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        sort: str = "relevance",
        min_date: Optional[str] = None,
        max_date: Optional[str] = None
    ) -> List[Paper]:
        """
        Search for papers on PubMed.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            sort: Sort order (relevance, pub_date)
            min_date: Minimum publication date (YYYY/MM/DD)
            max_date: Maximum publication date (YYYY/MM/DD)
            
        Returns:
            List of Paper objects
        """
        max_results = max_results or self.max_results
        
        # Search PubMed
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "sort": sort
        }
        
        if min_date:
            search_params["mindate"] = min_date
        if max_date:
            search_params["maxdate"] = max_date
        
        handle = Entrez.esearch(**search_params)
        record = Entrez.read(handle)
        handle.close()
        
        id_list = record.get("IdList", [])
        
        if not id_list:
            return []
        
        # Fetch paper details
        fetch_handle = Entrez.efetch(
            db="pubmed",
            id=id_list,
            rettype="xml",
            retmode="xml"
        )
        
        papers = self._parse_papers(fetch_handle.read())
        fetch_handle.close()
        
        return papers
    
    def _parse_papers(self, xml_data: bytes) -> List[Paper]:
        """Parse PubMed XML response into Paper objects."""
        papers = []
        
        root = ET.fromstring(xml_data)
        
        for article in root.findall(".//PubmedArticle"):
            try:
                # Get PMID
                pmid = article.find(".//PMID")
                paper_id = pmid.text if pmid is not None else ""
                
                # Get title
                title_elem = article.find(".//ArticleTitle")
                title = title_elem.text if title_elem is not None else ""
                
                # Get abstract
                abstract_parts = article.findall(".//AbstractText")
                abstract = " ".join([
                    part.text for part in abstract_parts 
                    if part.text
                ])
                
                # Get authors
                authors = []
                for author in article.findall(".//Author"):
                    last_name = author.find("LastName")
                    first_name = author.find("ForeName")
                    if last_name is not None:
                        name = last_name.text
                        if first_name is not None:
                            name = f"{first_name.text} {name}"
                        authors.append(name)
                
                # Get publication date
                pub_date = None
                pub_date_elem = article.find(".//PubDate")
                if pub_date_elem is not None:
                    year = pub_date_elem.find("Year")
                    month = pub_date_elem.find("Month")
                    day = pub_date_elem.find("Day")
                    
                    if year is not None:
                        try:
                            y = int(year.text)
                            m = int(month.text) if month is not None else 1
                            d = int(day.text) if day is not None else 1
                            pub_date = datetime(y, m, d)
                        except (ValueError, TypeError):
                            pass
                
                # Determine field from MeSH terms
                field = self._determine_field(article)
                
                paper = Paper(
                    id=paper_id,
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{paper_id}/",
                    source="pubmed",
                    published_date=pub_date,
                    field=field
                )
                papers.append(paper)
                
            except Exception:
                continue
        
        return papers
    
    def _determine_field(self, article) -> str:
        """Determine research field from MeSH terms."""
        mesh_terms = []
        for mesh in article.findall(".//MeshHeading/DescriptorName"):
            if mesh.text:
                mesh_terms.append(mesh.text.lower())
        
        mesh_str = " ".join(mesh_terms)
        
        # Check for field indicators
        if any(term in mesh_str for term in ["neural", "brain", "neuron", "cognitive"]):
            return "neuroscience"
        elif any(term in mesh_str for term in ["drug", "therapy", "clinical", "patient"]):
            return "medicine"
        elif any(term in mesh_str for term in ["gene", "protein", "cell", "molecular"]):
            return "biology"
        elif any(term in mesh_str for term in ["chemical", "compound", "synthesis"]):
            return "chemistry"
        
        return "biology"  # Default for PubMed
    
    def search_by_field(
        self,
        query: str,
        field: str,
        max_results: Optional[int] = None
    ) -> List[Paper]:
        """
        Search papers with field-specific filters.
        
        Args:
            query: Search query
            field: Research field
            max_results: Maximum results
            
        Returns:
            List of Paper objects
        """
        # Add field-specific MeSH terms to query
        field_filters = {
            "biology": "[MeSH Terms] AND (genetics OR molecular biology)",
            "medicine": "[MeSH Terms] AND (therapy OR clinical trial)",
            "neuroscience": "[MeSH Terms] AND (brain OR neuroscience)",
            "chemistry": "[MeSH Terms] AND (chemistry OR biochemistry)"
        }
        
        filter_term = field_filters.get(field, "")
        enhanced_query = f"({query}) {filter_term}" if filter_term else query
        
        return self.search(enhanced_query, max_results)
    
    def get_paper_by_pmid(self, pmid: str) -> Optional[Paper]:
        """
        Get a specific paper by PubMed ID.
        
        Args:
            pmid: PubMed ID
            
        Returns:
            Paper object or None
        """
        fetch_handle = Entrez.efetch(
            db="pubmed",
            id=pmid,
            rettype="xml",
            retmode="xml"
        )
        
        papers = self._parse_papers(fetch_handle.read())
        fetch_handle.close()
        
        return papers[0] if papers else None
    
    def as_langchain_tool(self):
        """Return as a LangChain tool."""
        @tool
        def pubmed_search(query: str, max_results: int = 10) -> str:
            """
            Search for biomedical research papers on PubMed.
            
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
                if paper.abstract:
                    result += f"   Abstract: {paper.abstract[:300]}...\n"
                result += "\n"
            
            return result
        
        return pubmed_search

