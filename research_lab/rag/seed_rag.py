"""Automated RAG seeding - fetches top-cited papers from Semantic Scholar."""

import threading
from typing import List
from pathlib import Path
from tools.semantic_scholar import SemanticScholarTool
from states.agent_state import Paper
from config.settings import FIELD_DISPLAY_NAMES

# Global lock to prevent concurrent seeding of the same field
_seeding_locks: dict[str, threading.Lock] = {}
_seeding_lock_global = threading.Lock()


# Field-specific seed queries to fetch foundational papers
FIELD_SEED_QUERIES = {
    "ai_ml": [
        "machine learning survey review",
        "deep learning neural networks",
        "transformer attention mechanism",
        "natural language processing",
        "computer vision"
    ],
    "physics": [
        "quantum mechanics review",
        "general relativity",
        "particle physics",
        "condensed matter physics",
        "astrophysics cosmology"
    ],
    "biology": [
        "molecular biology review",
        "genetics genomics",
        "cell biology",
        "evolutionary biology",
        "systems biology"
    ],
    "chemistry": [
        "organic chemistry review",
        "physical chemistry",
        "biochemistry",
        "materials chemistry",
        "computational chemistry"
    ],
    "mathematics": [
        "mathematical analysis",
        "algebra topology",
        "statistics probability",
        "differential equations",
        "number theory"
    ],
    "neuroscience": [
        "neuroscience review",
        "cognitive neuroscience",
        "computational neuroscience",
        "neurobiology",
        "brain imaging"
    ],
    "medicine": [
        "clinical medicine review",
        "pharmacology",
        "pathology",
        "public health",
        "medical research"
    ],
    "computer_science": [
        "algorithms data structures",
        "computer systems",
        "software engineering",
        "distributed systems",
        "programming languages"
    ]
}


def fetch_seed_papers(field: str, num_papers: int = 10) -> List[Paper]:
    """
    Fetch foundational papers for a field from Semantic Scholar.
    
    Args:
        field: Research field
        num_papers: Target number of papers
        
    Returns:
        List of Paper objects
    """
    if field not in FIELD_SEED_QUERIES:
        return []
    
    queries = FIELD_SEED_QUERIES[field]
    tool = SemanticScholarTool()
    
    papers = []
    seen_paper_ids = set()
    remaining = num_papers
    
    for query in queries:
        if remaining <= 0:
            break
        
        try:
            results = tool.search(query, max_results=min(remaining + 5, 15))
            
            for paper in results:
                if remaining <= 0:
                    break
                
                # Skip duplicates
                paper_id = paper.id or paper.title.lower()
                if paper_id in seen_paper_ids:
                    continue
                seen_paper_ids.add(paper_id)
                
                # Update field
                paper.field = field
                
                papers.append(paper)
                remaining -= 1
        
        except Exception as e:
            print(f"Warning: Failed to fetch papers for query '{query}': {e}")
            continue
    
    return papers


def seed_rag_if_empty(vector_store, field: str, num_papers: int = 10) -> bool:
    """
    Seed RAG collection with foundational papers if it's empty.
    
    Uses a lock mechanism to prevent concurrent seeding of the same field.
    
    Args:
        vector_store: VectorStore instance to seed
        field: Research field
        num_papers: Number of seed papers to fetch
        
    Returns:
        True if seeding was performed, False if collection already had papers
    """
    # Get or create lock for this field
    with _seeding_lock_global:
        if field not in _seeding_locks:
            _seeding_locks[field] = threading.Lock()
        field_lock = _seeding_locks[field]
    
    # Check if collection is empty (with lock to prevent race conditions)
    with field_lock:
        # Double-check pattern: check count again inside lock
        if vector_store.count > 0:
            return False
        
        # Only seed domain fields that have seed queries defined
        if field not in FIELD_SEED_QUERIES:
            # Silently skip support agents or unknown fields
            return False
        
        print(f"[INFO] Seeding RAG for field '{field}' with foundational papers...")
        
        try:
            # Fetch seed papers
            seed_papers = fetch_seed_papers(field, num_papers)
            
            if not seed_papers:
                print(f"[WARNING] No seed papers found for field '{field}'")
                return False
            
            # Add papers to vector store
            added_count = 0
            for paper in seed_papers:
                try:
                    # Ensure paper has the correct field
                    paper.field = field
                    vector_store.add_paper(paper)
                    added_count += 1
                except Exception as e:
                    # Check if it's an embeddings API error (only relevant for OpenAI)
                    error_str = str(e)
                    from config.settings import settings
                    
                    if settings.embeddings_provider == "openai":
                        if "401" in error_str or "invalid_api_key" in error_str.lower() or "Incorrect API key" in error_str:
                            print(f"Warning: Failed to add paper '{paper.title}': Authentication error - check your embeddings API key")
                            # If it's an auth error, might be key issue - continue trying other papers
                        elif "404" in error_str or "Not Found" in error_str:
                            print(f"Warning: Failed to add paper '{paper.title}': Embeddings endpoint not found")
                            # If embeddings endpoint not found, stop trying
                            print(f"[WARNING] Embeddings API endpoint not available. Stopping RAG seeding for field '{field}'.")
                            break
                        else:
                            print(f"Warning: Failed to add paper '{paper.title}': {e}")
                    else:
                        # For BGE-M3, just log the error and continue
                        print(f"Warning: Failed to add paper '{paper.title}': {e}")
                    continue
            
            if added_count > 0:
                print(f"[SUCCESS] Seeded RAG with {added_count} papers for field '{field}'")
            else:
                print(f"[WARNING] No papers were successfully added to RAG for field '{field}'")
            
            return added_count > 0
            
        except Exception as e:
            print(f"[ERROR] Error seeding RAG for field '{field}': {e}")
            return False
