"""Automated RAG seeding - fetches top-cited papers from Semantic Scholar."""

from typing import List
from tools.semantic_scholar import SemanticScholarTool
from states.agent_state import Paper
from config.settings import FIELD_DISPLAY_NAMES


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
    Fetch top-cited foundational papers for a field from Semantic Scholar.
    
    Uses multiple queries to get diverse, high-quality papers, then sorts by
    citation count to get the most influential ones.
    
    Args:
        field: Research field (ai_ml, physics, etc.)
        num_papers: Target number of papers to fetch
        
    Returns:
        List of Paper objects sorted by citation count (highest first)
    """
    ss_tool = SemanticScholarTool()
    all_papers = []
    seen_ids = set()
    
    # Get seed queries for this field
    queries = FIELD_SEED_QUERIES.get(field, [])
    if not queries:
        # Fallback: use field name as query
        field_name = FIELD_DISPLAY_NAMES.get(field, field)
        queries = [field_name]
    
    # Fetch papers from multiple queries to get diversity
    papers_per_query = max(5, num_papers // len(queries) + 1)
    
    # Use top 3 queries to avoid too many API calls
    queries_to_use = queries[:min(3, len(queries))]
    
    for query in queries_to_use:
        try:
            papers = ss_tool.search_by_field(
                query=query,
                field=field,
                max_results=papers_per_query
            )
            
            # Add papers we haven't seen yet
            for paper in papers:
                if paper.id and paper.id not in seen_ids:
                    seen_ids.add(paper.id)
                    all_papers.append(paper)
        except Exception as e:
            # Continue if one query fails
            print(f"Warning: Failed to fetch seed papers for query '{query}': {e}")
            continue
    
    # Sort by citation count (highest first) and take top N
    all_papers.sort(key=lambda p: p.citations or 0, reverse=True)
    
    # Filter to ensure we have papers with abstracts (better for RAG)
    papers_with_abstracts = [p for p in all_papers if p.abstract and len(p.abstract) > 100]
    
    # Return top papers (prefer those with abstracts)
    if len(papers_with_abstracts) >= num_papers:
        return papers_with_abstracts[:num_papers]
    else:
        # Fill remaining slots with any papers
        result = papers_with_abstracts.copy()
        remaining = num_papers - len(result)
        for paper in all_papers:
            if paper not in result and remaining > 0:
                result.append(paper)
                remaining -= 1
        return result


def seed_rag_if_empty(vector_store, field: str, num_papers: int = 10) -> bool:
    """
    Seed RAG collection with foundational papers if it's empty.
    
    Args:
        vector_store: VectorStore instance to seed
        field: Research field
        num_papers: Number of seed papers to fetch
        
    Returns:
        True if seeding was performed, False if collection already had papers
    """
    # Check if collection is empty
    if vector_store.count > 0:
        return False
    
    # Only seed domain fields that have seed queries defined
    if field not in FIELD_SEED_QUERIES:
        # Silently skip support agents or unknown fields
        return False
    
    print(f"🌱 Seeding RAG for field '{field}' with foundational papers...")
    
    try:
        # Fetch seed papers
        seed_papers = fetch_seed_papers(field, num_papers)
        
        if not seed_papers:
            print(f"⚠️  No seed papers found for field '{field}'")
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
                        print(f"⚠️  Embeddings API endpoint not available. Stopping RAG seeding for field '{field}'.")
                        break
                    else:
                        print(f"Warning: Failed to add paper '{paper.title}': {e}")
                else:
                    # For BGE-M3, just log the error and continue
                    print(f"Warning: Failed to add paper '{paper.title}': {e}")
                continue
        
        if added_count > 0:
            print(f"✅ Seeded RAG with {added_count} papers for field '{field}'")
        else:
            print(f"⚠️  No papers were successfully added to RAG for field '{field}'")
        
        return added_count > 0
        
    except Exception as e:
        print(f"❌ Error seeding RAG for field '{field}': {e}")
        return False

