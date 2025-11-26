"""
Component-by-component testing for the Research Lab system.
Run each test individually to identify any issues.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_banner(name: str):
    """Print a test banner."""
    print("\n" + "=" * 60)
    print(f"  TESTING: {name}")
    print("=" * 60)


def test_1_config():
    """Test configuration and settings."""
    test_banner("Configuration & Settings")
    try:
        from config.settings import settings, RESEARCH_FIELDS
        print(f"✓ Settings loaded successfully")
        print(f"  - OpenAI Model: {settings.openai_model}")
        print(f"  - ChromaDB Path: {settings.chroma_persist_directory}")
        print(f"  - Available Fields: {RESEARCH_FIELDS[:3]}...")
        
        # Check if API key is set
        if settings.openai_api_key and len(settings.openai_api_key) > 10:
            print(f"  - OpenAI API Key: ****{settings.openai_api_key[-4:]}")
        else:
            print("  ⚠ OpenAI API Key not set (set OPENAI_API_KEY env var)")
        
        return True
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False


def test_2_states():
    """Test Pydantic state models."""
    test_banner("State Models (Pydantic)")
    try:
        from states.agent_state import (
            Paper, ResearchQuery, AgentState, 
            ResearchResult, AgentStatus
        )
        from states.workflow_state import WorkflowState
        
        # Test Paper model
        paper = Paper(
            title="Test Paper",
            authors=["Author 1", "Author 2"],
            abstract="This is a test abstract.",
            url="https://example.com/paper",
            source="arxiv"
        )
        print(f"✓ Paper model: {paper.title}")
        
        # Test ResearchQuery
        query = ResearchQuery(
            query="What is machine learning?",
            field="ai"
        )
        print(f"✓ ResearchQuery model: {query.query[:30]}...")
        
        # Test AgentState
        state = AgentState(
            agent_id="test_agent",
            agent_type="domain",
            field="ai",
            display_name="AI Agent"
        )
        print(f"✓ AgentState model: {state.agent_id}")
        
        # Test ResearchResult
        result = ResearchResult(
            agent_id="test_agent",
            agent_field="ai",
            query="test query",
            summary="Test summary",
            confidence_score=0.85
        )
        print(f"✓ ResearchResult model: confidence={result.confidence_score}")
        
        # Test WorkflowState (TypedDict)
        print(f"✓ WorkflowState TypedDict imported")
        
        return True
    except Exception as e:
        print(f"✗ States test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_3_memory():
    """Test memory systems."""
    test_banner("Memory Systems")
    try:
        from memory.short_term import ShortTermMemory
        from memory.long_term import LongTermMemory
        
        # Test Short-term memory
        stm = ShortTermMemory(max_size=5, agent_id="test")
        stm.add_user_message("Hello")
        stm.add_assistant_message("Hi there!", agent_id="test")
        print(f"✓ ShortTermMemory: {stm.size} messages stored")
        
        # Test Long-term memory
        ltm = LongTermMemory(agent_id="test")
        print(f"✓ LongTermMemory initialized")
        
        return True
    except Exception as e:
        print(f"✗ Memory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_4_rag():
    """Test RAG components."""
    test_banner("RAG System")
    try:
        from rag.embeddings import get_embeddings_model
        from rag.vector_store import VectorStore
        
        # Test embeddings (requires API key)
        print("  Testing embeddings model...")
        try:
            embeddings = get_embeddings_model()
            print(f"✓ Embeddings model loaded")
        except Exception as e:
            print(f"  ⚠ Embeddings require OpenAI API key: {e}")
        
        # Test vector store
        vs = VectorStore(collection_name="test_collection")
        print(f"✓ VectorStore initialized: {vs.collection_name}")
        
        return True
    except Exception as e:
        print(f"✗ RAG test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_5_tools():
    """Test research tools."""
    test_banner("Research Tools")
    
    results = {}
    
    # Test Arxiv
    try:
        from tools.arxiv_tool import ArxivSearchTool
        arxiv = ArxivSearchTool()
        print(f"✓ ArxivSearchTool initialized")
        results['arxiv'] = True
    except Exception as e:
        print(f"✗ ArxivSearchTool failed: {e}")
        results['arxiv'] = False
    
    # Test Semantic Scholar
    try:
        from tools.semantic_scholar import SemanticScholarTool
        ss = SemanticScholarTool()
        print(f"✓ SemanticScholarTool initialized")
        results['semantic_scholar'] = True
    except Exception as e:
        print(f"✗ SemanticScholarTool failed: {e}")
        results['semantic_scholar'] = False
    
    # Test PubMed
    try:
        from tools.pubmed_tool import PubMedSearchTool
        pubmed = PubMedSearchTool()
        print(f"✓ PubMedSearchTool initialized")
        results['pubmed'] = True
    except Exception as e:
        print(f"✗ PubMedSearchTool failed: {e}")
        results['pubmed'] = False
    
    # Test Web Search
    try:
        from tools.web_search import WebSearchTool, ResearchToolkit
        ws = WebSearchTool()
        print(f"✓ WebSearchTool initialized")
        
        toolkit = ResearchToolkit()
        print(f"✓ ResearchToolkit initialized")
        results['web_search'] = True
    except Exception as e:
        print(f"✗ WebSearchTool failed: {e}")
        results['web_search'] = False
    
    return all(results.values())


def test_6_base_agent():
    """Test base agent class."""
    test_banner("Base Agent")
    try:
        from agents.base_agent import BaseResearchAgent
        print(f"✓ BaseResearchAgent class imported")
        print(f"  - Abstract class, cannot instantiate directly")
        return True
    except Exception as e:
        print(f"✗ BaseAgent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_7_domain_agents():
    """Test domain-specific agents."""
    test_banner("Domain Agents")
    
    agents_to_test = [
        ("agents.domain.ai_agent", "AIMLAgent"),
        ("agents.domain.physics_agent", "PhysicsAgent"),
        ("agents.domain.biology_agent", "BiologyAgent"),
        ("agents.domain.chemistry_agent", "ChemistryAgent"),
        ("agents.domain.mathematics_agent", "MathematicsAgent"),
        ("agents.domain.neuroscience_agent", "NeuroscienceAgent"),
        ("agents.domain.medicine_agent", "MedicineAgent"),
        ("agents.domain.cs_agent", "ComputerScienceAgent"),
    ]
    
    results = {}
    for module_name, class_name in agents_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            agent_class = getattr(module, class_name)
            print(f"✓ {class_name} imported (field: {agent_class.FIELD})")
            results[class_name] = True
        except Exception as e:
            print(f"✗ {class_name} failed: {e}")
            results[class_name] = False
    
    return all(results.values())


def test_8_support_agents():
    """Test support agents."""
    test_banner("Support Agents")
    
    agents_to_test = [
        ("agents.support.literature_reviewer", "LiteratureReviewer"),
        ("agents.support.methodology_critic", "MethodologyCritic"),
        ("agents.support.fact_checker", "FactChecker"),
        ("agents.support.writing_assistant", "WritingAssistant"),
        ("agents.support.cross_domain_synthesizer", "CrossDomainSynthesizer"),
    ]
    
    results = {}
    for module_name, class_name in agents_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            agent_class = getattr(module, class_name)
            print(f"✓ {class_name} imported")
            results[class_name] = True
        except Exception as e:
            print(f"✗ {class_name} failed: {e}")
            results[class_name] = False
    
    return all(results.values())


def test_9_orchestrator():
    """Test orchestrator agent."""
    test_banner("Orchestrator")
    try:
        from agents.orchestrator import Orchestrator
        print(f"✓ Orchestrator class imported")
        return True
    except Exception as e:
        print(f"✗ Orchestrator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_10_research_graph():
    """Test LangGraph workflow."""
    test_banner("Research Graph (LangGraph)")
    try:
        from graphs.research_graph import ResearchGraph, create_research_graph
        print(f"✓ ResearchGraph imported")
        print(f"✓ create_research_graph function available")
        return True
    except Exception as e:
        print(f"✗ Research Graph test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_11_ui_components():
    """Test UI components."""
    test_banner("UI Components")
    try:
        from ui.components import (
            render_header,
            render_paper_card,
            render_sidebar
        )
        print(f"✓ UI component functions imported")
        
        from ui.pages.home import render_home_page
        print(f"✓ Home page imported")
        
        from ui.pages.team_setup import render_team_setup_page
        print(f"✓ Team setup page imported")
        
        from ui.pages.research_session import render_research_session_page
        print(f"✓ Research session page imported")
        
        return True
    except Exception as e:
        print(f"✗ UI test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all component tests."""
    print("\n" + "=" * 60)
    print("  RESEARCH LAB - COMPONENT TESTS")
    print("=" * 60)
    
    tests = [
        ("1. Config & Settings", test_1_config),
        ("2. State Models", test_2_states),
        ("3. Memory Systems", test_3_memory),
        ("4. RAG System", test_4_rag),
        ("5. Research Tools", test_5_tools),
        ("6. Base Agent", test_6_base_agent),
        ("7. Domain Agents", test_7_domain_agents),
        ("8. Support Agents", test_8_support_agents),
        ("9. Orchestrator", test_9_orchestrator),
        ("10. Research Graph", test_10_research_graph),
        ("11. UI Components", test_11_ui_components),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"✗ {name} crashed: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, passed_test in results.items():
        status = "✓ PASS" if passed_test else "✗ FAIL"
        print(f"  {status} - {name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    print("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Research Lab components")
    parser.add_argument(
        "--test", "-t", 
        type=int, 
        choices=range(1, 12),
        help="Run specific test (1-11)"
    )
    args = parser.parse_args()
    
    if args.test:
        test_funcs = {
            1: test_1_config,
            2: test_2_states,
            3: test_3_memory,
            4: test_4_rag,
            5: test_5_tools,
            6: test_6_base_agent,
            7: test_7_domain_agents,
            8: test_8_support_agents,
            9: test_9_orchestrator,
            10: test_10_research_graph,
            11: test_11_ui_components,
        }
        test_funcs[args.test]()
    else:
        success = run_all_tests()
        sys.exit(0 if success else 1)

