"""
Simple test runner to verify tests work.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, '.')

# Disable RAG seeding
os.environ["RAG_SEED_ENABLED"] = "false"

def run_basic_tests():
    """Run basic tests to verify the test suite works."""
    print("=" * 60)
    print("Running Basic Test Verification")
    print("=" * 60)
    
    # Test 1: Import test
    print("\n1. Testing imports...")
    try:
        from test_production_comprehensive import (
            TestWorkflowState,
            TestShortTermMemory,
            TestVectorStore
        )
        print("   ✅ All imports successful")
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False
    
    # Test 2: Test fixture creation
    print("\n2. Testing fixture creation...")
    try:
        from test_utils import (
            mock_paper,
            mock_team_config,
            initial_state
        )
        from states.agent_state import Paper, TeamConfiguration
        from states.workflow_state import create_initial_state
        
        # Create a paper
        paper = Paper(
            title="Test Paper",
            authors=["Author 1"],
            abstract="Test abstract",
            url="https://example.com",
            source="arxiv"
        )
        assert paper.title == "Test Paper"
        print("   ✅ Paper creation works")
        
        # Create team config
        config = TeamConfiguration(
            team_id="test",
            domain_agents=["ai_ml"]
        )
        assert config.is_valid
        print("   ✅ Team config creation works")
        
        # Create initial state
        state = create_initial_state("test", config, "structured")
        assert state["session_id"] == "test"
        print("   ✅ State creation works")
        
    except Exception as e:
        print(f"   ❌ Fixture test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Test class instantiation
    print("\n3. Testing test class instantiation...")
    try:
        from test_production_comprehensive import TestWorkflowState
        test_instance = TestWorkflowState()
        print("   ✅ Test class can be instantiated")
    except Exception as e:
        print(f"   ❌ Test class instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Run a simple test method
    print("\n4. Running a simple test method...")
    try:
        from test_production_comprehensive import TestWorkflowState
        from states.agent_state import TeamConfiguration
        from states.workflow_state import create_initial_state
        
        test_instance = TestWorkflowState()
        config = TeamConfiguration(
            team_id="test",
            domain_agents=["ai_ml", "physics"]
        )
        result = test_instance.test_create_initial_state(config)
        if result:
            print("   ✅ Test method executed successfully")
        else:
            print("   ⚠️  Test method returned False")
    except Exception as e:
        print(f"   ❌ Test method execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✅ All basic tests passed!")
    print("=" * 60)
    print("\nTo run full test suite with pytest:")
    print("  pytest test_production_comprehensive.py -v")
    print("\nTo run specific test class:")
    print("  pytest test_production_comprehensive.py::TestWorkflowState -v")
    
    return True

if __name__ == "__main__":
    success = run_basic_tests()
    sys.exit(0 if success else 1)

