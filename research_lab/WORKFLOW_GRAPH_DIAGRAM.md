# 🔬 Complete Research Lab Workflow Graph

## Full LangGraph Workflow Visualization

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           START                                              │
└────────────────────────────┬────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   INIT NODE     │
                    │  - Initialize   │
                    │  - Set phase    │
                    │  - Setup state  │
                    └────────┬────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  WORKFLOW DECISION NODE       │
              │  - Check workflow_mode        │
              │  - Route to appropriate path  │
              └───────────────┬───────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  ROUTING NODE   │
                    │  - Analyze query │
                    │  - Select agents │
                    │  - Set active    │
                    │    domains      │
                    └────────┬────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │    DOMAIN RESEARCH NODE        │
              │  - Execute domain agents       │
              │  - Search papers (Arxiv,        │
              │    Semantic Scholar, PubMed)   │
              │  - Add papers to RAG           │
              │  - Generate domain reports    │
              └───────────────┬───────────────┘
                              │
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                             │
        │  Route based on workflow_mode              │
        │                                             │
        ▼                                             ▼
┌───────────────────────┐              ┌──────────────────────────┐
│  STRUCTURED MODE      │              │   AUTOMATED MODE         │
│  (Traditional)        │              │   (Hypothesis Generation) │
│                       │              │                          │
│  ┌─────────────────┐  │              │  ┌────────────────────┐  │
│  │ SUPPORT REVIEW  │  │              │  │ KNOWLEDGE GRAPH     │  │
│  │ - Literature    │  │              │  │ - Collect papers    │  │
│  │   Reviewer      │  │              │  │   from domain_res   │  │
│  │ - Fact Checker  │  │              │  │ - Build graph       │  │
│  │ - Methodology   │  │              │  │ - Sample path       │  │
│  │   Critic        │  │              │  │   (random/shortest)│  │
│  └────────┬────────┘  │              │  └──────────┬─────────┘  │
│           │           │              │             │             │
│           ▼           │              │             ▼             │
│  ┌─────────────────┐ │              │  ┌────────────────────┐  │
│  │   SYNTHESIS     │ │              │  │   ONTOLOGIST       │  │
│  │ - Combine       │ │              │  │   (Collaborative)  │  │
│  │   findings      │ │              │  │ - AI/ML agent      │  │
│  │ - Generate      │ │              │  │   contributes      │  │
│  │   academic      │ │              │  │ - Physics agent    │  │
│  │   paper         │ │              │  │   contributes      │  │
│  └────────┬────────┘ │              │  │ - Biology agent    │  │
│           │           │              │  │   contributes      │  │
│           ▼           │              │  │ - Merge all        │  │
│  ┌─────────────────┐ │              │  │   perspectives    │  │
│  │    COMPLETE     │ │              │  └──────────┬─────────┘  │
│  │ - Finalize      │ │              │             │             │
│  │ - Collect stats │ │              │             │             │
│  └────────┬────────┘ │              │             ▼             │
│           │           │              │  ┌────────────────────┐  │
│           └───────────┼──────────────┼─▶│ HYPOTHESIS GEN     │  │
│                       │              │  │ (Scientist_1)      │  │
│                       │              │  │ - 7-field JSON     │  │
│                       │              │  │ - Uses collaborative│  │
│                       │              │  │   ontology        │  │
│                       │              │  └──────────┬─────────┘  │
│                       │              │             │             │
│                       │              │             ▼             │
│                       │              │  ┌────────────────────┐  │
│                       │              │  │ HYPOTHESIS EXPAND  │  │
│                       │              │  │ (Scientist_2)      │  │
│                       │              │  │ - Quantitative      │  │
│                       │              │  │ - MD simulations  │  │
│                       │              │  │ - Experimental     │  │
│                       │              │  └──────────┬─────────┘  │
│                       │              │             │             │
│                       │              │             ▼             │
│                       │              │  ┌────────────────────┐  │
│                       │              │  │     CRITIQUE       │  │
│                       │              │  │ - Review strengths │  │
│                       │              │  │ - Identify weak    │  │
│                       │              │  │ - Rate novelty    │  │
│                       │              │  │ - Rate feasibility │  │
│                       │              │  └──────────┬─────────┘  │
│                       │              │             │             │
│                       │              │             ▼             │
│                       │              │  ┌────────────────────┐  │
│                       │              │  │      PLANNER       │  │
│                       │              │  │ - Research phases  │  │
│                       │              │  │ - Timeline         │  │
│                       │              │  │ - Resources        │  │
│                       │              │  │ - Risk assessment  │  │
│                       │              │  └──────────┬─────────┘  │
│                       │              │             │             │
│                       │              │             ▼             │
│                       │              │  ┌────────────────────┐  │
│                       │              │  │   NOVELTY CHECK    │  │
│                       │              │  │ - Semantic Scholar │  │
│                       │              │  │ - Compare with     │  │
│                       │              │  │   existing work    │  │
│                       │              │  │ - Novelty score    │  │
│                       │              │  └──────────┬─────────┘  │
│                       │              │             │             │
│                       │              │             ▼             │
│                       │              │  ┌────────────────────┐  │
│                       │              └─▶│  SUPPORT REVIEW    │  │
│                       │                 │  (Same as left)    │  │
│                       │                 └──────────┬─────────┘  │
│                       │                            │             │
│                       │                            ▼             │
│                       │                 ┌────────────────────┐  │
│                       └─────────────────▶│    SYNTHESIS      │  │
│                                         │  (Same as left)    │  │
│                                         └──────────┬─────────┘  │
│                                                    │             │
│                                                    ▼             │
│                                         ┌────────────────────┐  │
│                                         │     COMPLETE      │  │
│                                         │  (Same as left)   │  │
│                                         └──────────┬─────────┘  │
│                                                    │             │
└────────────────────────────────────────────────────┴─────────────┘
                                                      │
                                                      ▼
                                            ┌─────────────────┐
                                            │       END       │
                                            └─────────────────┘
```

---

## Node Details

### Core Nodes

#### 1. **INIT NODE**
- Initializes workflow state
- Sets timestamps
- Prepares phase tracking
- Creates node_outputs structure

#### 2. **WORKFLOW DECISION NODE**
- Checks `workflow_mode` (structured/automated)
- Routes to appropriate path
- Both paths go through routing

#### 3. **ROUTING NODE**
- Analyzes user query
- Selects relevant domain agents
- Sets `active_domain_agents`
- Determines parallel execution

#### 4. **DOMAIN RESEARCH NODE**
- Executes selected domain agents in parallel
- Each agent:
  - Searches papers (Arxiv, Semantic Scholar, PubMed)
  - Uses RAG for context
  - Extracts papers from tool results
  - Adds papers to RAG
  - Generates domain-specific report
- Collects all domain results

---

### Structured Mode Path

#### 5. **SUPPORT REVIEW NODE**
- Literature Reviewer: Systematic summarization
- Fact Checker: Verifies claims
- Methodology Critic: Evaluates methods
- Cross-Domain Synthesizer: Finds connections

#### 6. **SYNTHESIS NODE**
- Combines all domain findings
- Generates academic paper format:
  - Abstract
  - Introduction
  - Methodology
  - Findings (by domain)
  - Discussion
  - Conclusion
  - References

#### 7. **COMPLETE NODE**
- Finalizes workflow
- Collects statistics
- Stores final papers
- Calculates execution time

---

### Automated Mode Path (Hypothesis Generation)

#### 8. **KNOWLEDGE GRAPH NODE**
- Collects papers from `domain_results`
- Creates temporary vector store
- Builds graph from query-relevant papers
- Extracts entities and relationships
- Samples path (random for novelty)
- Uses query keywords for better sampling

#### 9. **ONTOLOGIST NODE** (Collaborative)
- Each domain agent contributes:
  - Field-specific concept definitions
  - Field-specific relationship explanations
  - Domain expertise insights
- Merges all contributions
- Creates collaborative ontology
- **Checkpoint**: User can review ontology

#### 10. **HYPOTHESIS GENERATION NODE** (Scientist_1)
- Uses collaborative ontology
- Generates 7-field JSON structure:
  1. hypothesis
  2. outcome (quantitative)
  3. mechanisms
  4. design_principles
  5. unexpected_properties
  6. comparison
  7. novelty
- Tracks field contributions
- **Checkpoint**: User can review hypothesis

#### 11. **HYPOTHESIS EXPANSION NODE** (Scientist_2)
- Expands each field with quantitative details
- Suggests modeling approaches:
  - MD simulations (GROMACS, AMBER)
  - DFT calculations
  - Experimental techniques
- Provides specific protocols
- Includes quantitative predictions

#### 12. **CRITIQUE NODE**
- Reviews hypothesis comprehensively
- Identifies strengths and weaknesses
- Suggests improvements
- Rates novelty (1-10)
- Rates feasibility (1-10)
- **Checkpoint**: User can review critique

#### 13. **PLANNER NODE**
- Creates research roadmap
- Breaks into phases with timelines
- Identifies resource requirements
- Risk assessment and mitigation
- Success criteria

#### 14. **NOVELTY CHECK NODE**
- Searches Semantic Scholar
- Compares with existing work
- Identifies novel aspects
- Lists similar papers
- Provides novelty score

#### 15. **SUPPORT REVIEW NODE** (Same as structured)
- Literature review
- Fact checking
- Methodology critique

#### 16. **SYNTHESIS NODE** (Same as structured)
- Combines hypothesis + domain research
- Generates comprehensive paper

#### 17. **COMPLETE NODE** (Same as structured)
- Finalizes workflow
- Collects all outputs

---

## State Flow

```
WorkflowState {
    messages: List[BaseMessage]
    current_query: ResearchQuery
    team_config: TeamConfiguration
    active_domain_agents: List[str]
    
    # Domain research results
    domain_results: List[ResearchResult]
    
    # Hypothesis generation (automated mode)
    knowledge_graph_path: Dict
    ontology: Dict (collaborative)
    hypothesis: Dict (7-field JSON)
    expanded_hypothesis: Dict
    critique: Dict
    research_plan: Dict
    novelty_assessment: Dict
    
    # Workflow control
    workflow_mode: "structured" | "automated"
    checkpoint_pending: str | None
    checkpoint_data: Dict | None
    user_approvals: Dict[str, bool]
    
    # Final output
    final_response: str
    final_papers: List[Paper]
    node_outputs: Dict[str, Dict]
}
```

---

## Checkpoints (Human-in-the-Loop)

1. **After Ontology**: User reviews collaborative ontology
2. **After Hypothesis**: User reviews generated hypothesis
3. **After Critique**: User reviews critique and ratings

Each checkpoint:
- Pauses workflow
- Displays output in UI
- Waits for user approval
- Can continue or request revision

---

## Key Features

### ✅ Dynamic Knowledge Graph
- Built from query-relevant papers
- Not from static RAG collection
- Uses papers found during domain research

### ✅ Collaborative Ontology
- Domain agents contribute field expertise
- Multiple perspectives merged
- Field-specific definitions

### ✅ Structured Hypothesis
- 7-field JSON structure
- Quantitative outputs enforced
- Collaborative generation

### ✅ Iterative Refinement
- Scientist_1 → Scientist_2 → Critic
- Each stage adds depth
- User checkpoints for control

### ✅ Dual Mode Support
- Structured: Reliable, traditional workflow
- Automated: Exploratory, hypothesis generation

---

## Execution Flow Example

**Query**: "How can graphene and silk fibroin be used for bioelectronics?"

1. **INIT** → Initialize state
2. **WORKFLOW DECISION** → Route to routing (both modes)
3. **ROUTING** → Select: AI/ML, Physics, Biology agents
4. **DOMAIN RESEARCH** → 
   - AI/ML: Finds bioelectronics papers
   - Physics: Finds graphene papers
   - Biology: Finds silk fibroin papers
   - All papers added to RAG
5. **Route Decision**:
   - **Structured**: → Support Review → Synthesis → Complete
   - **Automated**: → Knowledge Graph → Ontologist → ...
6. **KNOWLEDGE GRAPH** → Build from found papers, sample path
7. **ONTOLOGIST** → 
   - AI/ML: "Bioelectronics = neural interfaces..."
   - Physics: "Graphene = high conductivity..."
   - Biology: "Silk fibroin = biocompatible..."
   - Merge → Collaborative ontology
8. **HYPOTHESIS GENERATION** → 7-field hypothesis using ontology
9. **HYPOTHESIS EXPANSION** → Add MD simulations, experimental plans
10. **CRITIQUE** → Review, rate novelty/feasibility
11. **PLANNER** → Create research roadmap
12. **NOVELTY CHECK** → Verify against Semantic Scholar
13. **SUPPORT REVIEW** → Fact check, methodology review
14. **SYNTHESIS** → Generate comprehensive paper
15. **COMPLETE** → Finalize and return

---

## Graph Statistics

- **Total Nodes**: 17
- **Decision Points**: 2 (workflow_mode, after_domain_research)
- **Checkpoints**: 3 (ontology, hypothesis, critique)
- **Parallel Execution**: Domain agents run in parallel
- **Sequential Refinement**: Hypothesis generation chain

---

**Last Updated**: 2024
**Status**: ✅ Complete and Integrated

