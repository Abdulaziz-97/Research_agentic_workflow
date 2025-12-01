# 🔬 Research Lab – Complete System Documentation (Updated)

## Master Guide to the Multi‑Agent, Knowledge‑Graph‑Driven Research System

> This document describes the **current** production architecture, including:
> - Dual workflow modes (Structured vs Automated)
> - Knowledge Graph + Ontologist + Hypothesis workflow (SciAgents‑style)
> - New tools (`url_context`, PDF reader, Wikipedia API)
> - BGE‑M3 local embeddings + OpenAI/DeepSeek chat models
> - Human‑in‑the‑loop checkpoints in the Streamlit UI

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Deep Dive](#2-architecture-deep-dive)
3. [Core Concepts](#3-core-concepts)
4. [Configuration & Settings](#4-configuration--settings)
5. [State Management (Pydantic Models)](#5-state-management-pydantic-models)
6. [Memory Systems](#6-memory-systems)
7. [RAG System & Embeddings](#7-rag-system--embeddings)
8. [Knowledge Graph & Ontology Workflow](#8-knowledge-graph--ontology-workflow)
9. [Hypothesis Generation & Refinement](#9-hypothesis-generation--refinement)
10. [Research Tools (Arxiv, URL Context, PDF, etc.)](#10-research-tools-arxiv-url-context-pdf-etc)
11. [Agent System](#11-agent-system)
12. [Orchestrator](#12-orchestrator)
13. [LangGraph Workflows (Structured vs Automated)](#13-langgraph-workflows-structured-vs-automated)
14. [Streamlit UI & Human‑in‑the‑Loop Checkpoints](#14-streamlit-ui--human-in-the-loop-checkpoints)
15. [Data Flow](#15-data-flow)
16. [Extending the System](#16-extending-the-system)
17. [Troubleshooting & Diagnostics](#17-troubleshooting--diagnostics)

---

## 1. System Overview

### What is the Research Lab?

The Research Lab is a **multi‑agent AI research environment** that simulates a team of domain experts and support scientists. It can:

- Search and read **papers, URLs, and PDFs**
- Build a **dynamic knowledge graph** from query‑relevant papers
- Generate **structured hypotheses** and **research roadmaps**
- Run in either:
  - **Structured mode** – classic multi‑agent literature review and synthesis
  - **Automated mode** – SciAgents‑style hypothesis discovery and refinement

### Core Technologies

- **LLM / Chat**: `ChatOpenAI` wrapper, typically configured for **DeepSeek** via `OPENAI_BASE_URL=https://api.deepseek.com`
- **Embeddings**: 
  - **Default**: `BGE‑M3` (local, via `FlagEmbedding`, free, high quality)
  - Optional: OpenAI embeddings via API
- **Vector DB**: `ChromaDB` for paper storage, RAG, and long‑term memory
- **Workflow**: `LangGraph` for agent orchestration and branching
- **Agents**: Domain and support agents built on `LangChain`
- **UI**: `Streamlit` multi‑page app with workflow visualization
- **Knowledge Graph**: `networkx` graph built from **query‑relevant papers**

### Key Capabilities (Current System)

| Capability | Description |
|-----------|-------------|
| **Multi‑Domain Research** | 8 domain agents (AI/ML, Physics, Biology, Chemistry, Mathematics, Neuroscience, Medicine, CS) |
| **Support Agents** | Literature Reviewer, Methodology Critic, Fact Checker, Writing Assistant, Cross‑Domain Synthesizer |
| **RAG** | Retrieve‑Reflect‑Retry across field‑specific Chroma collections |
| **Knowledge Graph** | Built dynamically from papers found for the current query |
| **Ontology Generation** | Domain agents collaboratively act as ontologists |
| **Hypothesis Workflow** | Structured 7‑field hypothesis → expansion → critique → planner → novelty check |
| **Human‑in‑the‑Loop** | Checkpoints after ontology, hypothesis, and critique |
| **Tools** | Arxiv, Semantic Scholar, PubMed, Tavily web search, URL context (Gemini + scraping + Wikipedia API), PDF reader |

### Updated Project Structure

```text
research_lab/
├── app.py                      # Streamlit entry point
├── .env                        # Environment variables (API keys, provider settings)
├── requirements.txt            # Python dependencies
│
├── config/
│   └── settings.py             # Configuration & constants (OpenAI/DeepSeek, embeddings, Gemini, etc.)
│
├── states/
│   ├── agent_state.py          # Pydantic models for agents
│   └── workflow_state.py       # LangGraph workflow state (structured + automated modes)
│
├── memory/
│   ├── short_term.py           # Conversation buffer memory
│   └── long_term.py            # Persistent ChromaDB memory
│
├── rag/
│   ├── embeddings.py           # BGE‑M3 + OpenAI embeddings (singleton cache)
│   ├── vector_store.py         # ChromaDB vector store wrapper
│   ├── retriever.py            # Retrieve‑Reflect‑Retry logic
│   └── seed_rag.py             # Thread‑safe auto‑seeding of foundational papers
│
├── tools/
│   ├── arxiv_tool.py           # Arxiv paper search
│   ├── semantic_scholar.py     # Semantic Scholar API wrapper
│   ├── pubmed_tool.py          # PubMed/NCBI search
│   ├── web_search.py           # Tavily web search
│   ├── url_context.py          # URL content extraction (Gemini url_context + scraping + Wikipedia API)
│   └── pdf_reader.py           # PDF reading utility (PyMuPDF)
│
├── knowledge_graph/
│   └── service.py              # KnowledgeGraphService (build graph, sample paths, definitions)
│
├── agents/
│   ├── base_agent.py           # BaseResearchAgent (LLM, tools, memory, RAG, state)
│   ├── orchestrator.py         # Main coordinator (routing & synthesis)
│   ├── domain/                 # 8 domain‑specific agents
│   │   ├── ai_agent.py
│   │   ├── physics_agent.py
│   │   ├── biology_agent.py
│   │   ├── chemistry_agent.py
│   │   ├── mathematics_agent.py
│   │   ├── neuroscience_agent.py
│   │   ├── medicine_agent.py
│   │   └── cs_agent.py
│   └── support/                # Support + new SciAgents‑style agents
│       ├── literature_reviewer.py
│       ├── methodology_critic.py
│       ├── fact_checker.py
│       ├── writing_assistant.py
│       ├── cross_domain_synthesizer.py
│       ├── ontologist.py               # Graph path → ontology
│       ├── hypothesis_generator.py     # Scientist_1 (7‑field JSON)
│       ├── hypothesis_expander.py      # Scientist_2 (expansion, modeling, simulation)
│       ├── hypothesis_critic.py        # Critic (novelty/feasibility scoring)
│       ├── research_planner.py         # Planner (roadmap)
│       └── novelty_checker.py          # Novelty checker (Semantic Scholar)
│
├── graphs/
│   └── research_graph.py       # LangGraph workflow (both modes + checkpoints)
│
├── ui/
│   ├── components.py           # Reusable UI components
│   └── pages/
│       ├── home.py
│       ├── team_setup.py
│       └── research_session.py # Workflow mode toggle, step display, checkpoints
│
├── WORKFLOW_GRAPH_DIAGRAM.md        # Updated workflow diagrams & node descriptions
├── KNOWLEDGE_GRAPH_COMPREHENSIVE_GUIDE.md
├── KNOWLEDGE_GRAPH_IMPROVEMENTS.md
├── WORKFLOW_ERRORS_FIXED.md
├── ENV_FIX_GUIDE.md
└── DIAGNOSIS_REPORT.md              # Documented issues & fixes
```

---

## 2. Architecture Deep Dive

### Layered Architecture (Updated)

```text
┌─────────────────────────────────────────────────────────────────┐
│                      PRESENTATION LAYER                         │
│                       (Streamlit UI)                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │    Home     │  │ Team Setup  │  │   Research Session      │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                      ORCHESTRATION LAYER                        │
│                  (LangGraph + Orchestrator + Modes)             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ ResearchGraph                                           │    │
│  │  - Structured Mode: classic RAG → synthesis             │    │
│  │  - Automated Mode: RAG → KG → Ontology → Hypothesis     │    │
│  │ Orchestrator: routing + synthesis                       │    │
│  └─────────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│                        AGENT LAYER                              │
│  ┌───────────────────────┐  ┌────────────────────────────────┐  │
│  │    Domain Agents      │  │     Support + SciAgents‑Style  │  │
│  │  ┌─────┐ ┌─────┐     │  │  ┌─────────┐ ┌─────────────┐   │  │
│  │  │AI/ML│ │Phys │ ... │  │  │Lit Rev  │ │Fact Checker │   │  │
│  │  └─────┘ └─────┘     │  │  └─────────┘ └─────────────┘   │  │
│  │                      │  │  Ontologist, Hypothesis_*      │  │
│  └───────────────────────┘  └────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                      INTELLIGENCE LAYER                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  RAG System │  │   Memory    │  │    Chat LLM (DeepSeek)  │  │
│  │ (Retrieve-  │  │ Short+Long  │  │ or OpenAI via ChatOpenAI│  │
│  │  Reflect-   │  │   Term      │  │                         │  │
│  │  Retry)     │  │             │  │                         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                        TOOLS LAYER                              │
│  ┌─────────┐  ┌──────────────┐  ┌────────┐  ┌─────────────┐    │
│  │  Arxiv  │  │Semantic      │  │ PubMed │  │   Tavily    │    │
│  │         │  │Scholar       │  │        │  │ Web Search  │    │
│  ├─────────┴──────────────────────────────────────────────┤    │
│  │ URL Context (Gemini url_context + scraping + wiki API) │    │
│  │ PDF Reader (PyMuPDF)                                   │    │
│  └────────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│                       STORAGE LAYER                             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │             ChromaDB (Vector Database)                   │    │
│  │  - Field‑specific RAG collections                        │    │
│  │  - Long‑term memory collections per agent                │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Agent Hierarchy (Conceptual)

```
                         ┌──────────────────┐
                         │   ORCHESTRATOR   │
                         │  (Coordinator)   │
                         └────────┬─────────┘
                                  │
            ┌─────────────────────┼─────────────────────┐
            │                     │                     │
            ▼                     ▼                     ▼
     ┌────────────┐        ┌────────────┐        ┌────────────┐
     │  Domain    │        │  Domain    │        │  Domain    │
     │  Agent 1   │        │  Agent 2   │        │  Agent 3   │
     │ (AI/ML)    │        │ (Physics)  │        │ (Biology)  │
     └─────┬──────┘        └─────┬──────┘        └─────┬──────┘
           │                     │                     │
           │    Each has:        │                     │
           │    - Own RAG        │                     │
           │    - Own Memory     │                     │
           │    - Own Tools      │                     │
           └─────────────────────┴─────────────────────┘
                                  │
            ┌─────────────────────┼─────────────────────┐
            │         │           │           │         │
            ▼         ▼           ▼           ▼         ▼
       ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
       │Lit Rev │ │Method  │ │Fact    │ │Writing │ │Cross-  │
       │        │ │Critic  │ │Checker │ │Assist  │ │Domain  │
       └────────┘ └────────┘ └────────┘ └────────┘ └────────┘
                     SUPPORT AGENTS (Always Available)
```

---

## 3. Core Concepts

### 3.1 Agentic Workflow

An **agentic workflow** is a system where AI agents:
1. Receive tasks
2. Reason about how to accomplish them
3. Use tools to gather information
4. Collaborate with other agents
5. Produce synthesized outputs

### 3.2 LangChain Fundamentals

**LangChain** provides the building blocks:

```python
# Core components used in this system:

from langchain_openai import ChatOpenAI          # LLM wrapper
from langchain_core.prompts import ChatPromptTemplate  # Prompt management
from langchain_core.tools import tool            # Tool decorator
from langchain.agents import AgentExecutor       # Agent execution
```

### 3.3 LangGraph Concepts

**LangGraph** enables complex workflows:

```python
from langgraph.graph import StateGraph, END

# Key concepts:
# 1. State: TypedDict that flows through the graph
# 2. Nodes: Functions that transform state
# 3. Edges: Connections between nodes (can be conditional)
# 4. Checkpoints: State persistence between runs
```

### 3.4 RAG (Retrieval‑Augmented Generation)

RAG enhances LLM responses with external knowledge:

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   RETRIEVE   │ ──▶ │   REFLECT    │ ──▶ │    RETRY     │
│              │     │              │     │  (if needed) │
│ Search for   │     │ Is this      │     │ Refine query │
│ relevant     │     │ sufficient?  │     │ and search   │
│ documents    │     │ Relevant?    │     │ again        │
└──────────────┘     └──────────────┘     └──────────────┘
        │                    │                    │
        └────────────────────┴────────────────────┘
                             │
                             ▼
                    ┌──────────────┐
                    │   GENERATE   │
                    │   Response   │
                    └──────────────┘
```

---

## 4. Configuration & Settings (Updated)

### File: `config/settings.py`

> **Note:** The real `settings.py` includes more fields than shown here (embeddings, Gemini, etc.). This section summarizes the most important ones.

```python
from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):
    # --- Chat LLM configuration (OpenAI / DeepSeek via ChatOpenAI wrapper) ---
    openai_api_key: str = ""             # Your DeepSeek/OpenAI API key
    openai_base_url: str = ""            # e.g. https://api.deepseek.com
    openai_model: str = "deepseek-reasoner"
    
    # --- Embeddings configuration ---
    embeddings_provider: Literal["bge-m3", "openai"] = "bge-m3"
    openai_embeddings_api_key: str = ""  # Only if embeddings_provider='openai'
    openai_embeddings_base_url: str = "https://api.openai.com/v1"
    openai_embeddings_model: str = "text-embedding-3-small"
    bge_m3_model_name: str = "BAAI/bge-m3"
    bge_m3_use_fp16: bool = True
    
    # --- Gemini / URL context ---
    gemini_api_key: str = ""             # For optional Gemini url_context integration
    
    # --- Tavily Web Search ---
    tavily_api_key: str = ""
    
    # --- ChromaDB ---
    chroma_persist_directory: str = "./data/chroma_db"
    chroma_collection_prefix: str = "research_lab"
    
    # --- Memory Settings ---
    short_term_memory_size: int = 10
    long_term_memory_threshold: float = 0.7
    
    # --- Research Tools ---
    arxiv_max_results: int = 10
    semantic_scholar_max_results: int = 10
    pubmed_max_results: int = 10
```

### Environment Variables (`.env`)

Typical DeepSeek + BGE‑M3 setup:

```bash
# Chat LLM (DeepSeek via OpenAI-compatible API)
OPENAI_API_KEY=sk-your-deepseek-key
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-reasoner

# Embeddings (local, free)
EMBEDDINGS_PROVIDER=bge-m3
# DO NOT set OPENAI_EMBEDDINGS_API_KEY / OPENAI_EMBEDDINGS_BASE_URL when using BGE-M3

# Tavily Web Search
TAVILY_API_KEY=your-tavily-key

# ChromaDB
CHROMA_PERSIST_DIRECTORY=./data/chroma_db
CHROMA_COLLECTION_PREFIX=research_lab
```

### Research Fields Configuration

```python
RESEARCH_FIELDS = [
    "ai_ml",           # AI/Machine Learning
    "physics",         # Physics
    "biology",         # Biology
    "chemistry",       # Chemistry
    "mathematics",     # Mathematics
    "neuroscience",    # Neuroscience
    "medicine",        # Medicine
    "computer_science" # Computer Science
]

FIELD_DISPLAY_NAMES = {
    "ai_ml": "AI/Machine Learning",
    "physics": "Physics",
    # ... etc
}
```

---

## 5. State Management (Pydantic Models)

### File: `states/agent_state.py`

Pydantic models ensure type safety and validation throughout the system.

### 5.1 Paper Model

```python
class Paper(BaseModel):
    """Represents a research paper."""
    
    id: str                          # Unique identifier
    title: str                       # Paper title
    authors: List[str] = []          # Author names
    abstract: str = ""               # Paper abstract
    url: str = ""                    # Link to paper
    source: str = ""                 # arxiv, pubmed, etc.
    published_date: Optional[datetime] = None
    citations: int = 0               # Citation count
    relevance_score: float = 0.0     # 0-1 relevance to query
    field: str = ""                  # Research field
```

**Usage:**
```python
paper = Paper(
    title="Attention Is All You Need",
    authors=["Vaswani, A.", "Shazeer, N."],
    source="arxiv",
    citations=50000,
    relevance_score=0.95
)
```

### 5.2 Research Query Model

```python
class ResearchQuery(BaseModel):
    """Represents a user's research query."""
    
    id: str                              # Auto-generated UUID
    query: str                           # The actual question
    field: Optional[str] = None          # Target field (or None for auto-detect)
    priority: int = 1                    # 1-5 priority level
    sources_required: List[str] = []     # Required sources
    max_papers: int = 10                 # Max papers to retrieve
    timestamp: datetime                  # When query was created
```

### 5.3 Agent State Model

```python
class AgentStatus(str, Enum):
    """Possible agent states."""
    IDLE = "idle"
    RESEARCHING = "researching"
    REFLECTING = "reflecting"
    RESPONDING = "responding"
    WAITING = "waiting"
    ERROR = "error"

class AgentState(BaseModel):
    """Tracks an agent's current state."""
    
    agent_id: str                    # Unique identifier
    agent_type: str                  # "domain" or "support"
    field: str                       # Research field
    display_name: str                # Human-readable name
    
    status: AgentStatus = AgentStatus.IDLE
    current_task: Optional[str] = None
    confidence_score: float = 0.0
    retrieved_papers: List[Paper] = []
    reflection_notes: List[str] = []
    
    def update_status(self, status: AgentStatus, task: str = None):
        """Update agent status and optionally current task."""
        self.status = status
        if task:
            self.current_task = task
```

### 5.4 Research Result Model

```python
class ResearchResult(BaseModel):
    """Complete research result from an agent."""
    
    agent_id: str                    # Which agent produced this
    agent_field: str                 # Agent's research field
    query: str                       # Original query
    summary: str                     # Main research summary
    papers: List[Paper] = []         # Papers found
    insights: List[str] = []         # Key insights extracted
    confidence_score: float = 0.0    # 0-1 confidence
    sources_used: List[str] = []     # Which sources were used
    reflection_notes: List[str] = [] # Agent's reflections
    
    def to_markdown(self) -> str:
        """Convert result to formatted markdown."""
        # Returns formatted output for display
```

### 5.5 Team Configuration Model

```python
class TeamConfiguration(BaseModel):
    """Configuration for a research team."""
    
    team_id: str                     # Unique team ID
    name: str = "Research Team"      # Team name
    domain_agents: List[str]         # Selected domains (max 3)
    support_agents: List[str]        # Always includes all 5
    
    @field_validator('domain_agents')
    def validate_domain_count(cls, v):
        if len(v) > 3:
            raise ValueError("Maximum 3 domain agents allowed")
        return v
```

### File: `states/workflow_state.py`

`WorkflowState` is a `TypedDict` that captures **both** the classic structured workflow and the new automated SciAgents‑style pipeline.

```python
from typing import TypedDict, List, Dict, Any, Optional, Literal, Annotated
from langchain_core.messages import BaseMessage

class WorkflowState(TypedDict, total=False):
    """Main state for the LangGraph research workflow."""
    
    # Messages (LangGraph add_messages aggregation)
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Query + team
    current_query: Optional[ResearchQuery]
    team_config: Optional[TeamConfiguration]
    
    # Agent states & results
    agent_states: Dict[str, AgentState]
    domain_results: List[ResearchResult]
    support_results: Dict[str, ResearchResult]
    
    # Workflow control
    current_phase: str
    active_domain_agents: List[str]
    active_support_agents: List[str]
    routing_reasoning: str
    
    # Final output
    final_response: Optional[str]
    final_papers: List[Paper]
    
    # Research stats & per‑phase details
    research_stats: Dict[str, Any]
    phase_details: Dict[str, Any]
    
    # Node outputs (for Streamlit step‑by‑step display)
    node_outputs: Dict[str, Dict[str, Any]]
    
    # Hypothesis workflow (automated mode)
    knowledge_graph_path: Optional[Dict[str, Any]]   # Sampled KG path
    ontology: Optional[Dict[str, Any]]               # Ontology from Ontologist
    hypothesis: Optional[Dict[str, Any]]             # Initial hypothesis
    expanded_hypothesis: Optional[Dict[str, Any]]    # Expanded hypothesis
    critique: Optional[Dict[str, Any]]               # Critique results
    research_plan: Optional[Dict[str, Any]]          # Planner output
    novelty_assessment: Optional[Dict[str, Any]]     # Novelty check
    
    # Workflow mode
    workflow_mode: Literal["structured", "automated"]
    
    # Human‑in‑the‑loop checkpoints
    checkpoint_pending: Optional[str]        # e.g. "ontology", "hypothesis", "critique"
    checkpoint_data: Optional[Dict[str, Any]]
    user_approvals: Dict[str, bool]
```

---

## 6. Memory Systems

### 6.1 Short-Term Memory

**File:** `memory/short_term.py`

Short-term memory holds the recent conversation context:

```python
class ConversationTurn(BaseModel):
    """Single conversation message."""
    id: str
    role: str           # "user" or "assistant"
    content: str        # Message content
    agent_id: str       # Which agent responded
    timestamp: datetime

class ShortTermMemory:
    """Sliding window conversation buffer."""
    
    def __init__(self, max_size: int = 10, agent_id: str = "default"):
        self.max_size = max_size
        self._buffer: deque[ConversationTurn] = deque(maxlen=max_size)
    
    def add_user_message(self, content: str):
        """Add user message to memory."""
        
    def add_assistant_message(self, content: str, agent_id: str):
        """Add assistant response to memory."""
    
    def get_messages(self, limit: int = None) -> List[ConversationTurn]:
        """Get recent messages."""
    
    def get_langchain_messages(self) -> List[BaseMessage]:
        """Convert to LangChain message format for prompts."""
    
    def get_context_string(self) -> str:
        """Get formatted conversation history for prompts."""
```

**How it's used:**

```python
# In an agent's research method:
self._short_term.add_user_message(query.query)
# ... agent processes ...
self._short_term.add_assistant_message(response, agent_id=self.agent_id)

# When building prompts:
chat_history = self._short_term.get_langchain_messages(limit=5)
```

### 6.2 Long-Term Memory

**File:** `memory/long_term.py`

Long-term memory persists important insights using vector similarity:

```python
class LongTermMemory:
    """Persistent memory using ChromaDB."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self._embeddings = EmbeddingManager()
        self._collection = chromadb.PersistentClient().get_or_create_collection(
            name=f"memory_{agent_id}"
        )
    
    def store(self, content: str, memory_type: str, 
              importance: float, query: str = None):
        """Store a memory with its embedding."""
        embedding = self._embeddings.embed_query(content)
        self._collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[{...}]
        )
    
    def store_insight(self, insight: str, query: str, confidence: float):
        """Store a research insight."""
    
    def store_paper_summary(self, paper: Paper, summary: str):
        """Store a paper summary for future reference."""
    
    def recall(self, query: str, limit: int = 5) -> List[MemoryEntry]:
        """Recall relevant memories using semantic search."""
        query_embedding = self._embeddings.embed_query(query)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=limit
        )
        return [MemoryEntry(...) for r in results]
```

**Memory Flow:**

```
User Query ──▶ Check Long-Term Memory ──▶ Relevant memories found?
                      │                            │
                      │                     ┌──────┴──────┐
                      │                     │ Yes         │ No
                      │                     ▼             ▼
                      │              Add to context   Fresh search
                      │                     │             │
                      ▼                     └──────┬──────┘
              Agent processes                      │
                      │                            │
                      ▼                            ▼
              Store important           Include in response
              insights in LTM
```

---

## 7. RAG System & Embeddings

### 7.1 Embeddings (`rag/embeddings.py`)

The embeddings layer now supports:

- **BGE‑M3** (default, local, free)
- OpenAI embeddings (optional, API‑based)
- A **singleton cache** so BGE‑M3 only loads **once**, even with many agents.

```python
class OpenAIEmbeddingsWrapper(BaseEmbeddings):
    # Wraps langchain_openai OpenAIEmbeddings with base URL support
    ...

class BGEM3Embeddings(BaseEmbeddings):
    # Wraps FlagEmbedding BGE-M3 model
    ...

# Global singleton cache
_embeddings_cache: dict[str, BaseEmbeddings] = {}

def get_embeddings_model(model: Optional[str] = None) -> BaseEmbeddings:
    """
    Get an embeddings model based on settings.embeddings_provider.
    Uses a singleton pattern so BGE-M3 loads only once.
    """
    provider = settings.embeddings_provider
    
    if provider == "bge-m3":
        cache_key = f"bge-m3_{settings.bge_m3_model_name}_{settings.bge_m3_use_fp16}"
    else:
        cache_key = f"openai_{model or settings.openai_embeddings_model}"
    
    if cache_key in _embeddings_cache:
        return _embeddings_cache[cache_key]
    
    if provider == "bge-m3":
        embeddings = BGEM3Embeddings(
            model_name=settings.bge_m3_model_name,
            use_fp16=settings.bge_m3_use_fp16,
        )
    else:
        embeddings = OpenAIEmbeddingsWrapper(
            model=model or settings.openai_embeddings_model,
        )
    
    _embeddings_cache[cache_key] = embeddings
    return embeddings

class EmbeddingManager:
    """Thin wrapper that uses get_embeddings_model()."""
    
    def __init__(self, model: Optional[str] = None):
        self._model = get_embeddings_model(model)
    
    def embed_query(self, text: str) -> List[float]:
        return self._model.embed_query(text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._model.embed_documents(texts)
```

### 7.2 Vector Store (`rag/vector_store.py`)

**File:** `rag/vector_store.py`

```python
class VectorStore:
    """
    ChromaDB vector store for research documents.
    """
    
    def __init__(self, collection_name: str):
        self._client = chromadb.PersistentClient(
            path=settings.chroma_persist_directory
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name
        )
        self.embedding_manager = EmbeddingManager()
    
    def add_document(self, content: str,
                     doc_id: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a single document (with robust error handling)."""
        doc_id = doc_id or str(uuid.uuid4())
        
        try:
            embedding = self.embedding_manager.embed_query(content)
        except Exception as e:
            # If using OpenAI embeddings, convert 401/404 into clearer errors
            error_msg = str(e)
            if settings.embeddings_provider == "openai":
                ...
            # For BGE-M3, re-raise original error
            raise
        
        metadata = metadata or {}
        metadata["added_at"] = datetime.now().isoformat()
        
        self._collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata],
        )
        return doc_id
```

### 7.3 Retrieve‑Reflect‑Retry Pattern (`rag/retriever.py`)

**File:** `rag/retriever.py`

This is the core RAG pattern that makes retrieval intelligent:

```python
class RetrieveReflectRetryRAG:
    """
    Implements the Retrieve-Reflect-Retry pattern:
    
    1. RETRIEVE: Get documents from vector store
    2. REFLECT: Use LLM to assess if documents are sufficient
    3. RETRY: If not sufficient, refine query and search again
    """
    
    def __init__(self, vector_store: VectorStore, field: str, 
                 max_retries: int = 3, min_confidence: float = 0.7):
        self.vector_store = vector_store
        self.max_retries = max_retries
        self.min_confidence = min_confidence
        self._llm = ChatOpenAI(...)
        
        self._reflection_prompt = ChatPromptTemplate.from_messages([
            ("system", """Evaluate if retrieved documents are sufficient.
            Consider: Relevance, Completeness, Quality, Recency.
            
            Respond with:
            - SUFFICIENT: if documents adequately answer the query
            - INSUFFICIENT: <reason> if more info needed
            - REFINE: <new_query> if query should be modified"""),
            ("human", "Query: {query}\n\nDocuments:\n{documents}")
        ])
    
    def retrieve_with_reflection(self, query: str) -> Tuple[List, float]:
        """Main retrieval method with reflection loop."""
        
        for attempt in range(self.max_retries):
            # 1. RETRIEVE
            documents = self.vector_store.search(query)
            
            if not documents:
                continue
            
            # 2. REFLECT
            reflection = self._reflect(query, documents)
            
            if reflection.startswith("SUFFICIENT"):
                return documents, self._calculate_confidence(documents)
            
            elif reflection.startswith("REFINE:"):
                # 3. RETRY with refined query
                query = reflection.replace("REFINE:", "").strip()
            
            # If INSUFFICIENT, try again with same query
        
        return documents, 0.5  # Return what we have with lower confidence
    
    def _reflect(self, query: str, documents: List) -> str:
        """Use LLM to reflect on document quality."""
        chain = self._reflection_prompt | self._llm | StrOutputParser()
        return chain.invoke({
            "query": query,
            "documents": self._format_docs(documents)
        })
```

**Visual Flow:**

```
        ┌─────────────────┐
        │   User Query    │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │    RETRIEVE     │◀─────────────────┐
        │  Search Vector  │                  │
        │     Store       │                  │
        └────────┬────────┘                  │
                 │                           │
                 ▼                           │
        ┌─────────────────┐                  │
        │    REFLECT      │                  │
        │  LLM evaluates  │                  │
        │   sufficiency   │                  │
        └────────┬────────┘                  │
                 │                           │
        ┌────────┴────────┐                  │
        │                 │                  │
   SUFFICIENT        INSUFFICIENT/REFINE     │
        │                 │                  │
        ▼                 └──────────────────┘
  Return Results              (RETRY)
```

---

## 8. Knowledge Graph & Ontology Workflow

### 8.1 Knowledge Graph Service (`knowledge_graph/service.py`)

The **KnowledgeGraphService** builds a graph **from papers relevant to the current query**, not from a static corpus.

Key ideas:

- **Nodes**: scientific concepts/entities
- **Edges**: relationships between concepts
- **Construction**:
  - Use vector store collection of papers (temporary per session)
  - For each paper: extract entities + relationships via LLM (with fallback)
- **Path Sampling**:
  - Random and shortest path sampling
  - Used as input to the Ontologist and Hypothesis Generator

```python
class KnowledgeGraphService:
    def __init__(self, vector_store: VectorStore, field: Optional[str] = None):
        self.vector_store = vector_store
        self.field = field
        self.graph = nx.MultiDiGraph()
        self.embedding_manager = EmbeddingManager()
        self._built = False
    
    def build_graph(self, max_papers: Optional[int] = None,
                    min_entities_per_paper: int = 3) -> Dict[str, Any]:
        """
        Build knowledge graph from papers in the vector store.
        """
        # reads all docs with doc_type='paper' (optionally filtered by field)
        ...
    
    def sample_path(self, source: Optional[str] = None,
                    target: Optional[str] = None,
                    path_type: str = "random",
                    max_length: int = 10,
                    random_waypoints: int = 2) -> PathSamplingResult:
        """
        Sample a path through the knowledge graph (random or shortest).
        Returns PathSamplingResult with:
          - nodes
          - edges
          - subgraph JSON
        """
        ...
```

The extraction step uses an LLM prompt that returns:

```json
{
  "entities": ["graphene", "silk fibroin", "neural interfaces"],
  "relationships": [
    ["graphene", "combined_with", "silk fibroin"],
    ["graphene-silk composite", "enables", "neural interfaces"]
  ]
}
```

If the LLM fails, a fallback regex‑based extractor finds capitalized phrases as a minimal baseline.

### 8.2 Graph Usage in the Workflow (`graphs/research_graph.py`)

The **knowledge graph node** is only used in **automated** mode:

```python
async def _knowledge_graph_node(self, state: WorkflowState) -> WorkflowState:
    state["current_phase"] = "knowledge_graph"
    
    # 1. Collect all papers from domain research results
    all_papers = []
    for result in state.get("domain_results", []):
        ...
    
    if not all_papers:
        raise ValueError("No papers found from domain research. Cannot build knowledge graph.")
    
    # 2. Build temporary vector store with these papers
    temp_collection = f"temp_kg_{state['session_id']}"
    vector_store = VectorStore(collection_name=temp_collection)
    for paper in all_papers:
        vector_store.add_paper(paper)
    
    # 3. Build graph & sample path
    kg_service = KnowledgeGraphService(vector_store=vector_store, field=None)
    stats = kg_service.build_graph(max_papers=len(all_papers))
    
    # 4. Use query keywords as source/target seeds when possible
    ...
    path_result = kg_service.sample_path(source=source, target=target, ...)
    
    state["knowledge_graph_path"] = {
        "path": path_result.path.nodes,
        "edges": path_result.path.edges,
        "subgraph": path_result.path.subgraph,
        "stats": stats,
        "papers_used": len(all_papers),
    }
    
    # 5. Store human‑readable output for UI
    state["node_outputs"]["knowledge_graph"] = {...}
    return state
```

This path is then passed to the Ontologist node.

### 8.3 Ontologist Agent (`agents/support/ontologist.py`)

The Ontologist:

- Receives the **sampled graph path** and subgraph
- Each **domain agent** contributes field‑specific interpretations
- Produces a **structured ontology** used as context for hypothesis generation

The ontology is stored in `state["ontology"]` and surfaced in the UI at a checkpoint.

---

## 9. Hypothesis Generation & Refinement

This is the SciAgents‑style pipeline implemented with dedicated agents.

### 9.1 Hypothesis Generator (`agents/support/hypothesis_generator.py`)

- Takes `ontology` + `knowledge_graph_path` + original query
- Produces **strict 7‑field JSON** hypothesis, including:
  - Problem statement
  - Mechanistic explanation
  - Key variables
  - Quantitative predictions
  - Experimental strategy

Output is stored as `state["hypothesis"]`.

### 9.2 Hypothesis Expander (`agents/support/hypothesis_expander.py`)

- Extends the hypothesis with:
  - Specific **modeling / simulation** suggestions (e.g., MD simulations)
  - Tools (e.g., GROMACS, AMBER, LAMMPS)
  - Experimental priorities
  - More detailed quantitative predictions

Stored as `state["expanded_hypothesis"]`.

### 9.3 Hypothesis Critic (`agents/support/hypothesis_critic.py`)

- Reviews the (expanded) hypothesis
- Rates:
  - **Novelty**
  - **Feasibility**
  - **Impact**
- Identifies weaknesses and potential failure modes
- Suggests refinements

Stored as `state["critique"]`.

### 9.4 Planner & Novelty Checker

- **ResearchPlanner**:
  - Turns the hypothesis + critique into a **research roadmap**
  - Contains phases, milestones, and resource hints
  - Stored as `state["research_plan"]`

- **NoveltyChecker**:
  - Uses Semantic Scholar to check if closely related work exists
  - Rates novelty and lists similar papers
  - Stored as `state["novelty_assessment"]`

### 9.5 Human‑in‑the‑Loop Checkpoints

The workflow pauses at:

1. After **ontology generation**
2. After **initial hypothesis**
3. After **critique**

At each point:

- `state["checkpoint_pending"]` is set (e.g., `"ontology"`)
- `state["checkpoint_data"]` holds the relevant object
- The Streamlit UI shows a card with:
  - Ontology / Hypothesis / Critique
  - **Accept & Continue** button
- When the user accepts:
  - The UI sets `user_approvals[checkpoint_name] = True`
  - The graph execution resumes

---

## 10. Research Tools (Arxiv, URL Context, PDF, etc.)

### 8.1 Tool Architecture

Each tool follows this pattern:

```python
class ResearchTool:
    """Base pattern for research tools."""
    
    def search(self, query: str, max_results: int) -> List[Paper]:
        """Execute search and return Paper objects."""
        
    def as_langchain_tool(self):
        """Return as LangChain tool for agent use."""
        @tool
        def tool_function(query: str) -> str:
            results = self.search(query)
            return self._format_results(results)
        return tool_function
```

### 8.2 Arxiv Tool

**File:** `tools/arxiv_tool.py`

```python
class ArxivSearchTool:
    """Search arXiv for research papers."""
    
    def __init__(self):
        self._client = arxiv.Client()
    
    def search(self, query: str, max_results: int = 10) -> List[Paper]:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        papers = []
        for result in self._client.results(search):
            papers.append(Paper(
                title=result.title,
                authors=[a.name for a in result.authors],
                abstract=result.summary,
                url=result.pdf_url,
                source="arxiv",
                published_date=result.published
            ))
        return papers
```

**Best for:** AI/ML, Physics, Mathematics, Computer Science papers.

### 8.3 Semantic Scholar Tool

**File:** `tools/semantic_scholar.py`

```python
class SemanticScholarTool:
    """Search Semantic Scholar for academic papers."""
    
    API_URL = "https://api.semanticscholar.org/graph/v1"
    
    def search(self, query: str, max_results: int = 10) -> List[Paper]:
        response = requests.get(f"{self.API_URL}/paper/search", params={
            "query": query,
            "limit": max_results,
            "fields": "title,authors,abstract,citationCount,url"
        })
        
        papers = []
        for item in response.json().get("data", []):
            papers.append(Paper(
                title=item["title"],
                citations=item.get("citationCount", 0),
                # ... other fields
            ))
        return papers
```

**Best for:** Cross-domain academic search, citation-aware retrieval.

### 8.4 PubMed Tool

**File:** `tools/pubmed_tool.py`

```python
class PubMedSearchTool:
    """Search PubMed/NCBI for biomedical literature."""
    
    def __init__(self):
        from Bio import Entrez
        Entrez.email = "research@lab.ai"  # Required by NCBI
    
    def search(self, query: str, max_results: int = 10) -> List[Paper]:
        # Search for IDs
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        record = Entrez.read(handle)
        ids = record["IdList"]
        
        # Fetch details
        handle = Entrez.efetch(db="pubmed", id=ids, rettype="xml")
        records = Entrez.read(handle)
        
        # Parse into Paper objects
        # ...
```

**Best for:** Medicine, Biology, Neuroscience, Chemistry (biomedical focus).

### 8.5 Tavily Web Search

**File:** `tools/web_search.py`

```python
class WebSearchTool:
    """Web search using Tavily API."""
    
    TAVILY_API_URL = "https://api.tavily.com/search"
    
    def __init__(self):
        self.api_key = settings.tavily_api_key
    
    def search(self, query: str, max_results: int = 10) -> List[WebSearchResult]:
        response = requests.post(self.TAVILY_API_URL, json={
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "advanced",
            "include_answer": True
        })
        
        results = []
        for item in response.json().get("results", []):
            results.append(WebSearchResult(
                title=item["title"],
                url=item["url"],
                snippet=item["content"],
                score=item["score"]
            ))
        return results
    
    def search_academic(self, query: str) -> List[WebSearchResult]:
        """Focus on academic domains."""
        return self.search(
            query,
            include_domains=["arxiv.org", "nature.com", "science.org", ...]
        )
```

**Best for:** Recent news, general knowledge, when papers aren't available.

### 10.6 Research Toolkit (Unified Interface)

```python
class ResearchToolkit:
    """Combined toolkit providing field-appropriate tools."""
    
    def __init__(self):
        self.arxiv = ArxivSearchTool()
        self.semantic_scholar = SemanticScholarTool()
        self.pubmed = PubMedSearchTool()
        self.web = WebSearchTool()
    
    def get_tools_for_field(self, field: str) -> List:
        """Get appropriate tools for a research field."""
        tools = [self.web.as_langchain_tool()]  # Always include
        
        if field in ["ai_ml", "physics", "mathematics", "computer_science"]:
            tools.append(self.arxiv.as_langchain_tool())
            tools.append(self.semantic_scholar.as_langchain_tool())
        
        if field in ["biology", "medicine", "neuroscience", "chemistry"]:
            tools.append(self.pubmed.as_langchain_tool())
            tools.append(self.semantic_scholar.as_langchain_tool())
        
        return tools
```

---

## 11. Agent System

### 9.1 Base Research Agent

**File:** `agents/base_agent.py`

The `BaseResearchAgent` is the foundation for all agents:

```python
class BaseResearchAgent(ABC):
    """
    Abstract base class providing:
    - LLM integration
    - Tool management
    - Memory (short + long term)
    - RAG system
    - State tracking
    """
    
    # Override in subclasses
    FIELD: str = "general"
    DISPLAY_NAME: str = "Research Agent"
    AGENT_TYPE: str = "domain"  # or "support"
    
    def __init__(self, agent_id: str = None, tools: List = None):
        # Generate unique ID
        self.agent_id = agent_id or f"{self.FIELD}_{uuid.uuid4().hex[:8]}"
        
        # Initialize LLM (supports custom base URL)
        llm_kwargs = {
            "model": settings.openai_model,
            "temperature": 0.7,
            "openai_api_key": settings.openai_api_key
        }
        if settings.openai_base_url:
            llm_kwargs["openai_api_base"] = settings.openai_base_url
        self._llm = ChatOpenAI(**llm_kwargs)
        
        # Initialize tools for this field
        self._toolkit = ResearchToolkit()
        self._tools = tools or self._toolkit.get_tools_for_field(self.FIELD)
        
        # Initialize memory systems
        self._short_term = ShortTermMemory(max_size=10, agent_id=self.agent_id)
        self._long_term = LongTermMemory(agent_id=self.agent_id)
        
        # Initialize RAG
        self._vector_store = VectorStore(
            collection_name=f"rag_{self.FIELD}_{self.agent_id}"
        )
        self._rag = RetrieveReflectRetryRAG(
            vector_store=self._vector_store,
            field=self.FIELD
        )
        
        # Initialize state
        self._state = AgentState(
            agent_id=self.agent_id,
            agent_type=self.AGENT_TYPE,
            field=self.FIELD,
            display_name=self.DISPLAY_NAME
        )
        
        # Build agent executor
        self._agent_executor = self._build_agent_executor()
    
    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Each agent defines its own persona and expertise."""
        pass
    
    async def research(self, query: ResearchQuery) -> ResearchResult:
        """
        Main research method - the core workflow:
        
        1. Update state to RESEARCHING
        2. Retrieve context from RAG
        3. Get conversation history
        4. Build enhanced prompt with context
        5. Execute agent with tools
        6. Store in memory
        7. Build and return result
        """
        self._state.update_status(AgentStatus.RESEARCHING, query.query)
        
        # Get RAG context
        context, existing_papers, confidence = self._rag.get_context_for_query(query.query)
        
        # Get chat history
        chat_history = self._short_term.get_langchain_messages(limit=5)
        
        # Build input
        enhanced_input = self._build_research_input(query, context)
        
        # Reflect
        self._state.update_status(AgentStatus.REFLECTING)
        
        # Execute
        result = await self._agent_executor.ainvoke({
            "input": enhanced_input,
            "chat_history": chat_history
        })
        
        # Store in memory
        self._short_term.add_user_message(query.query)
        self._short_term.add_assistant_message(result["output"], self.agent_id)
        
        # Store insights in long-term memory
        if confidence < 0.5:  # New information worth remembering
            self._long_term.store_insight(
                result["output"][:500],
                query.query,
                confidence=0.7
            )
        
        # Build result
        return ResearchResult(
            agent_id=self.agent_id,
            agent_field=self.FIELD,
            query=query.query,
            summary=result["output"],
            papers=existing_papers,
            insights=self._extract_insights(result["output"]),
            confidence_score=self._calculate_confidence(result["output"], existing_papers)
        )
```

### 9.2 Domain Agents

Each domain agent specializes in a field with custom prompts:

**AI/ML Agent** (`agents/domain/ai_agent.py`):

```python
class AIMLAgent(BaseResearchAgent):
    FIELD = "ai_ml"
    DISPLAY_NAME = "AI/ML Research Agent"
    
    def _get_system_prompt(self) -> str:
        return """You are an expert AI/ML research scientist specializing in:
        - Machine Learning algorithms and theory
        - Deep Learning architectures (CNNs, RNNs, Transformers)
        - Natural Language Processing
        - Computer Vision
        - Reinforcement Learning
        
        Your approach:
        1. Search for relevant papers using arxiv and semantic scholar
        2. Analyze methodologies and results critically
        3. Identify connections to related work
        4. Provide clear, technical explanations
        5. Cite all sources properly
        
        Always prioritize peer-reviewed sources and recent publications."""
```

**Physics Agent** (`agents/domain/physics_agent.py`):

```python
class PhysicsAgent(BaseResearchAgent):
    FIELD = "physics"
    DISPLAY_NAME = "Physics Research Agent"
    
    def _get_system_prompt(self) -> str:
        return """You are an expert physicist specializing in:
        - Theoretical physics
        - Quantum mechanics and quantum computing
        - Particle physics
        - Astrophysics and cosmology
        - Condensed matter physics
        
        Your approach:
        1. Ground explanations in fundamental principles
        2. Use mathematical formalism when appropriate
        3. Connect theory to experimental evidence
        4. Acknowledge uncertainties and open questions
        
        Use arxiv and specialized physics databases."""
```

**Similar patterns for:** Biology, Chemistry, Mathematics, Neuroscience, Medicine, Computer Science.

### 9.3 Support Agents

Support agents have specialized roles:

**Literature Reviewer** (`agents/support/literature_reviewer.py`):

```python
class LiteratureReviewer(BaseResearchAgent):
    FIELD = "literature_review"
    DISPLAY_NAME = "Literature Reviewer"
    AGENT_TYPE = "support"
    
    def _get_system_prompt(self) -> str:
        return """You are a systematic literature review specialist.
        
        Your responsibilities:
        1. Summarize papers comprehensively
        2. Identify research gaps
        3. Trace the evolution of ideas
        4. Compare methodologies across papers
        5. Highlight consensus and controversies
        
        Produce structured summaries with clear organization."""
```

**Methodology Critic** (`agents/support/methodology_critic.py`):

```python
class MethodologyCritic(BaseResearchAgent):
    FIELD = "methodology_analysis"
    DISPLAY_NAME = "Methodology Critic"
    AGENT_TYPE = "support"
    
    def _get_system_prompt(self) -> str:
        return """You are a research methodology expert.
        
        Evaluate:
        - Experimental design validity
        - Statistical analysis appropriateness
        - Sample sizes and power
        - Potential biases and confounds
        - Reproducibility concerns
        
        Be constructive but thorough in criticism."""
```

**Fact Checker** (`agents/support/fact_checker.py`):

```python
class FactChecker(BaseResearchAgent):
    FIELD = "fact_checking"
    DISPLAY_NAME = "Fact Checker"
    AGENT_TYPE = "support"
    
    def _get_system_prompt(self) -> str:
        return """You verify claims against primary sources.
        
        Process:
        1. Identify specific claims
        2. Find original sources
        3. Verify accuracy
        4. Note any nuances or caveats
        5. Flag unsupported claims
        
        Be rigorous and cite verification sources."""
```

**Writing Assistant** (`agents/support/writing_assistant.py`):

```python
class WritingAssistant(BaseResearchAgent):
    FIELD = "writing"
    DISPLAY_NAME = "Writing Assistant"
    AGENT_TYPE = "support"
    
    def _get_system_prompt(self) -> str:
        return """You help produce clear scientific writing.
        
        Capabilities:
        - Draft abstracts and summaries
        - Improve clarity and flow
        - Ensure proper citation format
        - Maintain academic tone
        - Structure arguments logically
        
        Preserve technical accuracy while improving readability."""
```

**Cross-Domain Synthesizer** (`agents/support/cross_domain_synthesizer.py`):

```python
class CrossDomainSynthesizer(BaseResearchAgent):
    FIELD = "synthesis"
    DISPLAY_NAME = "Cross-Domain Synthesizer"
    AGENT_TYPE = "support"
    
    def _get_system_prompt(self) -> str:
        return """You find connections between different fields.
        
        Your role:
        1. Identify common patterns across domains
        2. Suggest cross-disciplinary applications
        3. Highlight transferable methodologies
        4. Synthesize insights into unified understanding
        
        Bridge gaps between specialized knowledge areas."""
```

---

## 12. Orchestrator

### File: `agents/orchestrator.py`

The Orchestrator is the central coordinator:

```python
class RoutingDecision(BaseModel):
    """Routing decision made by orchestrator."""
    domain_agents: List[str]      # Which domain agents to activate
    support_agents: List[str]     # Which support agents to use
    parallel: bool = True         # Run domain agents in parallel?
    reasoning: str                # Why this routing decision

class Orchestrator:
    """Coordinates the research team."""
    
    def __init__(self, team_config: TeamConfiguration):
        self.team_config = team_config
        self.session_id = str(uuid.uuid4())
        
        self._llm = ChatOpenAI(...)  # For routing decisions
        self._domain_agents: Dict[str, BaseResearchAgent] = {}
        self._support_agents: Dict[str, BaseResearchAgent] = {}
        
        self._init_agents()
        
        self._routing_prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_routing_system_prompt()),
            ("human", """Query: {query}
            
            Available Domain Agents: {domain_agents}
            Available Support Agents: {support_agents}
            
            Provide routing decision...""")
        ])
    
    def _init_agents(self):
        """Initialize agents based on team config."""
        # Create domain agents
        for field in self.team_config.domain_agents:
            agent_class = DOMAIN_AGENT_REGISTRY[field]
            self._domain_agents[field] = agent_class()
        
        # Create support agents
        for agent_type in self.team_config.support_agents:
            agent_class = SUPPORT_AGENT_REGISTRY[agent_type]
            self._support_agents[agent_type] = agent_class()
    
    async def route_query(self, query: ResearchQuery) -> RoutingDecision:
        """Decide which agents should handle this query."""
        chain = self._routing_prompt | self._llm | ...
        
        # LLM analyzes query and decides routing
        decision = await chain.ainvoke({
            "query": query.query,
            "domain_agents": list(self._domain_agents.keys()),
            "support_agents": list(self._support_agents.keys())
        })
        
        return RoutingDecision(**decision)
    
    async def execute_research(self, query: ResearchQuery) -> Dict[str, Any]:
        """Execute full research workflow."""
        
        # 1. Route query
        routing = await self.route_query(query)
        
        # 2. Execute domain agents (parallel or sequential)
        domain_results = []
        if routing.parallel:
            tasks = [
                self._domain_agents[field].research(query)
                for field in routing.domain_agents
            ]
            domain_results = await asyncio.gather(*tasks)
        else:
            for field in routing.domain_agents:
                result = await self._domain_agents[field].research(query)
                domain_results.append(result)
        
        # 3. Execute support agents
        support_results = []
        for agent_type in routing.support_agents:
            agent = self._support_agents[agent_type]
            result = await agent.process(domain_results)
            support_results.append(result)
        
        # 4. Synthesize final response
        final_response = await self._synthesize(
            query, domain_results, support_results
        )
        
        return {
            "query": query,
            "routing": routing,
            "domain_results": domain_results,
            "support_results": support_results,
            "final_response": final_response
        }
```

---

## 13. LangGraph Workflows (Structured vs Automated)

### 13.1 Structured Mode (Classic Multi‑Agent RAG)

In **structured** mode, the workflow behaves like the earlier version:

```python
graph.add_node("init", self._init_node)
graph.add_node("workflow_decision", self._workflow_decision_node)
graph.add_node("routing", self._routing_node)
graph.add_node("domain_research", self._domain_research_node)
graph.add_node("support_review", self._support_review_node)
graph.add_node("synthesis", self._synthesis_node)
graph.add_node("complete", self._complete_node)

graph.set_entry_point("init")

graph.add_edge("init", "workflow_decision")
graph.add_edge("workflow_decision", "routing")
graph.add_edge("routing", "domain_research")
graph.add_edge("domain_research", "support_review")
graph.add_edge("support_review", "synthesis")
graph.add_edge("synthesis", "complete")
graph.add_edge("complete", END)
```

### 13.2 Automated Mode (SciAgents‑Style)

In **automated** mode, the graph extends beyond support review into KG + hypothesis:

```python
# After routing and domain research:
graph.add_node("knowledge_graph", self._knowledge_graph_node)
graph.add_node("ontologist", self._ontologist_node)
graph.add_node("hypothesis_generation", self._hypothesis_generation_node)
graph.add_node("hypothesis_expansion", self._hypothesis_expansion_node)
graph.add_node("critique", self._critique_node)
graph.add_node("planner", self._planner_node)
graph.add_node("novelty_check", self._novelty_check_node)

# Automated path:
graph.add_edge("domain_research", "knowledge_graph")
graph.add_edge("knowledge_graph", "ontologist")
graph.add_edge("ontologist", "hypothesis_generation")
graph.add_edge("hypothesis_generation", "hypothesis_expansion")
graph.add_edge("hypothesis_expansion", "critique")
graph.add_edge("critique", "planner")
graph.add_edge("planner", "novelty_check")
graph.add_edge("novelty_check", "support_review")
```

The mode is selected by `workflow_mode` on the state and controlled from the Streamlit UI.

### 13.3 Visual High‑Level Flow

```
    ┌─────────────────────────────────────────────────────────┐
    │                      START                              │
    │                   (User Query)                          │
    └───────────────────────┬─────────────────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────────────┐
    │                   ROUTE NODE                            │
    │  - Analyze query                                        │
    │  - Select domain agents                                 │
    │  - Decide parallel vs sequential                        │
    └───────────────────────┬─────────────────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────────────┐
    │              DOMAIN RESEARCH NODE                       │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
    │  │ Agent 1  │  │ Agent 2  │  │ Agent 3  │   (parallel)  │
    │  │ (AI/ML)  │  │(Physics) │  │(Biology) │              │
    │  └────┬─────┘  └────┬─────┘  └────┬─────┘              │
    │       │             │             │                     │
    │       └─────────────┴─────────────┘                     │
    │                     │                                   │
    │              Combine Results                            │
    └───────────────────────┬─────────────────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────────────┐
    │             SUPPORT PROCESSING NODE                     │
    │  ┌─────────────┐  ┌────────────┐  ┌──────────────┐     │
    │  │ Lit Review  │  │Fact Check  │  │ Cross-Domain │     │
    │  └─────────────┘  └────────────┘  └──────────────┘     │
    └───────────────────────┬─────────────────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────────────┐
    │                 SYNTHESIZE NODE                         │
    │  - Combine all findings                                 │
    │  - Generate coherent response                           │
    │  - Add citations and structure                          │
    └───────────────────────┬─────────────────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────────────┐
    │                       END                               │
    │                (Final Response)                         │
    └─────────────────────────────────────────────────────────┘
```

---

## 14. Streamlit UI & Human‑in‑the‑Loop Checkpoints

### App Structure

**File:** `app.py` (Entry Point)

```python
import streamlit as st

# Configure page
st.set_page_config(
    page_title="Research Lab - AI Research Team",
    page_icon="🔬",
    layout="wide"
)

# Import pages
from ui.pages.home import render_home_page
from ui.pages.team_setup import render_team_setup_page
from ui.pages.research_session import render_research_session_page

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["Home", "Team Setup", "Research Session"])

# Route to page
if page == "Home":
    render_home_page()
elif page == "Team Setup":
    render_team_setup_page()
else:
    render_research_session_page()
```

### Team Setup Page

**File:** `ui/pages/team_setup.py`

```python
def render_team_setup_page():
    st.header("👥 Configure Your Research Team")
    
    # Domain selection (max 3)
    st.subheader("Select Your Research Team")
    st.write("Choose up to 3 research domains:")
    
    selected_domains = []
    for field, display_name in FIELD_DISPLAY_NAMES.items():
        if st.checkbox(f"**{display_name}**", key=field):
            selected_domains.append(field)
    
    if len(selected_domains) > 3:
        st.error("Maximum 3 domains allowed!")
    
    # Show support agents
    st.subheader("🛠️ Support Agents (Always Available)")
    st.markdown("""
    - 📚 Literature Reviewer
    - 🔍 Methodology Critic
    - ✓ Fact Checker
    - ✍️ Writing Assistant
    - 🔗 Cross-Domain Synthesizer
    """)
    
    # Start button
    if st.button("Start Research Session", disabled=len(selected_domains) == 0):
        # Save to session state
        st.session_state["team_config"] = TeamConfiguration(
            team_id=str(uuid.uuid4()),
            domain_agents=selected_domains
        )
        st.session_state["page"] = "research_session"
        st.rerun()
```

### Research Session Page (Updated)

**File:** `ui/pages/research_session.py`

Key updates:

- Adds a **workflow mode toggle** (structured vs automated)
- Initializes `st.session_state.workflow_mode` safely (fixing prior `StreamlitAPIException`)
- Renders **workflow steps** and **node outputs**
- Implements **checkpoint UI** for ontology, hypothesis, and critique

> See `research_lab/ui/pages/research_session.py` for the full implementation.

---

## 15. Data Flow

### Complete Request Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. USER INPUT                                                    │
│    "What are the latest advances in quantum machine learning?"   │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. QUERY CREATION                                                │
│    ResearchQuery(                                                │
│      query="What are the latest...",                             │
│      field=None,  # Auto-detect                                  │
│      priority=3,                                                 │
│      sources_required=["arxiv", "semantic_scholar"]              │
│    )                                                             │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. ORCHESTRATOR ROUTING                                          │
│    - Analyzes query with LLM                                     │
│    - Detects fields: ["ai_ml", "physics"]                        │
│    - Decides: Run AI/ML + Physics agents in parallel             │
│    - Activates: Literature Reviewer, Fact Checker                │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
┌─────────────────────────┐   ┌─────────────────────────┐
│ 4a. AI/ML AGENT         │   │ 4b. PHYSICS AGENT       │
│ ┌─────────────────────┐ │   │ ┌─────────────────────┐ │
│ │ Check Long-Term Mem │ │   │ │ Check Long-Term Mem │ │
│ └──────────┬──────────┘ │   │ └──────────┬──────────┘ │
│            ▼            │   │            ▼            │
│ ┌─────────────────────┐ │   │ ┌─────────────────────┐ │
│ │ RAG: Retrieve       │ │   │ │ RAG: Retrieve       │ │
│ │      Reflect        │ │   │ │      Reflect        │ │
│ │      Retry          │ │   │ │      Retry          │ │
│ └──────────┬──────────┘ │   │ └──────────┬──────────┘ │
│            ▼            │   │            ▼            │
│ ┌─────────────────────┐ │   │ ┌─────────────────────┐ │
│ │ Use Tools:          │ │   │ │ Use Tools:          │ │
│ │ - Arxiv search      │ │   │ │ - Arxiv search      │ │
│ │ - Semantic Scholar  │ │   │ │ - Web search        │ │
│ └──────────┬──────────┘ │   │ └──────────┬──────────┘ │
│            ▼            │   │            ▼            │
│ ┌─────────────────────┐ │   │ ┌─────────────────────┐ │
│ │ Generate Response   │ │   │ │ Generate Response   │ │
│ │ with LLM            │ │   │ │ with LLM            │ │
│ └──────────┬──────────┘ │   │ └──────────┬──────────┘ │
│            ▼            │   │            ▼            │
│ ┌─────────────────────┐ │   │ ┌─────────────────────┐ │
│ │ Store in Memory     │ │   │ │ Store in Memory     │ │
│ └─────────────────────┘ │   │ └─────────────────────┘ │
│                         │   │                         │
│ ResearchResult(         │   │ ResearchResult(         │
│   papers=[...],         │   │   papers=[...],         │
│   insights=[...],       │   │   insights=[...],       │
│   confidence=0.85       │   │   confidence=0.78       │
│ )                       │   │ )                       │
└───────────┬─────────────┘   └───────────┬─────────────┘
            │                             │
            └──────────┬──────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. SUPPORT AGENTS                                                │
│                                                                  │
│ ┌─────────────────────┐   ┌─────────────────────┐               │
│ │ Literature Reviewer │   │ Fact Checker        │               │
│ │ - Summarize papers  │   │ - Verify claims     │               │
│ │ - Identify gaps     │   │ - Check sources     │               │
│ └──────────┬──────────┘   └──────────┬──────────┘               │
│            │                         │                           │
│            └────────────┬────────────┘                           │
│                         │                                        │
│                         ▼                                        │
│            ┌─────────────────────┐                               │
│            │ Cross-Domain Synth  │                               │
│            │ - Find connections  │                               │
│            │ - Unified insights  │                               │
│            └─────────────────────┘                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. SYNTHESIS                                                     │
│    - Combine all domain results                                  │
│    - Integrate support agent insights                            │
│    - Generate coherent final response                            │
│    - Format with citations                                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. FINAL RESPONSE                                                │
│                                                                  │
│    "Quantum Machine Learning (QML) represents an emerging        │
│     intersection of quantum computing and ML. Key advances       │
│     include:                                                     │
│                                                                  │
│     1. **Variational Quantum Eigensolvers (VQE)**...             │
│     2. **Quantum Neural Networks**...                            │
│     3. **Quantum Speedups**...                                   │
│                                                                  │
│     Sources:                                                     │
│     - [Quantum ML: What Quantum Computing Means...](arxiv:...)   │
│     - [TensorFlow Quantum: A Software Framework...](arxiv:...)   │
│    "                                                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 16. Extending the System

### Adding a New Domain Agent

1. **Create the agent file** in `agents/domain/`:

```python
# agents/domain/astronomy_agent.py

from agents.base_agent import BaseResearchAgent

class AstronomyAgent(BaseResearchAgent):
    FIELD = "astronomy"
    DISPLAY_NAME = "Astronomy Research Agent"
    
    def _get_system_prompt(self) -> str:
        return """You are an expert astronomer specializing in:
        - Observational astronomy
        - Stellar physics
        - Galaxies and cosmology
        - Exoplanets
        - Space telescopes and instrumentation
        
        Use arxiv and NASA ADS for literature search."""
```

2. **Register in the registry** (`agents/domain/__init__.py`):

```python
from .astronomy_agent import AstronomyAgent

DOMAIN_AGENT_REGISTRY = {
    # ... existing agents ...
    "astronomy": AstronomyAgent,
}
```

3. **Add to settings** (`config/settings.py`):

```python
RESEARCH_FIELDS = [
    # ... existing ...
    "astronomy"
]

FIELD_DISPLAY_NAMES = {
    # ... existing ...
    "astronomy": "Astronomy"
}
```

### Adding a New Tool

1. **Create the tool** in `tools/`:

```python
# tools/nasa_ads.py

class NASAADSTool:
    """Search NASA Astrophysics Data System."""
    
    API_URL = "https://api.adsabs.harvard.edu/v1/search/query"
    
    def __init__(self):
        self.api_key = settings.nasa_ads_api_key
    
    def search(self, query: str, max_results: int = 10) -> List[Paper]:
        # Implementation
        pass
    
    def as_langchain_tool(self):
        @tool
        def nasa_ads_search(query: str) -> str:
            """Search NASA ADS for astronomy papers."""
            results = self.search(query)
            return self._format(results)
        return nasa_ads_search
```

2. **Add to toolkit** (`tools/web_search.py`):

```python
def get_tools_for_field(self, field: str) -> List:
    tools = [self.web.as_langchain_tool()]
    
    if field == "astronomy":
        tools.append(self.nasa_ads.as_langchain_tool())
        tools.append(self.arxiv.as_langchain_tool())
    
    # ... rest
```

### Adding a New Support Agent

Same pattern as domain agents, but in `agents/support/`.

---

## 17. Troubleshooting & Diagnostics

### Common Issues

#### API Key & Endpoint Errors

```
Error: Invalid API key
```

**Solution:** Check your `.env` file:
- Ensure `OPENAI_API_KEY` is set correctly
- If using custom endpoint, set `OPENAI_BASE_URL`
- API key should start with `sk-` (OpenAI) or `voc-` (Vocareum)

#### ChromaDB Errors

```
Error: Collection not found
```

**Solution:**
- Check `CHROMA_PERSIST_DIRECTORY` path exists
- Delete `./data/chroma_db` folder to reset

#### Import / Dependency Errors

```
ModuleNotFoundError: No module named 'langchain_classic'
```

**Solution:**
```bash
pip install langchain-community
```

#### Memory / GPU Issues

```
Error: CUDA out of memory
```

**Solution:**
- Reduce `max_results` in tool settings
- Use smaller embedding model

### Debugging Tips & Helper Files

1. **Enable verbose mode** in agents:
```python
self._agent_executor = AgentExecutor(
    agent=agent,
    tools=self._tools,
    verbose=True,  # Shows all tool calls
    handle_parsing_errors=True
)
```

2. **Check agent state**:
```python
agent = AIMLAgent()
print(agent.get_state())  # Shows current status
print(agent.get_memory_stats())  # Shows memory usage
```

3. **Test components individually**:
```bash
python test_components.py    # Basic imports
python test_functionality.py  # Full functional tests
```

---

## Quick Reference Card

### Key Commands

```bash
# Activate environment
cd research_lab
.\venv\Scripts\activate.bat

# Run app
streamlit run app.py

# Run quick workflow tests
python test_workflow_quick.py

# Run comprehensive tests (slower, uses APIs)
python -m pytest test_production_comprehensive.py -v
```

### Key Files to Modify

| What | Where |
|------|-------|
| Add new field | `config/settings.py` |
| Add new agent | `agents/domain/` or `agents/support/` |
| Add new tool | `tools/` + register in toolkit |
| Change prompts | Agent's `_get_system_prompt()` |
| Change workflow | `graphs/research_graph.py` |
| Change UI | `ui/pages/` |

### State Flow Summary

```
User Query
    │
    ▼
ResearchQuery (Pydantic)
    │
    ▼
WorkflowState (TypedDict) ──▶ Flows through LangGraph
    │
    ▼
ResearchResult (Pydantic) ──▶ From each agent
    │
    ▼
Final Response ──▶ Displayed in UI
```

---

## Congratulations! 🎉

You now have an updated, end‑to‑end view of the **current** Research Lab system:

1. **Dual workflows**: Structured (classic) and Automated (SciAgents‑style)
2. **Knowledge graph** built from **query‑relevant** papers
3. **Collaborative ontology** from domain agents
4. **Structured hypothesis pipeline** with expansion, critique, planning, and novelty check
5. **Local BGE‑M3 embeddings** with singleton caching and error‑aware fallbacks
6. **Robust tools**: URL context, Wikipedia API, PDF reader, auto‑seeding RAG
7. **Human‑in‑the‑loop checkpoints** wired into the Streamlit UI

Happy researching! 🔬


