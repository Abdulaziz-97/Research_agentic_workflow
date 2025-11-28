# 🔬 Research Lab - Complete System Documentation

## Master Guide to the Multi-Agent Research System

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Deep Dive](#2-architecture-deep-dive)
3. [Core Concepts](#3-core-concepts)
4. [Configuration & Settings](#4-configuration--settings)
5. [State Management (Pydantic Models)](#5-state-management-pydantic-models)
6. [Memory Systems](#6-memory-systems)
7. [RAG System (Retrieve-Reflect-Retry)](#7-rag-system-retrieve-reflect-retry)
8. [Knowledge Graph System](#8-knowledge-graph-system)
9. [Research Tools](#9-research-tools)
10. [Agent System](#10-agent-system)
11. [API Key Management](#11-api-key-management)
12. [Orchestrator](#12-orchestrator)
13. [LangGraph Workflow](#13-langgraph-workflow)
14. [Streamlit UI](#14-streamlit-ui)
15. [Data Flow](#15-data-flow)
16. [Extending the System](#16-extending-the-system)
17. [Troubleshooting](#17-troubleshooting)

---

## 1. System Overview

### What is the Research Lab?

The Research Lab is a **state-of-the-art multi-agent AI system** that simulates a coalition of research scientists producing publication-quality research outputs. It uses:

- **DeepAgents** (langchain-ai/deepagents) for advanced agent capabilities with filesystem backends and planning
- **LangChain** for LLM interactions and tool management
- **LangGraph** for orchestrating complex multi-agent workflows
- **NetworkX** for knowledge graph construction and path sampling
- **ChromaDB** for vector storage and semantic search
- **Pydantic** for type-safe state management
- **Streamlit** for the modern, research command center UI

### Key Capabilities

| Capability | Description |
|------------|-------------|
| **Multi-Domain Research** | 8 specialized agents for different scientific fields |
| **SciAgents Workflow** | Knowledge graph → Ontologist → Scientist I/II → Critic → Planner → Novelty Checker |
| **Knowledge Graph** | NetworkX-based concept graphs with path sampling and entity extraction |
| **Deep Agents** | Advanced agents with filesystem backends, planning, and sub-agent delegation |
| **Paper Search** | Arxiv, Semantic Scholar, PubMed, Tavily web search |
| **RAG System** | Retrieve-Reflect-Retry pattern for intelligent retrieval |
| **Memory** | Short-term (conversation) + Long-term (persistent) |
| **API Key Rotation** | Automatic key rotation with health tracking and fallback |
| **Thinking Display** | Gemini-style reasoning trail showing agent thought processes |
| **Academic Output** | Publication-quality research briefs with proper citations and structure |
| **Collaboration** | Agents work together, synthesizing cross-domain insights |

### Project Structure

```
research_lab/
├── app.py                      # Streamlit entry point
├── .env                        # Environment variables (API keys)
├── requirements.txt            # Python dependencies
│
├── config/
│   ├── settings.py             # Configuration & constants
│   └── key_manager.py          # API key rotation system
│
├── states/
│   ├── agent_state.py          # Pydantic models for agents
│   └── workflow_state.py       # LangGraph state definitions
│
├── memory/
│   ├── short_term.py           # Conversation buffer memory
│   └── long_term.py            # Persistent ChromaDB memory
│
├── rag/
│   ├── embeddings.py           # OpenAI embeddings wrapper
│   ├── vector_store.py         # ChromaDB vector store
│   └── retriever.py            # Retrieve-Reflect-Retry logic
│
├── tools/
│   ├── arxiv_tool.py           # Arxiv paper search
│   ├── semantic_scholar.py     # Semantic Scholar API
│   ├── pubmed_tool.py          # PubMed/NCBI search
│   └── web_search.py           # Tavily web search
│
├── agents/
│   ├── base_agent.py           # Base class for all agents
│   ├── orchestrator.py         # Main coordinator
│   ├── domain/                 # 8 domain-specific agents
│   │   ├── ai_agent.py
│   │   ├── physics_agent.py
│   │   ├── biology_agent.py
│   │   ├── chemistry_agent.py
│   │   ├── mathematics_agent.py
│   │   ├── neuroscience_agent.py
│   │   ├── medicine_agent.py
│   │   └── cs_agent.py
│   └── support/                # Support agents
│       ├── literature_reviewer.py
│       ├── methodology_critic.py
│       ├── fact_checker.py
│       ├── writing_assistant.py
│       ├── cross_domain_synthesizer.py
│       └── scientific_workflow.py  # SciAgents workflow agents
│           ├── OntologistAgent
│           ├── ScientistOneAgent
│           ├── ScientistTwoAgent
│           ├── CriticAgent
│           ├── PlannerAgent
│           └── NoveltyCheckerAgent
│
├── knowledge_graph/
│   ├── service.py             # Knowledge graph service (NetworkX)
│   └── seed_graph.json        # Initial graph seed data
│
├── graphs/
│   └── research_graph.py       # LangGraph workflow definition
│
└── ui/
    ├── components.py           # Reusable UI components
    ├── components_thinking.py   # Gemini-style thinking display
    └── pages/
        ├── home.py
        ├── team_setup.py
        └── research_session.py
```

---

## 2. Architecture Deep Dive

### Layered Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      PRESENTATION LAYER                         │
│                      (Streamlit UI)                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │    Home     │  │ Team Setup  │  │   Research Session      │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                      ORCHESTRATION LAYER                        │
│                      (LangGraph + Orchestrator)                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  ResearchGraph: Nodes → Edges → Conditional Routing     │    │
│  │  Orchestrator: Task analysis, agent selection, synthesis│    │
│  └─────────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│                        AGENT LAYER                              │
│  ┌───────────────────────┐  ┌───────────────────────────────┐   │
│  │    Domain Agents      │  │      Support Agents           │   │
│  │  ┌─────┐ ┌─────┐     │  │  ┌─────────┐ ┌─────────────┐  │   │
│  │  │AI/ML│ │Phys │ ... │  │  │Lit Rev  │ │Fact Checker │  │   │
│  │  └─────┘ └─────┘     │  │  └─────────┘ └─────────────┘  │   │
│  └───────────────────────┘  │  ┌─────────┐ ┌─────────────┐  │   │
│                              │  │Ontologist│ │Scientist I/II│ │   │
│                              │  │Critic    │ │Planner      │ │   │
│                              │  │Novelty   │ │Hierarchical │ │   │
│                              │  └─────────┘ └─────────────┘  │   │
│                              └───────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                      INTELLIGENCE LAYER                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  RAG System │  │   Memory    │  │      LLM (OpenAI)       │  │
│  │ (Retrieve-  │  │ Short+Long  │  │   ChatGPT-4o / etc      │  │
│  │  Reflect-   │  │   Term      │  │   with Key Rotation     │  │
│  │  Retry)     │  │             │  │                         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │         Knowledge Graph (NetworkX)                      │    │
│  │  - Concept nodes & relationships                        │    │
│  │  - Path sampling (shortest/random walk)                │    │
│  │  - Entity extraction from papers                        │    │
│  └─────────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│                        TOOLS LAYER                              │
│  ┌─────────┐  ┌──────────────┐  ┌────────┐  ┌─────────────┐    │
│  │  Arxiv  │  │Semantic      │  │ PubMed │  │   Tavily    │    │
│  │         │  │Scholar       │  │        │  │ Web Search  │    │
│  └─────────┘  └──────────────┘  └────────┘  └─────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│                       STORAGE LAYER                             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              ChromaDB (Vector Database)                  │    │
│  │  - Paper embeddings    - Memory embeddings               │    │
│  │  - Field-specific collections                            │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Agent Hierarchy

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

### 3.4 DeepAgents Integration

**DeepAgents** (from langchain-ai/deepagents) provides advanced agent capabilities:

```python
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

# Key features:
# 1. Filesystem Backend: Agents have persistent workspaces
# 2. Planning: Agents can create and execute plans
# 3. Sub-agent Delegation: Agents can delegate to specialized sub-agents
# 4. Tool Integration: Seamless LangChain tool usage
```

### 3.5 Knowledge Graph Concepts

**Knowledge Graphs** provide structured concept scaffolding:

```python
from knowledge_graph import KnowledgeGraphService

# Key concepts:
# 1. Nodes: Concepts, materials, diseases, methods
# 2. Edges: Relationships (related_to, treats, uses, etc.)
# 3. Path Sampling: Shortest path or random walk between concepts
# 4. Entity Extraction: Automatic extraction from papers
# 5. Context Generation: Structured prompts from graph paths
```

### 3.6 RAG (Retrieval-Augmented Generation)

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

## 4. Configuration & Settings

### File: `config/settings.py`

```python
from typing import Literal
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Loaded automatically from .env"""
    
    # LLM Provider
    llm_provider: Literal["openai", "gemini"] = "openai"
    
    # OpenAI Configuration
    openai_api_key: str = ""
    openai_base_url: str = ""
    openai_model: str = "gpt-3.5-turbo"
    
    # Gemini Configuration
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.0-pro"
    gemini_embedding_model: str = "text-embedding-004"
    
    # Tavily Web Search
    tavily_api_key: str = ""
    
    # ChromaDB
    chroma_persist_directory: str = "./data/chroma_db"
    
    # Memory Settings
    short_term_memory_size: int = 10
    long_term_memory_threshold: float = 0.7
    
    # Research Tools
    arxiv_max_results: int = 10
    semantic_scholar_max_results: int = 10
    pubmed_max_results: int = 10
```

### Environment Variables (.env)

```bash
# Select provider
LLM_PROVIDER=gemini   # or openai

# Gemini Configuration
GEMINI_API_KEY=key1,key2,key3
GEMINI_MODEL=gemini-3.0-pro
GEMINI_EMBEDDING_MODEL=text-embedding-004

# OpenAI Configuration (if LLM_PROVIDER=openai)
OPENAI_API_KEY=key1,key2,key3
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-3.5-turbo

# Tavily Web Search
TAVILY_API_KEY=your-tavily-key

# ChromaDB
CHROMA_PERSIST_DIRECTORY=./data/chroma_db
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

LangGraph requires TypedDict for state:

```python
from typing import TypedDict, List, Dict, Any
from langchain_core.messages import BaseMessage

class WorkflowState(TypedDict):
    """State that flows through the LangGraph workflow."""
    
    messages: List[BaseMessage]           # Conversation history
    current_agent: str                    # Currently active agent
    current_query: ResearchQuery          # Current research query
    team_composition: List[str]           # Selected team members
    domain_results: List[ResearchResult]  # Results from domain agents
    support_results: Dict[str, ResearchResult] # Results from support agents
    
    # SciAgents workflow outputs
    knowledge_context: Dict[str, Any]     # Knowledge graph path and context
    ontology_blueprint: Dict[str, Any]    # Ontologist JSON output
    scientist_proposal: Dict[str, Any]    # Scientist I proposal
    scientist_expansion: Dict[str, Any]   # Scientist II expansion
    critic_feedback: str                  # Critic assessment
    planner_plan: Dict[str, Any]          # Planner roadmap
    novelty_report: Dict[str, Any]       # Novelty checker results
    hierarchical_sections: Dict[str, Any] # Hierarchical expander output
    
    research_context: Dict[str, Any]      # Shared context
    final_response: str                   # Synthesized final answer
    final_papers: List[Paper]             # Final paper list
    current_phase: str                    # Current workflow phase
    research_stats: Dict[str, Any]        # Research statistics
    phase_details: Dict[str, Any]         # Phase-specific details
    thinking_trail: List[Dict[str, Any]]  # Agent reasoning steps (for UI)
    iteration_count: int                  # Prevent infinite loops
    max_iterations: int                   # Maximum iterations
    error_message: Optional[str]          # Error handling
    retry_count: int                      # Retry counter
    session_id: str                       # Session identifier
    started_at: str                      # Start timestamp
    completed_at: Optional[str]           # Completion timestamp
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

## 7. RAG System (Retrieve-Reflect-Retry)

### 7.1 Embeddings

**File:** `rag/embeddings.py`

```python
def get_embeddings_model(model: str = "text-embedding-3-small"):
    """Get OpenAI embeddings with custom base URL support."""
    kwargs = {
        "model": model,
        "openai_api_key": settings.openai_api_key
    }
    if settings.openai_base_url:
        kwargs["openai_api_base"] = settings.openai_base_url
    return OpenAIEmbeddings(**kwargs)

class EmbeddingManager:
    """Manages embedding generation."""
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a query."""
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple documents."""
    
    def similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Calculate cosine similarity between embeddings."""
        # Returns 0-1 score
```

### 7.2 Vector Store

**File:** `rag/vector_store.py`

```python
class VectorStore:
    """ChromaDB wrapper for document storage and retrieval."""
    
    def __init__(self, collection_name: str):
        self._client = chromadb.PersistentClient(
            path=settings.chroma_persist_directory
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name
        )
        self.embedding_manager = EmbeddingManager()
    
    def add_document(self, content: str, metadata: dict = None) -> str:
        """Add a single document."""
        embedding = self.embedding_manager.embed_query(content)
        self._collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata]
        )
    
    def add_paper(self, paper: Paper) -> str:
        """Add a research paper."""
        content = f"Title: {paper.title}\nAuthors: {paper.authors}\nAbstract: {paper.abstract}"
        # ... store with metadata
    
    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Semantic search for documents."""
        query_embedding = self.embedding_manager.embed_query(query)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return self._format_results(results)
```

### 7.3 Retrieve-Reflect-Retry Pattern

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

## 8. Knowledge Graph System

### 8.1 Overview

The Knowledge Graph System provides structured concept scaffolding inspired by SciAgents methodology. It uses NetworkX to build and query graphs of scientific concepts, enabling path sampling and structured hypothesis generation.

**File:** `knowledge_graph/service.py`

### 8.2 Knowledge Graph Service

```python
from knowledge_graph import KnowledgeGraphService, KnowledgeGraphContext

class KnowledgeGraphService:
    """
    Builds lightweight knowledge graphs from seed data and documents.
    
    Features:
    - Seed graph loading from JSON
    - Entity extraction from papers
    - Path sampling (shortest/random walk)
    - Context generation for agents
    """
    
    def __init__(self, seed_path: Optional[Path] = None):
        self.graph = nx.Graph()
        self._load_seed_graph()
    
    def ingest_papers(self, papers: List[Paper]):
        """
        Augment the graph with nodes derived from papers.
        
        Extracts:
        - Materials (silk, fibroin, amyloid, etc.)
        - Medical conditions (syndromes, diseases)
        - Methods/techniques (molecular dynamics, simulations)
        - Creates relationships automatically
        """
    
    def map_keywords(self, query: str) -> List[str]:
        """Map query text to existing graph nodes."""
    
    def sample_path(
        self,
        query: str,
        strategy: str = "random",  # "shortest" or "random"
        max_steps: int = 12
    ) -> KnowledgeGraphContext:
        """
        Sample a subgraph path based on the query.
        
        Returns:
            KnowledgeGraphContext with:
            - nodes: List of node dictionaries
            - edges: List of edge dictionaries
            - path: List of node IDs in path order
            - summary: Text summary of the path
            - prompt: Structured prompt for agents
        """
```

### 8.3 Knowledge Graph Context

```python
@dataclass
class KnowledgeGraphContext:
    """Container for sampled knowledge graph context."""
    
    nodes: List[Dict[str, str]]      # Node data (id, label, description)
    edges: List[Dict[str, str]]       # Edge data (source, target, relation)
    path: List[str]                   # Node IDs in path order
    summary: str                      # Human-readable path summary
    prompt: str                       # Structured prompt for agents
```

### 8.4 Entity Extraction

The system automatically extracts entities from papers:

```python
# Materials extraction patterns
material_patterns = [
    r'\b(silk|fibroin|amyloid|fibril|hydrogel|polymer)\w*\b',
    r'\b(collagen|elastin|keratin|chitin|cellulose)\w*\b',
]

# Disease/condition patterns
disease_patterns = [
    r'\b(syndrome|disease|disorder|pathology)\w*\b',
    r'\b(Cogan|Alzheimer|Parkinson|diabetes)\w*\b',
]

# Method/technique patterns
method_patterns = [
    r'\b(molecular dynamics|simulation|modeling|therapy)\w*\b',
]
```

### 8.5 Path Sampling Strategies

**Shortest Path:**
- Finds the shortest path between two matched nodes
- Good for direct concept connections
- Uses `nx.shortest_path()`

**Random Walk:**
- Biased random walk toward target neighborhood
- More exploratory, discovers unexpected connections
- Configurable `max_steps` (default: 12)

### 8.6 Usage in Workflow

```python
# In research graph:
kg_service = KnowledgeGraphService()

# Sample initial path
kg_context = kg_service.sample_path(query, strategy="random")

# Store in state
state["knowledge_context"] = {
    "nodes": kg_context.nodes,
    "edges": kg_context.edges,
    "path": kg_context.path,
    "summary": kg_context.summary
}

# After domain research, enrich graph
kg_service.ingest_papers(domain_results.papers)

# Re-sample with enriched graph
kg_context = kg_service.sample_path(query, strategy="random")
```

---

## 9. Research Tools

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

### 8.6 Research Toolkit (Unified Interface)

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

## 10. Agent System

### 10.1 DeepAgents Integration

All agents are now built using **DeepAgents** (langchain-ai/deepagents), which provides:

- **Filesystem Backend**: Each agent has a persistent workspace directory
- **Planning Capabilities**: Agents can create and execute multi-step plans
- **Sub-agent Delegation**: Agents can delegate tasks to specialized sub-agents
- **Enhanced Tool Usage**: Better integration with LangChain tools

### 10.2 Base Research Agent (Updated)

### 9.1 Base Research Agent

**File:** `agents/base_agent.py`

The `BaseResearchAgent` is the foundation for all agents, now using DeepAgents:

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
        
        # Build deep agent executor
        self._agent_executor = self._build_deep_agent()
    
    def _build_deep_agent(self):
        """Build the Deep Agent with filesystem backend."""
        from deepagents import create_deep_agent
        from deepagents.backends import FilesystemBackend
        
        # Create workspace directory
        workspace_dir = os.path.abspath(f"./workspaces/{self.agent_id}")
        os.makedirs(workspace_dir, exist_ok=True)
        
        # Get LLM (with key manager support)
        if self._key_manager:
            current_key = self._key_manager.get_current_key()
            llm = ChatOpenAI(
                model=settings.openai_model,
                openai_api_key=current_key,
                openai_api_base=settings.openai_base_url if settings.openai_base_url else None
            )
        else:
            llm = self._llm
        
        return create_deep_agent(
            model=llm,
            tools=self._tools,
            system_prompt=self._get_system_prompt(),
            backend=FilesystemBackend(root_dir=workspace_dir)
        )
    
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
        
        # Execute (DeepAgents uses messages format)
        messages = [HumanMessage(content=enhanced_input)]
        if chat_history:
            messages = chat_history + messages
        
        result = await self._agent_executor.ainvoke({
            "messages": messages
        })
        
        # Extract output from DeepAgents result
        output = result["messages"][-1].content
        
        # Extract thinking steps for UI display
        thinking_steps = self._extract_thinking_steps(result["messages"])
        
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

### 10.3 Support Agents

#### Traditional Support Agents

**Literature Reviewer, Methodology Critic, Fact Checker, Writing Assistant, Cross-Domain Synthesizer** - Same as before, now using DeepAgents.

#### SciAgents-Inspired Workflow Agents

**File:** `agents/support/scientific_workflow.py`

These agents implement the SciAgents methodology for structured hypothesis generation:

**OntologistAgent:**
```python
class OntologistAgent(KnowledgeGraphAwareAgent):
    """
    Interprets knowledge graph paths and generates structured JSON hypotheses.
    
    Output format (JSON):
    {
        "hypothesis": "Testable hypothesis",
        "outcome": "Expected outcomes with metrics",
        "mechanisms": "Mechanistic explanation",
        "design_principles": "Design principles",
        "unexpected_properties": "Predicted unexpected behaviors",
        "comparison": "State-of-the-art comparison",
        "novelty": "Novelty statement"
    }
    """
```

**ScientistOneAgent:**
```python
class ScientistOneAgent(KnowledgeGraphAwareAgent):
    """
    Expands ontology blueprint into rigorous research proposal.
    
    Output: 1000+ word structured markdown with:
    - Formal hypothesis statements
    - Quantitative outcome metrics
    - Multi-scale mechanistic framework
    - Design principles with specific materials
    - State-of-the-art comparisons
    """
```

**ScientistTwoAgent:**
```python
class ScientistTwoAgent(KnowledgeGraphAwareAgent):
    """
    Provides quantitative depth and actionable protocols.
    
    Output: 1200+ word deep dive with:
    - Material property predictions with error bars
    - Energy budgets and kinetic parameters
    - Computational modeling plans
    - Detailed experimental protocols
    - Risk assessment
    """
```

**CriticAgent:**
```python
class CriticAgent(KnowledgeGraphAwareAgent):
    """
    Provides brutal honest assessment of proposals.
    
    Evaluates:
    - Logical gaps
    - Evidence requirements
    - Ethical/safety constraints
    - Revision checklists
    """
```

**PlannerAgent:**
```python
class PlannerAgent(KnowledgeGraphAwareAgent):
    """
    Creates actionable research roadmap.
    
    Output includes:
    - Mechanism deep-dives (prioritized list)
    - Modeling priorities table
    - Experimental priorities table
    - DeepAgents-compatible TODO list
    """
```

**NoveltyCheckerAgent:**
```python
class NoveltyCheckerAgent(KnowledgeGraphAwareAgent):
    """
    Assesses novelty by querying Semantic Scholar.
    
    Output (JSON):
    {
        "score": 0.0-1.0,  # Novelty score
        "overlapping_papers": [...],  # List of similar papers
        "summary": "...",  # Assessment summary
        "recommendations": "..."  # Recommendations
    }
    
    MUST use Semantic Scholar tool to query actual papers.
    """
```

**HierarchicalExpanderAgent:**
```python
class HierarchicalExpanderAgent(KnowledgeGraphAwareAgent):
    """
    Expands research into hierarchical sections before synthesis.
    
    Creates:
    - Mechanism deep-dives
    - Modeling roadmaps
    - Experimental roadmaps
    - Critical appraisal sections
    """
```

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

## 11. API Key Management

### 11.1 Key Manager Overview

**File:** `config/key_manager.py`

The `KeyManager` provides automatic API key rotation with health tracking and intelligent fallback. This is essential when using multiple API keys (e.g., Vocareum keys with limited credits).

### 11.2 Key Manager Features

```python
from config.key_manager import KeyManager, get_key_manager

class KeyManager:
    """
    Manages multiple API keys with automatic rotation.
    
    Features:
    - Automatic rotation on failure
    - Health tracking (failure count, disabled until)
    - Intelligent disable duration based on error type
    - Thread-safe operations
    """
    
    def __init__(self, keys: List[str], base_url: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.keys = [k.strip() for k in keys if k.strip()]
        self.base_url = base_url
        self.model = model
        self.key_health: Dict[str, Dict] = {}  # Tracks key health
        self.current_index = 0
```

### 11.3 Key Health Tracking

```python
# Key health structure:
{
    "key": {
        "failure_count": int,        # Number of failures
        "failed_at": datetime,       # Last failure time
        "disabled_until": datetime    # When key becomes available again
    }
}
```

### 11.4 Disable Duration Logic

The system intelligently disables keys based on error type:

```python
# Rate limit (429): 1 minute
# Budget/Insufficient credits: 1 hour
# Authentication (401): 365 days (permanently disabled)
# Other errors: 5 minutes
```

### 11.5 Usage in Agents

```python
# In BaseResearchAgent:
self._key_manager = get_key_manager()

# When making LLM calls:
if self._key_manager:
    current_key = self._key_manager.get_current_key()
    llm = ChatOpenAI(
        model=settings.openai_model,
        openai_api_key=current_key,
        openai_api_base=settings.openai_base_url
    )

# On error:
if is_key_error:
    await self._key_manager.mark_key_failed(current_key, error_str)
    # System automatically rotates to next key
```

### 11.6 Automatic Retry with Key Rotation

The system implements retry logic at multiple levels:

1. **Agent Level**: `BaseResearchAgent.research()` retries with key rotation
2. **Graph Level**: `_retry_with_key_rotation()` wrapper in research graph
3. **UI Level**: Automatic retry loop in Streamlit on key errors

### 11.7 Configuration

```python
# In .env file:
OPENAI_API_KEY=key1,key2,key3,key4,key5,key6  # Comma-separated

# In app.py:
from config.key_manager import init_key_manager
init_key_manager()  # Initialize before creating agents
```

### 11.8 Key Manager API

```python
# Get current active key
current_key = key_manager.get_current_key()

# Get all available keys (not disabled)
available = key_manager.get_available_keys()

# Mark key as failed
await key_manager.mark_key_failed(key, error_message)

# Mark key as successful (resets failure count)
key_manager.mark_key_success(key)

# Reset all keys (re-enable disabled keys)
key_manager.reset_all_keys()

# Get LLM instance with current key
llm = await key_manager.get_llm(temperature=0.7, max_tokens=4000)
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

## 13. LangGraph Workflow

### 13.1 Updated Workflow Overview

The research workflow now includes a comprehensive SciAgents-inspired pipeline:

```
START
  │
  ▼
Knowledge Context (Sample KG path)
  │
  ▼
Ontologist (Generate structured hypothesis JSON)
  │
  ▼
Scientist I (Expand to research proposal)
  │
  ▼
Scientist II (Add quantitative depth & protocols)
  │
  ▼
Critic (Assess and provide feedback)
  │
  ▼
Planner (Create actionable roadmap)
  │
  ▼
Novelty Checker (Query Semantic Scholar for novelty)
  │
  ▼
Hierarchical Expander (Deep-dive sections)
  │
  ▼
Domain Research (Parallel domain agents)
  │
  ▼
Synthesis (Combine all outputs into final paper)
  │
  ▼
END
```

### 13.2 New Workflow Nodes

### File: `graphs/research_graph.py`

LangGraph defines the workflow as a state machine:

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

class ResearchGraph:
    """LangGraph-based research workflow."""
    
    def __init__(self, team_config: TeamConfiguration):
        self.team_config = team_config
        self.orchestrator = Orchestrator(team_config)
        self.checkpointer = MemorySaver()  # State persistence
        
        self.graph = self._build_graph()
        self.compiled_graph = self.graph.compile(
            checkpointer=self.checkpointer
        )
    
    def _build_graph(self) -> StateGraph:
        """Build the workflow graph."""
        
        # Create graph with state schema
        graph = StateGraph(WorkflowState)
        
        # Add nodes
        graph.add_node("route", self._route_node)
        graph.add_node("domain_research", self._domain_research_node)
        graph.add_node("support_processing", self._support_processing_node)
        graph.add_node("synthesize", self._synthesize_node)
        
        # Set entry point
        graph.set_entry_point("route")
        
        # Add edges
        graph.add_edge("route", "domain_research")
        graph.add_edge("domain_research", "support_processing")
        graph.add_edge("support_processing", "synthesize")
        graph.add_edge("synthesize", END)
        
        return graph
```

### 13.2 New Workflow Nodes

The workflow now includes SciAgents-inspired nodes:

```python
async def _knowledge_context_node(self, state: WorkflowState) -> WorkflowState:
    """Sample knowledge graph path for query."""
    state["current_phase"] = "knowledge_graph"
    
    query = state["current_query"].query
    kg_context = self.kg_service.sample_path(query, strategy="random", max_steps=12)
    
    state["knowledge_context"] = {
        "nodes": kg_context.nodes,
        "edges": kg_context.edges,
        "path": kg_context.path,
        "summary": kg_context.summary
    }
    
    return state

async def _ontologist_node(self, state: WorkflowState) -> WorkflowState:
    """Generate structured hypothesis from knowledge graph."""
    state["current_phase"] = "ontologist"
    
    kg_context = state["knowledge_context"]
    query = ResearchQuery(query=kg_context["summary"])
    
    result = await self._retry_with_key_rotation(
        lambda: self.ontologist_agent.research(query),
        "Ontologist"
    )
    
    # Parse JSON output
    try:
        state["ontology_blueprint"] = json.loads(result.summary)
    except:
        state["ontology_blueprint"] = {"raw": result.summary}
    
    return state

async def _scientist_one_node(self, state: WorkflowState) -> WorkflowState:
    """Expand ontology into research proposal."""
    state["current_phase"] = "scientist_one"
    
    ontology = state["ontology_blueprint"]
    query = ResearchQuery(query=f"Expand this ontology: {json.dumps(ontology)}")
    
    result = await self._retry_with_key_rotation(
        lambda: self.scientist_one_agent.research(query),
        "Scientist I"
    )
    
    state["scientist_proposal"] = {"summary": result.summary}
    return state

async def _scientist_two_node(self, state: WorkflowState) -> WorkflowState:
    """Add quantitative depth and protocols."""
    state["current_phase"] = "scientist_two"
    
    proposal = state["scientist_proposal"]
    query = ResearchQuery(query=f"Add quantitative depth to: {proposal['summary']}")
    
    result = await self._retry_with_key_rotation(
        lambda: self.scientist_two_agent.research(query),
        "Scientist II"
    )
    
    state["scientist_expansion"] = {"summary": result.summary}
    return state

async def _critic_node(self, state: WorkflowState) -> WorkflowState:
    """Critically assess the proposal."""
    state["current_phase"] = "critic"
    
    proposal = state["scientist_proposal"]
    expansion = state["scientist_expansion"]
    query = ResearchQuery(query=f"Critically assess:\n{proposal['summary']}\n\n{expansion['summary']}")
    
    result = await self._retry_with_key_rotation(
        lambda: self.critic_agent.research(query),
        "Critic"
    )
    
    state["critic_feedback"] = result.summary
    return state

async def _planner_node(self, state: WorkflowState) -> WorkflowState:
    """Create actionable research roadmap."""
    state["current_phase"] = "planner"
    
    proposal = state["scientist_proposal"]
    expansion = state["scientist_expansion"]
    query = ResearchQuery(query=f"Create roadmap for:\n{proposal['summary']}\n\n{expansion['summary']}")
    
    result = await self._retry_with_key_rotation(
        lambda: self.planner_agent.research(query),
        "Planner"
    )
    
    state["planner_plan"] = {"summary": result.summary}
    return state

async def _novelty_node(self, state: WorkflowState) -> WorkflowState:
    """Check novelty using Semantic Scholar."""
    state["current_phase"] = "novelty"
    
    proposal = state["scientist_proposal"]
    query = ResearchQuery(query=f"Assess novelty of: {proposal['summary']}")
    
    result = await self._retry_with_key_rotation(
        lambda: self.novelty_agent.research(query),
        "Novelty Checker"
    )
    
    # Parse JSON output
    try:
        state["novelty_report"] = json.loads(result.summary)
    except:
        state["novelty_report"] = {"raw": result.summary}
    
    return state

async def _hierarchical_expander_node(self, state: WorkflowState) -> WorkflowState:
    """Expand into hierarchical sections."""
    state["current_phase"] = "hierarchical"
    
    # Combine all previous outputs
    context = self._context_string(state)
    query = ResearchQuery(query=f"Create hierarchical expansion:\n{context}")
    
    result = await self._retry_with_key_rotation(
        lambda: self.hierarchical_agent.research(query),
        "Hierarchical Expander"
    )
    
    state["hierarchical_sections"] = {"summary": result.summary}
    return state

async def _domain_research_node(self, state: WorkflowState) -> WorkflowState:
    """Domain research node - parallel agent execution."""
    state["current_phase"] = "domain_research"
    
    query = state["current_query"]
    
    # Execute domain agents in parallel
    tasks = [
        self._retry_with_key_rotation(
            lambda: agent.research(query),
            f"Domain Agent: {field}"
        )
        for field, agent in self.domain_agents.items()
    ]
    results = await asyncio.gather(*tasks)
    
    state["domain_results"] = results
    
    # Ingest papers into knowledge graph
    all_papers = []
    for result in results:
        all_papers.extend(result.papers)
    self.kg_service.ingest_papers(all_papers)
    
    return state

async def _synthesize_node(self, state: WorkflowState) -> WorkflowState:
    """Synthesize all results into publication-quality paper."""
    state["current_phase"] = "synthesis"
    
    # Combine all outputs
    workflow_outputs = {
        "knowledge_context": state.get("knowledge_context", {}),
        "ontology_blueprint": state.get("ontology_blueprint", {}),
        "scientist_proposal": state.get("scientist_proposal", {}),
        "scientist_expansion": state.get("scientist_expansion", {}),
        "critic_feedback": state.get("critic_feedback", ""),
        "planner_plan": state.get("planner_plan", {}),
        "novelty_report": state.get("novelty_report", {}),
        "hierarchical_sections": state.get("hierarchical_sections", {})
    }
    
    domain_findings = "\n\n".join([r.to_markdown() for r in state["domain_results"]])
    
    # Use synthesis prompt that incorporates all structured outputs
    prompt = SYNTHESIS_SYSTEM_PROMPT  # Academic paper format
    
    llm = await self._get_llm_with_key_rotation()
    chain = prompt | llm | StrOutputParser()
    
    state["final_response"] = await chain.ainvoke({
        "query": state["current_query"].query,
        "workflow_outputs": json.dumps(workflow_outputs, indent=2),
        "domain_findings": domain_findings
    })
    
    # Collect all papers
    all_papers = []
    for result in state["domain_results"]:
        all_papers.extend(result.papers)
    state["final_papers"] = all_papers
    
    return state
```

### Visual Workflow

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

### 13.3 Key Rotation in Workflow

All agent calls in the workflow are wrapped with `_retry_with_key_rotation()`:

```python
async def _retry_with_key_rotation(self, agent_call, agent_name: str, max_retries: int = 3):
    """
    Retry an agent call with automatic key rotation.
    
    - Detects key-related errors (rate limit, budget, auth)
    - Marks failed keys and rotates to next available
    - Retries up to max_retries times
    - Returns result or raises last error
    """
```

This ensures the workflow continues even if individual keys fail.

---

## 14. Streamlit UI

### 14.1 Modern UI Design

The UI has been completely redesigned with:
- **Dark theme** with glass-morphism effects
- **Card-based layouts** for better organization
- **Gemini-style thinking display** showing agent reasoning
- **Expandable sections** for all workflow outputs
- **Research command center** aesthetic

### 14.2 Thinking Display

**File:** `ui/components_thinking.py`

```python
def render_thinking_display(thinking_steps: List[Dict[str, Any]], agent_name: str = "Research Team"):
    """
    Render Gemini-style thinking display.
    
    Shows:
    - Step-by-step reasoning
    - Tool calls with parameters
    - Observations and results
    - Expandable cards for each step
    """
```

The thinking trail is extracted from DeepAgents' message history and displayed in an expandable format.

### 14.3 Workflow Output Display

The research session page now displays:

1. **Knowledge Graph Path**: Node count, summary, and path visualization
2. **Ontologist Blueprint**: Formatted JSON with hypothesis structure
3. **Scientist I Proposal**: Full markdown proposal (1000+ words)
4. **Scientist II Expansion**: Quantitative deep dive (1200+ words)
5. **Critic Assessment**: Critical feedback and requirements
6. **Planner Roadmap**: Actionable tables and TODO lists
7. **Novelty Assessment**: Score, overlapping papers, recommendations
8. **Final Synthesis**: Publication-quality research brief

All sections are expandable and formatted for readability.

### 14.4 Error Handling UI

The UI provides:
- **Automatic retry** with key rotation on API errors
- **Key status display**: Shows total keys, available keys, current key, failures
- **Reset All Keys** button to re-enable disabled keys
- **Clear error messages** with actionable guidance

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

### Research Session Page

**File:** `ui/pages/research_session.py`

```python
def render_research_session_page():
    st.header("🔬 Research Session")
    
    # Check for team config
    if "team_config" not in st.session_state:
        st.warning("Please configure your team first!")
        return
    
    team_config = st.session_state["team_config"]
    
    # Initialize graph if needed
    if "research_graph" not in st.session_state:
        st.session_state["research_graph"] = create_research_graph(team_config)
    
    # Display team
    st.sidebar.subheader("Your Team")
    for field in team_config.domain_agents:
        st.sidebar.write(f"• {FIELD_DISPLAY_NAMES[field]}")
    
    # Chat interface
    st.subheader("Ask a Research Question")
    
    # Message history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    
    # Display messages
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # Input
    if query := st.chat_input("Enter your research question..."):
        # Add user message
        st.session_state["messages"].append({"role": "user", "content": query})
        
        # Show user message
        with st.chat_message("user"):
            st.write(query)
        
        # Process with research graph
        with st.chat_message("assistant"):
            with st.spinner("Researching..."):
                # Create query
                research_query = ResearchQuery(query=query)
                
                # Run graph
                graph = st.session_state["research_graph"]
                result = graph.run_sync(query)
                
                # Display result
                st.write(result["final_response"])
                
                # Show sources
                with st.expander("📚 Sources"):
                    for paper in result.get("papers", []):
                        st.write(f"- [{paper.title}]({paper.url})")
        
        # Add to history
        st.session_state["messages"].append({
            "role": "assistant",
            "content": result["final_response"]
        })
```

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

## 17. Troubleshooting

### 17.1 API Key Issues

**Error: "Insufficient budget available"**

**Solution:**
1. Check if you have multiple keys configured in `.env` (comma-separated)
2. The system will automatically rotate to the next key
3. If all keys are exhausted, add credits or add more keys
4. Use "Reset All Keys" button to re-enable disabled keys

**Error: "Rate limit exceeded"**

**Solution:**
- Keys are automatically disabled for 1 minute on rate limit
- System rotates to next available key
- Consider using `gpt-3.5-turbo` instead of `gpt-4o` to reduce costs

### 17.2 Knowledge Graph Issues

**Error: "No path found between nodes"**

**Solution:**
- The system will return an empty context
- Ensure papers are being ingested: `kg_service.ingest_papers(papers)`
- Check `seed_graph.json` has initial nodes

### 17.3 JSON Parsing Issues

**Error: "Failed to parse JSON from Ontologist"**

**Solution:**
- The system has robust JSON extraction with regex fallbacks
- If parsing fails, raw output is stored in `{"raw": "..."}`
- Check agent prompts ensure JSON-only output

### 17.4 DeepAgents Issues

**Error: "Workspace directory not found"**

**Solution:**
- Workspaces are created automatically in `./workspaces/{agent_id}/`
- Ensure write permissions in project directory
- Check `FilesystemBackend` initialization

### Common Issues

#### API Key Errors

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

#### Import Errors

```
ModuleNotFoundError: No module named 'langchain_classic'
```

**Solution:**
```bash
pip install langchain-community
```

#### Memory Issues

```
Error: CUDA out of memory
```

**Solution:**
- Reduce `max_results` in tool settings
- Use smaller embedding model

### Debugging Tips

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

# Run tests
python test_functionality.py
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

You now have a complete understanding of the Research Lab system. Key takeaways:

1. **Modular Design**: Each component is self-contained and reusable
2. **DeepAgents Integration**: Advanced agent capabilities with filesystem backends
3. **Knowledge Graph System**: Structured concept scaffolding for hypothesis generation
4. **SciAgents Workflow**: Rigorous scientific methodology with structured outputs
5. **API Key Rotation**: Automatic fallback and health tracking
6. **Type Safety**: Pydantic ensures data consistency
7. **Intelligent Retrieval**: RAG with reflection improves quality
8. **Memory**: Both conversation context and persistent learning
9. **Orchestration**: LangGraph manages complex multi-agent workflows
10. **Academic Output**: Publication-quality research briefs with proper structure
11. **Thinking Display**: Gemini-style reasoning trail for transparency
12. **Extensibility**: Easy to add new agents, tools, and features

## Recent Major Updates

- ✅ **DeepAgents Integration**: All agents now use DeepAgents for advanced capabilities
- ✅ **Knowledge Graph System**: NetworkX-based concept graphs with path sampling
- ✅ **SciAgents Workflow**: Ontologist → Scientist I/II → Critic → Planner → Novelty Checker
- ✅ **API Key Rotation**: Automatic rotation with health tracking
- ✅ **Thinking Display**: Gemini-style reasoning trail in UI
- ✅ **Academic Quality**: Publication-quality output with proper citations
- ✅ **Cost Optimization**: Default model changed to `gpt-3.5-turbo`

Happy researching! 🔬

