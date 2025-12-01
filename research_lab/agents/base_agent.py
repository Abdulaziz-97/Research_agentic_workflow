"""Base research agent class with tools, memory, and RAG integration."""

from typing import List, Optional, Dict, Any, Type, Tuple
from abc import ABC, abstractmethod
from datetime import datetime
import uuid
import re
import logging

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent

from states.agent_state import (
    AgentState, 
    AgentStatus, 
    ResearchQuery, 
    ResearchResult,
    Paper
)
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from rag.retriever import RetrieveReflectRetryRAG
from rag.vector_store import VectorStore
from tools.web_search import ResearchToolkit
from config.settings import settings
from config.logging_config import setup_logging
from config.logging_config import setup_logging
from prompts.agent_prompts import get_base_agent_system_prompt
from agents.llm import DeepSeekChatOpenAI

# Initialize logger
logger = logging.getLogger(__name__)

class BaseResearchAgent(ABC):
    """
    Base class for all research agents.
    
    Provides core functionality including:
    - LLM integration with OpenAI
    - Tool execution
    - Short-term and long-term memory
    - RAG-based context retrieval
    - State management
    """
    
    # Class attributes to be overridden
    FIELD: str = "general"
    DISPLAY_NAME: str = "Research Agent"
    AGENT_TYPE: str = "domain"  # or "support"
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        tools: Optional[List] = None,
        memory_size: int = 10,
        llm: Optional[ChatOpenAI] = None,
        rag: Optional[RetrieveReflectRetryRAG] = None
    ):
        """
        Initialize the base research agent.
        
        Args:
            agent_id: Unique agent identifier
            tools: List of LangChain tools
            memory_size: Short-term memory size
            llm: Optional injected LLM (for testing)
            rag: Optional injected RAG (for testing)
        """
        # Use provided agent_id or generate a stable one based on field
        if agent_id:
            self.agent_id = agent_id
        else:
            self.agent_id = f"{self.FIELD}_agent"
        
        logger.info(f"Initializing {self.DISPLAY_NAME} ({self.agent_id})")
        
        # Initialize LLM
        if llm:
            self._llm = llm
        else:
            llm_kwargs = {
                "model": settings.openai_model,
                "temperature": 0.7,
                "openai_api_key": settings.openai_api_key
            }
            if settings.openai_base_url:
                llm_kwargs["openai_api_base"] = settings.openai_base_url
            
            # Use custom class to handle DeepSeek reasoning
            self._llm = DeepSeekChatOpenAI(**llm_kwargs)
        
        # Initialize tools
        self._toolkit = ResearchToolkit()
        self._tools = tools or self._get_default_tools()
        
        # Initialize memory
        self._short_term = ShortTermMemory(
            max_size=memory_size,
            agent_id=self.agent_id
        )
        self._long_term = LongTermMemory(
            agent_id=self.agent_id
        )
        
        # Initialize RAG
        if rag:
            self._rag = rag
            self._vector_store = rag.vector_store
        else:
            self._vector_store = VectorStore(
                collection_name=f"rag_{self.FIELD}"
            )
            self._rag = RetrieveReflectRetryRAG(
                vector_store=self._vector_store,
                field=self.FIELD
            )
            self._seed_rag_if_needed()
        
        # Initialize state
        self._state = AgentState(
            agent_id=self.agent_id,
            agent_type=self.AGENT_TYPE,
            field=self.FIELD,
            display_name=self.DISPLAY_NAME
        )
        
        # Build agent executor
        self._agent_executor = self._build_agent_executor()
    
    def _seed_rag_if_needed(self):
        """Seed RAG with foundational papers if enabled."""
        if settings.rag_seed_enabled and self.AGENT_TYPE == "domain":
            try:
                from rag.seed_rag import seed_rag_if_empty
                seed_rag_if_empty(
                    self._vector_store, 
                    self.FIELD, 
                    num_papers=settings.rag_seed_papers_per_field
                )
            except Exception as e:
                logger.warning(f"RAG seeding failed for {self.FIELD}: {e}")

    def _get_default_tools(self) -> List:
        """Get default tools for this agent's field."""
        return self._toolkit.get_tools_for_field(self.FIELD)
    
    def _build_agent_executor(self) -> AgentExecutor:
        """Build the LangChain agent executor."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        agent = create_openai_tools_agent(self._llm, self._tools, prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=self._tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )

    async def research(self, query: ResearchQuery) -> ResearchResult:
        """
        Conduct research on a query.
        """
        logger.info(f"Starting research for query: {query.query}")
        self._state.update_status(AgentStatus.RESEARCHING, query.query)
        
        try:
            # 1. Retrieve Context
            context, existing_papers, rag_confidence = await self._retrieve_context(query)
            
            # 2. Execute Agent
            output, result = await self._execute_agent(query, context)
            
            # 3. Process Results (Extract papers, insights)
            papers = self._process_papers(existing_papers, result, output)
            
            # 4. Update Memory
            self._update_memory(query, output, papers, rag_confidence)
            
            # 5. Build Result
            research_result = self._build_research_result(query, output, papers, rag_confidence)
            
            logger.info(f"Research completed. Found {len(papers)} papers. Confidence: {research_result.confidence_score:.2f}")
            return research_result
            
        except Exception as e:
            logger.error(f"Error during research: {e}", exc_info=True)
            self._state.update_status(AgentStatus.ERROR)
            return self._create_error_result(query, str(e))

    async def _retrieve_context(self, query: ResearchQuery) -> Tuple[str, List[Paper], float]:
        """Retrieve relevant context from RAG."""
        # Note: RAG retrieval is currently synchronous in the original code, 
        # but we wrap it here for future async support
        context, existing_papers, rag_confidence = self._rag.get_context_for_query(
            query.query
        )
        return context, existing_papers, rag_confidence

    async def _execute_agent(self, query: ResearchQuery, context: str) -> Tuple[str, Dict[str, Any]]:
        """Execute the LangChain agent."""
        chat_history = self._short_term.get_langchain_messages(limit=5)
        enhanced_input = self._build_research_input(query, context)
        
        self._state.update_status(AgentStatus.REFLECTING)
        
        result = await self._agent_executor.ainvoke({
            "input": enhanced_input,
            "chat_history": chat_history
        })
        
        output = result.get("output", "")
        return output, result

    def _process_papers(self, existing_papers: List[Paper], result: Dict[str, Any], output: str) -> List[Paper]:
        """Extract and deduplicate papers."""
        papers = existing_papers.copy()
        new_papers = self._extract_papers_from_result(result, output)
        
        existing_ids = {p.id for p in papers}
        for paper in new_papers:
            if paper.id not in existing_ids:
                papers.append(paper)
                existing_ids.add(paper.id)
        
        # Add to RAG
        if new_papers:
            try:
                self.add_papers_to_rag(new_papers)
            except Exception as e:
                logger.warning(f"Failed to add papers to RAG: {e}")
                
        return papers

    def _update_memory(self, query: ResearchQuery, output: str, papers: List[Paper], rag_confidence: float):
        """Update short-term and long-term memory."""
        self._short_term.add_user_message(query.query)
        self._short_term.add_assistant_message(output, agent_id=self.agent_id)
        
        if rag_confidence < 0.5:
            self._long_term.store_insight(
                output[:500],
                query.query,
                confidence=0.7
            )

    def _build_research_result(self, query: ResearchQuery, output: str, papers: List[Paper], rag_confidence: float) -> ResearchResult:
        """Construct the final ResearchResult object."""
        research_result = ResearchResult(
            agent_id=self.agent_id,
            agent_field=self.FIELD,
            query=query.query,
            summary=output,
            papers=papers,
            insights=self._extract_insights(output),
            confidence_score=self._calculate_confidence(output, papers),
            sources_used=query.sources_required,
            reflection_notes=self._state.reflection_notes
        )
        
        self._state.update_status(AgentStatus.RESPONDING)
        self._state.retrieved_papers = papers
        self._state.confidence_score = research_result.confidence_score
        
        return research_result

    def _create_error_result(self, query: ResearchQuery, error_str: str) -> ResearchResult:
        """Create a ResearchResult for error cases."""
        error_summary = error_str
        if "RetryError" in error_str:
            error_summary = "API request failed after multiple retries. Check network/API keys."
        elif "404" in error_str:
            error_summary = "API endpoint not found. Check configuration."
        elif "401" in error_str:
            error_summary = "Authentication failed. Check API keys."
            
        return ResearchResult(
            agent_id=self.agent_id,
            agent_field=self.FIELD,
            query=query.query,
            summary=f"Error during research: {error_summary}",
            confidence_score=0.0
        )

    def research_sync(self, query: ResearchQuery) -> ResearchResult:
        """Synchronous version of research."""
        import asyncio
        return asyncio.run(self.research(query))
    
    def _build_research_input(self, query: ResearchQuery, context: str) -> str:
        """Build enhanced input with RAG context."""
        # Keeping original logic but could be moved to prompts/
        input_parts = [
            "=" * 60,
            "RESEARCH INVESTIGATION REQUEST",
            "=" * 60,
            f"\n**Primary Research Question:**\n{query.query}\n"
        ]
        
        if query.field:
            input_parts.append(f"**Focus Domain:** {query.field}\n")
        
        if context:
            input_parts.append(f"**Relevant Prior Research Context:**\n{context}\n")
        
        input_parts.append("""
**INVESTIGATION PROTOCOL:**
1. **Literature Search**: Search arXiv, PubMed, Semantic Scholar.
2. **Analysis**: Identify consensus, debates, and quantitative results.
3. **Output Requirements**:
   - Executive Summary
   - Detailed Findings (with citations)
   - Key Papers Table
   - Research Gaps
   - Critical Assessment
4. **Citation Format**: Author et al. (Year)

**BEGIN INVESTIGATION NOW.**
""")
        return "\n".join(input_parts)
    
    def _extract_papers_from_result(self, result: Dict[str, Any], output: str) -> List[Paper]:
        """Extract Paper objects from agent execution result."""
        # TODO: Replace with Structured Output
        papers = []
        
        # Extract from tool calls (simplified)
        messages = result.get("messages", [])
        for msg in messages:
            if hasattr(msg, "content") and msg.content and "Found" in str(msg.content):
                papers.extend(self._parse_papers_from_text(str(msg.content)))
        
        if output:
            papers.extend(self._parse_papers_from_text(output))
        
        return papers
    
    def _parse_papers_from_text(self, text: str) -> List[Paper]:
        """Parse Paper objects from formatted text output."""
        # Legacy regex parsing - to be replaced
        papers = []
        lines = text.split("\n")
        current_paper = None
        
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("**")):
                title = ""
                if "**" in line:
                    match = re.search(r'\*\*(.+?)\*\*', line)
                    if match: title = match.group(1).strip()
                elif line[0].isdigit() and "." in line:
                    title = line.split(".", 1)[1].strip()
                
                if title:
                    if current_paper: papers.append(current_paper)
                    current_paper = Paper(
                        id=f"{self.FIELD}_{len(papers)}_{hash(title) % 10000}",
                        title=title,
                        field=self.FIELD
                    )
            
            if current_paper:
                if "Authors:" in line:
                    current_paper.authors = [a.strip() for a in line.split("Authors:", 1)[1].split(",")]
                if "URL:" in line:
                    match = re.search(r'https?://[^\s]+', line)
                    if match: current_paper.url = match.group(0)
        
        if current_paper: papers.append(current_paper)
        return papers
    
    def _extract_insights(self, output: str) -> List[str]:
        """Extract key insights."""
        insights = []
        for line in output.split("\n"):
            if line.strip().startswith(("- ", "• ", "* ")):
                insights.append(line.strip()[2:])
        return insights[:5]
    
    def _calculate_confidence(self, output: str, papers: List[Paper]) -> float:
        """Calculate confidence score based on evidence and output quality."""
        # Base confidence starts at 0.5
        confidence = 0.5
        
        # 1. Evidence Quantity Boost (up to 0.3)
        # +0.05 per paper, capped at 6 papers (0.3)
        if papers:
            paper_score = min(len(papers) * 0.05, 0.3)
            confidence += paper_score
            
        # 2. Output Depth Boost (up to 0.1)
        # +0.1 if output is substantial (>1000 chars)
        if len(output) > 1000:
            confidence += 0.1
        elif len(output) > 500:
            confidence += 0.05
            
        # 3. Uncertainty Penalty
        # Check for hedging language
        hedging_terms = ["uncertain", "unclear", "limited evidence", "insufficient data", "cannot confirm"]
        output_lower = output.lower()
        for term in hedging_terms:
            if term in output_lower:
                confidence -= 0.1
                
        # 4. Source Diversity Boost (up to 0.1)
        # If papers come from different domains or authors (simplified proxy: just high paper count implies diversity)
        if len(papers) >= 5:
            confidence += 0.1
            
        return min(max(confidence, 0.0), 1.0)
    
    def add_papers_to_rag(self, papers: List[Paper]):
        self._rag.add_papers(papers)
    
    def get_state(self) -> AgentState:
        return self._state
    
    def reset(self):
        self._state.reset_for_new_task()
        self._short_term.clear_working_context()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        return {
            "short_term": {"size": self._short_term.size},
            "rag": {"documents": self._vector_store.count}
        }
