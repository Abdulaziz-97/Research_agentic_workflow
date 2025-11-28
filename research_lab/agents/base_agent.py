"""Base research agent class with tools, memory, and RAG integration."""

from typing import List, Optional, Dict, Any, Type
from abc import ABC, abstractmethod
from datetime import datetime
import uuid

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
import os

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
from config.llm_factory import create_chat_model
from config.key_manager import get_key_manager


class BaseResearchAgent(ABC):
    """
    Base class for all research agents.
    
    Provides core functionality including:
    - LLM integration with the configured provider
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
        memory_size: int = 10
    ):
        """
        Initialize the base research agent.
        
        Args:
            agent_id: Unique agent identifier
            tools: List of LangChain tools
            memory_size: Short-term memory size
        """
        self.agent_id = agent_id or f"{self.FIELD}_{uuid.uuid4().hex[:8]}"
        
        # Initialize LLM (will use key manager if available, otherwise fallback to direct key)
        self._key_manager = get_key_manager()
        if self._key_manager:
            # Key manager will handle key rotation
            self._llm = None  # Will be created on-demand with current key
        else:
            # Fallback to direct key (backward compatibility)
            self._llm = create_chat_model(
                model=settings.llm_model,
                temperature=0.7
            )
        
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
        
        # Build deep agent
        self._agent_executor = self._build_deep_agent()
    
    def _get_default_tools(self) -> List:
        """Get default tools for this agent's field."""
        return self._toolkit.get_tools_for_field(self.FIELD)
    
    async def _get_llm(self):
        """Get LLM instance, using key manager if available."""
        if self._key_manager:
            return await self._key_manager.get_llm(temperature=0.7)
        return self._llm
    
    def _build_deep_agent(self):
        """Build the Deep Agent."""
        # Create a temp workspace for this agent
        workspace_dir = os.path.abspath(f"./workspaces/{self.agent_id}")
        os.makedirs(workspace_dir, exist_ok=True)
        
        # For deep agents, we need to create the LLM synchronously
        # We'll use the key manager's current key
        if self._key_manager:
            current_key = self._key_manager.get_current_key()
            llm = create_chat_model(
                api_key=current_key,
                model=settings.llm_model,
                temperature=0.7
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
        """Get the system prompt for this agent. Must be overridden."""
        pass
    
    async def research(self, query: ResearchQuery) -> ResearchResult:
        """
        Conduct research on a query.
        
        Args:
            query: Research query to investigate
            
        Returns:
            ResearchResult with findings
        """
        # Update state
        self._state.update_status(AgentStatus.RESEARCHING, query.query)
        
        try:
            # Retrieve relevant context from RAG
            context, existing_papers, rag_confidence = self._rag.get_context_for_query(
                query.query
            )
            
            # Get conversation history
            chat_history = self._short_term.get_langchain_messages(limit=5)
            
            # Build enhanced input with context
            enhanced_input = self._build_research_input(query, context)
            
            # Update to reflecting state
            self._state.update_status(AgentStatus.REFLECTING)
            
            # Execute agent with retry and key rotation
            # Convert chat history to list of messages
            messages = list(chat_history)
            messages.append(HumanMessage(content=enhanced_input))
            
            max_retries = 3
            last_error = None
            result = None
            
            for attempt in range(max_retries):
                try:
                    result = await self._agent_executor.ainvoke({
                        "messages": messages
                    })
                    
                    # Success - mark key as working
                    if self._key_manager:
                        current_key = self._key_manager.get_current_key()
                        if current_key:
                            self._key_manager.mark_key_success(current_key)
                    break
                    
                except Exception as e:
                    last_error = e
                    error_str = str(e)
                    
                    # Check if this is a key-related error
                    is_key_error = any(term in error_str.lower() for term in [
                        "insufficient", "budget", "rate limit", "429", "401", "authentication"
                    ])
                    
                    # If we have a key manager and this is a key error, try rotating to next key
                    if self._key_manager and is_key_error and attempt < max_retries - 1:
                        current_key = self._key_manager.get_current_key()
                        if current_key:
                            await self._key_manager.mark_key_failed(current_key, error_str)
                        
                        # Check if we have more keys to try
                        available = self._key_manager.get_available_keys()
                        if available:
                            # Rebuild agent with new key
                            self._agent_executor = self._build_deep_agent()
                            continue
                        else:
                            # No more keys available
                            break
                    else:
                        # Not a key error, no key manager, or no more retries - just raise
                        raise
            
            if last_error and not result:
                raise last_error
            
            # Extract thinking/reasoning steps from all messages
            thinking_steps = self._extract_thinking_steps(result["messages"])
            
            # Parse output
            output = result["messages"][-1].content
            
            # Extract papers from tool calls if available
            papers = existing_papers.copy()
            
            # Store in memory
            self._short_term.add_user_message(query.query)
            self._short_term.add_assistant_message(output, agent_id=self.agent_id)
            
            # Store important findings in long-term memory
            if rag_confidence < 0.5:  # New information
                self._long_term.store_insight(
                    output[:500],
                    query.query,
                    confidence=0.7
                )
            
            # Build result
            research_result = ResearchResult(
                agent_id=self.agent_id,
                agent_field=self.FIELD,
                query=query.query,
                summary=output,
                papers=papers,
                insights=self._extract_insights(output),
                confidence_score=self._calculate_confidence(output, papers),
                sources_used=query.sources_required,
                reflection_notes=self._state.reflection_notes,
                thinking_steps=thinking_steps
            )
            
            # Update state
            self._state.update_status(AgentStatus.RESPONDING)
            self._state.retrieved_papers = papers
            self._state.confidence_score = research_result.confidence_score
            
            return research_result
            
        except Exception as e:
            self._state.update_status(AgentStatus.ERROR)
            return ResearchResult(
                agent_id=self.agent_id,
                agent_field=self.FIELD,
                query=query.query,
                summary=f"Error during research: {str(e)}",
                confidence_score=0.0
            )
    
    def research_sync(self, query: ResearchQuery) -> ResearchResult:
        """
        Synchronous version of research.
        
        Args:
            query: Research query
            
        Returns:
            ResearchResult
        """
        import asyncio
        return asyncio.run(self.research(query))
    
    def _build_research_input(
        self, 
        query: ResearchQuery, 
        context: str
    ) -> str:
        """Build enhanced input with RAG context for rigorous academic research."""
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

1. **Literature Search Phase**
   - Search multiple databases (arXiv, PubMed, Semantic Scholar) for relevant papers
   - Prioritize recent publications (last 5 years) but include seminal works
   - Search for both broad reviews and specific empirical studies

2. **Analysis Requirements**
   - For each relevant paper found, note: Authors, Year, Key findings, Methodology
   - Identify consensus views in the field
   - Highlight ongoing debates or contradictions
   - Note quantitative results (effect sizes, p-values, sample sizes when available)

3. **Output Format Requirements**
   Your response MUST include:
   
   a) **Executive Summary** (2-3 sentences of key findings)
   
   b) **Detailed Findings** organized by theme or methodology:
      - Each finding must cite the specific paper/author
      - Include direct quotes where particularly insightful
      - Note the strength of evidence (meta-analysis > RCT > observational > theoretical)
   
   c) **Key Papers Table**:
      | Authors (Year) | Title | Key Finding | Relevance |
      
   d) **Research Gaps**: What questions remain unanswered?
   
   e) **Critical Assessment**: What are the limitations of current research?

4. **Citation Format**
   Always cite as: Author et al. (Year) or (Author, Year)
   Be specific - don't say "studies show" without citing which studies

**BEGIN INVESTIGATION NOW. Search thoroughly and report comprehensively.**
""")
        
        return "\n".join(input_parts)
    
    def _extract_insights(self, output: str) -> List[str]:
        """Extract key insights from research output."""
        insights = []
        
        # Simple extraction - look for bullet points or numbered items
        lines = output.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith(("- ", "• ", "* ", "1.", "2.", "3.")):
                insight = line.lstrip("-•* 0123456789.")
                if len(insight) > 20:  # Meaningful insight
                    insights.append(insight.strip())
        
        return insights[:5]  # Top 5 insights
    
    def _extract_thinking_steps(self, messages: List) -> List[Dict[str, Any]]:
        """Extract thinking/reasoning steps from agent messages (Gemini-style)."""
        thinking_steps = []
        step_id = 0
        
        for msg in messages:
            if isinstance(msg, AIMessage):
                step_id += 1
                step = {
                    "step_id": step_id,
                    "agent_id": self.agent_id,
                    "agent_name": self.DISPLAY_NAME,
                    "timestamp": datetime.now().isoformat(),
                    "reasoning": "",
                    "tool_calls": [],
                    "tool_results": [],
                    "content": ""
                }
                
                # Extract content blocks (deepagents format)
                if hasattr(msg, 'content_blocks') or isinstance(msg.content, list):
                    content_blocks = msg.content if isinstance(msg.content, list) else getattr(msg, 'content_blocks', [])
                    
                    for block in content_blocks:
                        if isinstance(block, dict):
                            block_type = block.get("type", "")
                            
                            # Reasoning block
                            if block_type == "reasoning" or "reasoning" in block:
                                step["reasoning"] = block.get("reasoning", block.get("text", ""))
                            
                            # Tool call
                            elif block_type == "tool_call" or "tool_calls" in block:
                                tool_calls = block.get("tool_calls", [block] if "name" in block else [])
                                for tc in tool_calls:
                                    step["tool_calls"].append({
                                        "name": tc.get("name", "unknown"),
                                        "args": tc.get("args", tc.get("input", {})),
                                        "id": tc.get("id", "")
                                    })
                            
                            # Text content
                            elif block_type == "text" or isinstance(block, str):
                                text = block.get("text", "") if isinstance(block, dict) else str(block)
                                if text:
                                    step["content"] += text + "\n"
                        elif isinstance(block, str):
                            step["content"] += block + "\n"
                
                # Fallback: extract from message content directly
                if not step["content"] and hasattr(msg, 'content'):
                    content = msg.content
                    if isinstance(content, str):
                        step["content"] = content
                
                # Extract tool calls from additional_kwargs
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        step["tool_calls"].append({
                            "name": getattr(tc, 'name', 'unknown'),
                            "args": getattr(tc, 'args', {}),
                            "id": getattr(tc, 'id', '')
                        })
                
                # Extract tool results from following messages
                # (This is a simplified version - in practice, tool results come in subsequent messages)
                
                if step["reasoning"] or step["tool_calls"] or step["content"]:
                    thinking_steps.append(step)
        
        return thinking_steps
    
    def _calculate_confidence(
        self, 
        output: str, 
        papers: List[Paper]
    ) -> float:
        """Calculate confidence score for the research result."""
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on papers found
        if papers:
            confidence += min(len(papers) * 0.05, 0.25)
        
        # Increase confidence based on output length (more thorough)
        if len(output) > 500:
            confidence += 0.1
        if len(output) > 1000:
            confidence += 0.1
        
        # Check for uncertainty markers
        uncertainty_markers = ["uncertain", "unclear", "might", "possibly", "may be"]
        for marker in uncertainty_markers:
            if marker in output.lower():
                confidence -= 0.05
        
        return min(max(confidence, 0.0), 1.0)
    
    def add_papers_to_rag(self, papers: List[Paper]):
        """Add papers to the RAG system."""
        self._rag.add_papers(papers)
    
    def get_state(self) -> AgentState:
        """Get current agent state."""
        return self._state
    
    def reset(self):
        """Reset agent for new task."""
        self._state.reset_for_new_task()
        self._short_term.clear_working_context()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "short_term": {
                "size": self._short_term.size,
                "max_size": self._short_term.max_size
            },
            "long_term": self._long_term.get_stats(),
            "rag": {
                "documents": self._vector_store.count
            }
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id}, field={self.FIELD})"

