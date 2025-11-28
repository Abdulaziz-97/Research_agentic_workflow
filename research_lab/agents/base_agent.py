"""Base research agent class with tools, memory, and RAG integration."""

from typing import List, Optional, Dict, Any, Type
from abc import ABC, abstractmethod
from datetime import datetime
import uuid
import re

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
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
        memory_size: int = 10
    ):
        """
        Initialize the base research agent.
        
        Args:
            agent_id: Unique agent identifier
            tools: List of LangChain tools
            memory_size: Short-term memory size
        """
        # Use provided agent_id or generate a stable one based on field
        # This ensures agents of the same field share the same RAG across sessions
        if agent_id:
            self.agent_id = agent_id
        else:
            # Generate stable ID: field-based (not random) so RAG persists across sessions
            # Format: "ai_ml_agent", "physics_agent", etc.
            self.agent_id = f"{self.FIELD}_agent"
        
        # Initialize LLM
        llm_kwargs = {
            "model": settings.openai_model,
            "temperature": 0.7,
            "openai_api_key": settings.openai_api_key
        }
        # Add base URL if configured (for custom endpoints like Vocareum)
        if settings.openai_base_url:
            llm_kwargs["openai_api_base"] = settings.openai_base_url
        
        self._llm = ChatOpenAI(**llm_kwargs)
        
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
        
        # Initialize RAG with FIELD-based collection name
        # This ensures all agents of the same field share the same RAG collection
        # Papers persist across sessions and are accessible to any agent of that field
        # Format: "rag_ai_ml", "rag_physics", etc. (not per-agent-instance)
        self._vector_store = VectorStore(
            collection_name=f"rag_{self.FIELD}"
        )
        self._rag = RetrieveReflectRetryRAG(
            vector_store=self._vector_store,
            field=self.FIELD
        )
        
        # Seed RAG with foundational papers if collection is empty (one-time operation)
        # Only seed domain agents, not support agents
        # This provides initial context and improves first-query performance
        if settings.rag_seed_enabled and self.AGENT_TYPE == "domain":
            try:
                from rag.seed_rag import seed_rag_if_empty
                seed_rag_if_empty(
                    self._vector_store, 
                    self.FIELD, 
                    num_papers=settings.rag_seed_papers_per_field
                )
            except Exception as e:
                # Don't fail agent initialization if seeding fails
                print(f"Warning: RAG seeding failed for {self.FIELD}: {e}")
        
        # Initialize state
        self._state = AgentState(
            agent_id=self.agent_id,
            agent_type=self.AGENT_TYPE,
            field=self.FIELD,
            display_name=self.DISPLAY_NAME
        )
        
        # Build agent executor
        self._agent_executor = self._build_agent_executor()
    
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
        
        agent = create_openai_tools_agent(
            llm=self._llm,
            tools=self._tools,
            prompt=prompt
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self._tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
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
            
            # Execute agent
            result = await self._agent_executor.ainvoke({
                "input": enhanced_input,
                "chat_history": chat_history
            })
            
            # Parse output
            output = result.get("output", "")
            
            # Extract papers from tool calls and agent messages
            papers = existing_papers.copy()
            new_papers = self._extract_papers_from_result(result, output)
            
            # Add newly found papers to the list (avoid duplicates)
            existing_ids = {p.id for p in papers}
            for paper in new_papers:
                if paper.id not in existing_ids:
                    papers.append(paper)
                    existing_ids.add(paper.id)
            
            # Store newly found papers in RAG for future queries
            if new_papers:
                try:
                    self.add_papers_to_rag(new_papers)
                except Exception as e:
                    # Log but don't fail if RAG storage fails
                    print(f"Warning: Failed to add papers to RAG: {e}")
            
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
                reflection_notes=self._state.reflection_notes
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
    
    def _extract_papers_from_result(
        self, 
        result: Dict[str, Any], 
        output: str
    ) -> List[Paper]:
        """
        Extract Paper objects from agent execution result.
        
        Attempts to extract papers from:
        1. Tool call results in agent messages
        2. Parsed output text (if papers are mentioned)
        
        Args:
            result: Agent executor result dictionary
            output: Final output text
            
        Returns:
            List of Paper objects found
        """
        papers = []
        
        # Try to extract from agent messages (tool calls)
        messages = result.get("messages", [])
        for msg in messages:
            # Check if message has tool calls
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                # Tool calls are in the message, but results are in subsequent messages
                # We'll need to parse the output text for now
                pass
            
            # Check message content for paper information
            if hasattr(msg, "content") and msg.content:
                content = str(msg.content)
                # Look for paper patterns in tool results
                # Tools return formatted strings like "Found X papers: ..."
                if "Found" in content and "papers" in content.lower():
                    # Try to parse papers from formatted tool output
                    parsed = self._parse_papers_from_text(content)
                    papers.extend(parsed)
        
        # Also try to extract from final output
        if output:
            parsed = self._parse_papers_from_text(output)
            papers.extend(parsed)
        
        # Remove duplicates by ID
        seen_ids = set()
        unique_papers = []
        for paper in papers:
            if paper.id and paper.id not in seen_ids:
                seen_ids.add(paper.id)
                unique_papers.append(paper)
        
        return unique_papers
    
    def _parse_papers_from_text(self, text: str) -> List[Paper]:
        """
        Parse Paper objects from formatted text output.
        
        Attempts to extract paper information from tool output formats like:
        - "Found X papers: 1. **Title** Authors: ... URL: ..."
        - Arxiv/Semantic Scholar formatted outputs
        
        Args:
            text: Text containing paper information
            
        Returns:
            List of Paper objects
        """
        papers = []
        lines = text.split("\n")
        
        current_paper = None
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Look for paper title patterns (numbered lists with bold titles)
            if line and (line[0].isdigit() or line.startswith("**")):
                # Extract title
                title = ""
                if "**" in line:
                    # Extract text between ** markers
                    import re
                    title_match = re.search(r'\*\*(.+?)\*\*', line)
                    if title_match:
                        title = title_match.group(1).strip()
                elif line[0].isdigit() and "." in line:
                    # Format: "1. Title" or "1. **Title**"
                    title_part = line.split(".", 1)[1].strip()
                    title_match = re.search(r'\*\*(.+?)\*\*', title_part)
                    if title_match:
                        title = title_match.group(1).strip()
                    else:
                        title = title_part.strip()
                
                if title:
                    # Create new paper
                    if current_paper:
                        papers.append(current_paper)
                    current_paper = Paper(
                        id=f"{self.FIELD}_{len(papers)}_{hash(title) % 1000000}",
                        title=title,
                        authors=[],
                        abstract="",
                        url="",
                        source="unknown",
                        field=self.FIELD
                    )
            
            # Extract authors
            if current_paper and "Authors:" in line:
                authors_text = line.split("Authors:", 1)[1].strip()
                # Remove "et al." if present
                authors_text = authors_text.replace("et al.", "").strip()
                if authors_text:
                    # Split by comma
                    authors = [a.strip() for a in authors_text.split(",")]
                    current_paper.authors = authors
            
            # Extract URL
            if current_paper and ("URL:" in line or "http" in line):
                url_match = re.search(r'https?://[^\s]+', line)
                if url_match:
                    current_paper.url = url_match.group(0)
            
            # Extract abstract
            if current_paper and "Abstract:" in line:
                abstract_text = line.split("Abstract:", 1)[1].strip()
                if abstract_text:
                    current_paper.abstract = abstract_text
            elif current_paper and current_paper.abstract == "" and line and not any(
                marker in line for marker in ["Authors:", "URL:", "Field:", "**"]
            ):
                # Continuation of abstract
                if len(line) > 20:
                    current_paper.abstract += " " + line
        
        # Add last paper if exists
        if current_paper:
            papers.append(current_paper)
        
        return papers
    
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

