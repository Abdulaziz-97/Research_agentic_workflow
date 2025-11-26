"""Short-term memory implementation using conversation buffer."""

from typing import List, Optional, Dict, Any
from datetime import datetime
from collections import deque
import uuid

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field

from states.agent_state import MemoryEntry


class ConversationTurn(BaseModel):
    """Represents a single conversation turn."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: str = Field(..., description="user or assistant")
    content: str = Field(..., description="Message content")
    agent_id: Optional[str] = Field(None, description="Agent that responded")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class ShortTermMemory:
    """
    Short-term memory for agents using a sliding window buffer.
    
    Stores recent conversation history and working context for the current
    research session. Automatically manages capacity by removing oldest
    entries when limit is reached.
    """
    
    def __init__(self, max_size: int = 10, agent_id: str = "default"):
        """
        Initialize short-term memory.
        
        Args:
            max_size: Maximum number of conversation turns to keep
            agent_id: ID of the agent this memory belongs to
        """
        self.max_size = max_size
        self.agent_id = agent_id
        self._buffer: deque[ConversationTurn] = deque(maxlen=max_size)
        self._working_context: Dict[str, Any] = {}
        self._created_at = datetime.now()
    
    def add_message(
        self, 
        role: str, 
        content: str, 
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConversationTurn:
        """
        Add a message to short-term memory.
        
        Args:
            role: "user" or "assistant"
            content: Message content
            agent_id: Optional agent ID for assistant messages
            metadata: Optional additional metadata
            
        Returns:
            The created ConversationTurn
        """
        turn = ConversationTurn(
            role=role,
            content=content,
            agent_id=agent_id or self.agent_id,
            metadata=metadata or {}
        )
        self._buffer.append(turn)
        return turn
    
    def add_user_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> ConversationTurn:
        """Add a user message."""
        return self.add_message("user", content, metadata=metadata)
    
    def add_assistant_message(
        self, 
        content: str, 
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConversationTurn:
        """Add an assistant message."""
        return self.add_message("assistant", content, agent_id=agent_id, metadata=metadata)
    
    def get_messages(self, limit: Optional[int] = None) -> List[ConversationTurn]:
        """
        Get recent messages from memory.
        
        Args:
            limit: Optional limit on number of messages to return
            
        Returns:
            List of conversation turns, oldest first
        """
        messages = list(self._buffer)
        if limit:
            return messages[-limit:]
        return messages
    
    def get_langchain_messages(self, limit: Optional[int] = None) -> List[BaseMessage]:
        """
        Get messages in LangChain format.
        
        Args:
            limit: Optional limit on number of messages
            
        Returns:
            List of LangChain BaseMessage objects
        """
        turns = self.get_messages(limit)
        messages = []
        
        for turn in turns:
            if turn.role == "user":
                messages.append(HumanMessage(content=turn.content))
            else:
                messages.append(AIMessage(
                    content=turn.content,
                    additional_kwargs={"agent_id": turn.agent_id}
                ))
        
        return messages
    
    def get_context_string(self, limit: Optional[int] = None) -> str:
        """
        Get conversation history as a formatted string.
        
        Args:
            limit: Optional limit on number of messages
            
        Returns:
            Formatted conversation string
        """
        turns = self.get_messages(limit)
        lines = []
        
        for turn in turns:
            prefix = "User" if turn.role == "user" else f"Assistant ({turn.agent_id})"
            lines.append(f"{prefix}: {turn.content}")
        
        return "\n".join(lines)
    
    def set_working_context(self, key: str, value: Any):
        """Set a value in working context."""
        self._working_context[key] = value
    
    def get_working_context(self, key: str, default: Any = None) -> Any:
        """Get a value from working context."""
        return self._working_context.get(key, default)
    
    def clear_working_context(self):
        """Clear all working context."""
        self._working_context = {}
    
    def get_all_working_context(self) -> Dict[str, Any]:
        """Get all working context."""
        return self._working_context.copy()
    
    def clear(self):
        """Clear all short-term memory."""
        self._buffer.clear()
        self._working_context = {}
    
    def to_memory_entries(self) -> List[MemoryEntry]:
        """
        Convert buffer to MemoryEntry objects for potential long-term storage.
        
        Returns:
            List of MemoryEntry objects
        """
        entries = []
        for turn in self._buffer:
            entries.append(MemoryEntry(
                id=turn.id,
                content=f"{turn.role}: {turn.content}",
                memory_type="short_term",
                agent_id=turn.agent_id or self.agent_id,
                importance=0.5,  # Default importance
                metadata={
                    "role": turn.role,
                    "original_timestamp": turn.timestamp.isoformat()
                },
                created_at=turn.timestamp
            ))
        return entries
    
    @property
    def size(self) -> int:
        """Current size of the buffer."""
        return len(self._buffer)
    
    @property
    def is_empty(self) -> bool:
        """Check if memory is empty."""
        return len(self._buffer) == 0
    
    @property
    def is_full(self) -> bool:
        """Check if memory is at capacity."""
        return len(self._buffer) >= self.max_size
    
    def __len__(self) -> int:
        return self.size
    
    def __repr__(self) -> str:
        return f"ShortTermMemory(agent_id={self.agent_id}, size={self.size}/{self.max_size})"

