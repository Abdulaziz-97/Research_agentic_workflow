"""Custom LLM wrappers."""

from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

class DeepSeekChatOpenAI(ChatOpenAI):
    """Custom ChatOpenAI that handles DeepSeek's reasoning_content field."""
    
    def _create_message_dicts(
        self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        dicts = super()._create_message_dicts(messages, stop)
        
        # Log the full prompt being sent
        import logging
        logger = logging.getLogger(__name__)
        try:
            import json
            logger.debug(f"LLM REQUEST MESSAGES:\n{json.dumps(dicts, indent=2, ensure_ascii=False)}")
        except:
            pass
            
        for d, m in zip(dicts, messages):
            if isinstance(m, AIMessage) and "reasoning_content" in m.additional_kwargs:
                d["reasoning_content"] = m.additional_kwargs["reasoning_content"]
        return dicts

    def _create_chat_result(self, response: Any, *args: Any, **kwargs: Any) -> Any:
        """Override to capture reasoning_content from the API response."""
        chat_result = super()._create_chat_result(response, *args, **kwargs)
        
        # Handle Pydantic object (ChatCompletion) or dict
        choices = []
        if hasattr(response, "choices"):
            choices = response.choices
        elif isinstance(response, dict) and "choices" in response:
            choices = response["choices"]
            
        for i, choice in enumerate(choices):
            reasoning = None
            # Check if choice is object or dict
            if hasattr(choice, "message"):
                message = choice.message
                if hasattr(message, "reasoning_content"):
                    reasoning = message.reasoning_content
                # Also check model_extra/additional_properties if not direct attribute
                elif hasattr(message, "model_extra") and message.model_extra:
                    reasoning = message.model_extra.get("reasoning_content")
            elif isinstance(choice, dict):
                message = choice.get("message", {})
                reasoning = message.get("reasoning_content")
            
            if reasoning and i < len(chat_result.generations):
                # Add to additional_kwargs of the generated message
                chat_result.generations[i].message.additional_kwargs["reasoning_content"] = reasoning
                
                # Log reasoning for visibility
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"\n{'='*40}\nAGENT REASONING:\n{reasoning}\n{'='*40}\n")
                    
        return chat_result

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ):
        """Override _astream to force non-streaming and capture reasoning_content."""
        # Fallback to _agenerate to use our patched _create_chat_result
        # This effectively disables streaming but ensures we capture the reasoning_content
        result = await self._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        
        # Convert ChatResult to ChatGenerationChunk
        for generation in result.generations:
            # Create a chunk that looks like the full message
            message_chunk = AIMessageChunk(
                content=generation.message.content,
                additional_kwargs=generation.message.additional_kwargs,
                response_metadata=generation.message.response_metadata,
                id=generation.message.id
            )
            
            yield ChatGenerationChunk(
                message=message_chunk,
                generation_info=generation.generation_info
            )
