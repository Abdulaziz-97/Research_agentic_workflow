"""API Key Manager with automatic rotation and fallback."""

import os
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import asyncio
from config.llm_factory import create_chat_model


class KeyManager:
    """
    Manages multiple API keys with automatic rotation and fallback.
    
    When a key fails (rate limit, budget, etc.), automatically tries the next one.
    Tracks key health and temporarily disables failed keys.
    """
    
    def __init__(
        self,
        keys: List[str],
        base_url: Optional[str] = None,
        model: str = "gpt-4o",
        provider: str = "openai",
    ):
        """
        Initialize the key manager.
        
        Args:
            keys: List of API keys to use
            base_url: Base URL for the API (e.g., Vocareum endpoint)
            model: Model to use
        """
        self.keys = [k.strip() for k in keys if k.strip()]
        self.base_url = base_url
        self.model = model
        self.provider = provider
        
        # Track key health: {key: {"failed_at": datetime, "failure_count": int, "disabled_until": datetime}}
        self.key_health: Dict[str, Dict] = {key: {"failure_count": 0, "disabled_until": None} for key in self.keys}
        
        # Current key index
        self.current_index = 0
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
    
    def get_all_keys(self) -> List[str]:
        """Get all available keys."""
        return self.keys.copy()
    
    def get_current_key(self) -> Optional[str]:
        """Get the current active key."""
        if not self.keys:
            return None
        return self.keys[self.current_index]
    
    def get_available_keys(self) -> List[str]:
        """Get keys that are currently available (not disabled)."""
        available = []
        now = datetime.now()
        
        for key in self.keys:
            health = self.key_health.get(key, {})
            disabled_until = health.get("disabled_until")
            
            # Key is available if not disabled or disabled period has passed
            if disabled_until is None or now > disabled_until:
                available.append(key)
        
        return available
    
    async def mark_key_failed(self, key: str, error: Optional[str] = None):
        """
        Mark a key as failed and disable it temporarily.
        
        Args:
            key: The key that failed
            error: Error message (to determine disable duration)
        """
        async with self._lock:
            if key not in self.key_health:
                return
            
            health = self.key_health[key]
            health["failure_count"] = health.get("failure_count", 0) + 1
            health["failed_at"] = datetime.now()
            
            # Determine disable duration based on error type
            if error:
                error_lower = error.lower()
                if "rate limit" in error_lower or "429" in error:
                    # Rate limit: disable for 1 minute
                    health["disabled_until"] = datetime.now() + timedelta(minutes=1)
                elif "budget" in error_lower or "insufficient" in error_lower:
                    # Budget issue: disable for 1 hour (likely needs manual intervention)
                    health["disabled_until"] = datetime.now() + timedelta(hours=1)
                elif "401" in error or "authentication" in error_lower:
                    # Auth error: disable permanently (until restart)
                    health["disabled_until"] = datetime.now() + timedelta(days=365)
                else:
                    # Other error: disable for 5 minutes
                    health["disabled_until"] = datetime.now() + timedelta(minutes=5)
            else:
                # Default: disable for 5 minutes
                health["disabled_until"] = datetime.now() + timedelta(minutes=5)
            
            # Move to next key
            self._rotate_to_next_available()
    
    def mark_key_success(self, key: str):
        """Mark a key as successful (reset failure count after some successes)."""
        if key in self.key_health:
            health = self.key_health[key]
            # Reset failure count after 10 successful calls
            if health.get("failure_count", 0) > 0:
                health["failure_count"] = max(0, health.get("failure_count", 0) - 1)
            health["disabled_until"] = None
    
    def reset_all_keys(self):
        """Reset all keys - clear failures and re-enable all keys."""
        now = datetime.now()
        for key in self.keys:
            if key in self.key_health:
                health = self.key_health[key]
                # Only reset if disabled period has passed or force reset
                if health.get("disabled_until") and now > health.get("disabled_until"):
                    health["disabled_until"] = None
                    health["failure_count"] = 0
                elif health.get("disabled_until") is None:
                    # Already enabled, just reset failure count
                    health["failure_count"] = 0
        
        # Reset to first key
        self.current_index = 0
    
    def force_reset_all_keys(self):
        """Force reset all keys immediately (ignore disable timers)."""
        for key in self.keys:
            if key in self.key_health:
                health = self.key_health[key]
                health["disabled_until"] = None
                health["failure_count"] = 0
        
        # Reset to first key
        self.current_index = 0
    
    def _rotate_to_next_available(self):
        """Rotate to the next available key."""
        available = self.get_available_keys()
        
        if not available:
            # All keys disabled, reset and try again from beginning
            # This allows retrying disabled keys after their timeout
            self.current_index = (self.current_index + 1) % len(self.keys) if self.keys else 0
            return
        
        # Find current key in available list
        current_key = self.get_current_key()
        if current_key in available:
            current_idx = available.index(current_key)
            next_idx = (current_idx + 1) % len(available)
        else:
            # Current key not available, use first available
            next_idx = 0
        
        # Find the index in the full keys list
        if available:
            next_key = available[next_idx]
            self.current_index = self.keys.index(next_key)
        else:
            # No available keys, just rotate to next in full list
            self.current_index = (self.current_index + 1) % len(self.keys) if self.keys else 0
    
    async def get_llm(self, **kwargs):
        """
        Get a chat model instance with the current active key.
        
        Args:
            **kwargs: Additional arguments to pass to the LLM factory
        
        Returns:
            LangChain-compatible chat model configured with current key
        """
        key = self.get_current_key()
        if not key:
            raise ValueError("No API keys available")
        
        return create_chat_model(
            api_key=key,
            model=self.model,
            base_url=self.base_url,
            **kwargs
        )
    
    async def test_key(self, key: str) -> tuple[bool, Optional[str]]:
        """
        Test if a key is working.
        
        Args:
            key: API key to test
        
        Returns:
            (success: bool, error_message: Optional[str])
        """
        try:
            llm = create_chat_model(
                api_key=key,
                model=self.model,
                base_url=self.base_url,
                temperature=0.1,
                max_tokens=10
            )
            
            await llm.ainvoke("test")
            return True, None
        except Exception as e:
            return False, str(e)
    
    def get_status(self) -> Dict:
        """Get status of all keys."""
        now = datetime.now()
        status = {
            "total_keys": len(self.keys),
            "available_keys": len(self.get_available_keys()),
            "current_key": self.get_current_key()[:20] + "..." if self.get_current_key() else None,
            "keys": []
        }
        
        for key in self.keys:
            health = self.key_health.get(key, {})
            disabled_until = health.get("disabled_until")
            is_available = disabled_until is None or now > disabled_until
            
            key_status = {
                "key": key[:20] + "...",
                "available": is_available,
                "failure_count": health.get("failure_count", 0),
                "disabled_until": disabled_until.isoformat() if disabled_until and now < disabled_until else None,
                "is_current": key == self.get_current_key()
            }
            status["keys"].append(key_status)
        
        return status


# Global key manager instance
_key_manager: Optional[KeyManager] = None


def initialize_key_manager(
    keys: List[str],
    base_url: Optional[str] = None,
    model: str = "gpt-4o",
    provider: str = "openai",
) -> KeyManager:
    """Initialize the global key manager."""
    global _key_manager
    _key_manager = KeyManager(keys, base_url, model, provider)
    return _key_manager


def get_key_manager() -> Optional[KeyManager]:
    """Get the global key manager instance."""
    return _key_manager

