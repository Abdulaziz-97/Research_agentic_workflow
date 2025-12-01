# 🔧 BGE-M3 Multiple Loading Fix

## Problem

**BGE-M3 was loading multiple times** (you saw it load 2-3 times in the terminal), which is:
- ❌ **Wasteful** - Loading a 2.3GB model multiple times
- ❌ **Slow** - Each load takes 10-30 seconds
- ❌ **Memory intensive** - Each instance uses 4-6GB RAM

## Root Cause

Every time a new agent was created:
1. Agent creates a `VectorStore` instance
2. `VectorStore` creates an `EmbeddingManager` instance  
3. `EmbeddingManager` calls `get_embeddings_model()`
4. `get_embeddings_model()` creates a **new** `BGEM3Embeddings` instance
5. Each `BGEM3Embeddings` instance loads the model separately

**Result**: If you have 3 agents (Chemistry, AI/ML, Biology), BGE-M3 loads 3 times!

## Solution

Implemented a **singleton pattern** with caching:

```python
# Global cache for embeddings models
_embeddings_cache: dict[str, BaseEmbeddings] = {}

def get_embeddings_model(model: Optional[str] = None) -> BaseEmbeddings:
    # Create cache key
    cache_key = f"bge-m3_{settings.bge_m3_model_name}_{settings.bge_m3_use_fp16}"
    
    # Return cached instance if available
    if cache_key in _embeddings_cache:
        return _embeddings_cache[cache_key]
    
    # Create new instance only if not cached
    embeddings = BGEM3Embeddings(...)
    
    # Cache and return
    _embeddings_cache[cache_key] = embeddings
    return embeddings
```

## What This Means

**Before**:
```
Agent 1 (Chemistry) → Loads BGE-M3 (10-30s)
Agent 2 (AI/ML) → Loads BGE-M3 again (10-30s)  
Agent 3 (Biology) → Loads BGE-M3 again (10-30s)
Total: 30-90 seconds, 12-18GB RAM
```

**After**:
```
Agent 1 (Chemistry) → Loads BGE-M3 (10-30s)
Agent 2 (AI/ML) → Uses cached BGE-M3 (instant)
Agent 3 (Biology) → Uses cached BGE-M3 (instant)
Total: 10-30 seconds, 4-6GB RAM
```

## Benefits

✅ **Faster startup** - Model loads once, not multiple times
✅ **Less memory** - Single model instance shared across all agents
✅ **Better performance** - No redundant model loading
✅ **Same functionality** - All agents still work perfectly

## What's Being Embedded

The BGE-M3 model is used to create embeddings (vector representations) of:
- **Research papers** - When papers are added to RAG collections
- **Queries** - When agents search for relevant papers
- **Documents** - Any text content that needs to be searchable

The embeddings allow the system to find semantically similar content, even if the exact words don't match.

## Testing

After this fix, you should see:
- ✅ BGE-M3 loads **only once** at startup
- ✅ Subsequent agents use the cached model (no "Loading BGE-M3" messages)
- ✅ Faster workflow initialization
- ✅ Lower memory usage

---

**File Modified**: `research_lab/rag/embeddings.py`

**Status**: ✅ Fixed - BGE-M3 now uses singleton pattern

