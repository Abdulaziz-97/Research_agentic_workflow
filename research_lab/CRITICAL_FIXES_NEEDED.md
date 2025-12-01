# 🚨 CRITICAL FIXES NEEDED - Workflow Failing

## Current Status: ❌ BROKEN

**All agents failing:**
- ❌ AI/ML: 404 error (embeddings)
- ❌ Biology: 404 error (embeddings)  
- ❌ Chemistry: RetryError (chat API)
- ❌ **0 papers found**
- ❌ **0% confidence**

---

## 🔴 Issue #1: Invalid API Key

**Problem**: Your `.env` file has:
```env
OPENAI_API_KEY=a
```

**This is WRONG!** The API key is just "a" - it needs to be a real DeepSeek API key.

**Fix**:
1. Get your DeepSeek API key from: https://platform.deepseek.com/
2. Update `.env`:
```env
OPENAI_API_KEY=sk-your-actual-deepseek-key-here
```

---

## 🔴 Issue #2: Embeddings Still Using OpenAI (404 Error)

**Problem**: Even though `EMBEDDINGS_PROVIDER=bge-m3` is set, you're getting 404 errors from embeddings API.

**Possible Causes**:
1. **BGE-M3 not installed properly**
2. **Settings not being read correctly**
3. **Some code path still using OpenAI**

**Fix Steps**:

1. **Verify BGE-M3 is installed**:
```bash
pip install FlagEmbedding
```

2. **Verify .env file**:
```env
EMBEDDINGS_PROVIDER=bge-m3
# Remove or comment out these lines:
# OPENAI_EMBEDDINGS_API_KEY=...
# OPENAI_EMBEDDINGS_BASE_URL=...
```

3. **Restart Streamlit** (settings are loaded at startup)

---

## 🔴 Issue #3: DeepSeek API Configuration

**Your current config**:
```env
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-reasoner
```

**Check**:
1. ✅ Base URL is correct: `https://api.deepseek.com`
2. ⚠️ Model name: Verify `deepseek-reasoner` is correct
   - Common models: `deepseek-chat`, `deepseek-coder`
   - Check DeepSeek docs for exact model name

3. ⚠️ **API Key must be valid** - Currently set to "a" which is wrong!

---

## ✅ Complete .env File (Corrected)

```env
# OpenAI/DeepSeek API Configuration
OPENAI_API_KEY=sk-your-actual-deepseek-key-here
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat  # or deepseek-reasoner if that's correct

# Embeddings Configuration (BGE-M3 - FREE, LOCAL)
EMBEDDINGS_PROVIDER=bge-m3
# DO NOT SET OPENAI_EMBEDDINGS_API_KEY when using BGE-M3
# DO NOT SET OPENAI_EMBEDDINGS_BASE_URL when using BGE-M3

# Tavily Configuration
TAVILY_API_KEY=your-tavily-key-here

# ChromaDB Configuration
CHROMA_PERSIST_DIRECTORY=./data/chroma_db
CHROMA_COLLECTION_PREFIX=research_lab

# Memory Configuration
SHORT_TERM_MEMORY_SIZE=10
LONG_TERM_MEMORY_THRESHOLD=0.7

# Research Tools Configuration
ARXIV_MAX_RESULTS=10
SEMANTIC_SCHOLAR_MAX_RESULTS=10
PUBMED_MAX_RESULTS=10

# Agent Configuration
MAX_RETRIES=3
AGENT_TIMEOUT=60

# Streamlit Configuration
STREAMLIT_PORT=8501
```

---

## 🔧 Quick Fix Checklist

- [ ] **Get real DeepSeek API key** from https://platform.deepseek.com/
- [ ] **Update OPENAI_API_KEY** in `.env` with real key
- [ ] **Verify EMBEDDINGS_PROVIDER=bge-m3** in `.env`
- [ ] **Remove/comment out** OPENAI_EMBEDDINGS_API_KEY and OPENAI_EMBEDDINGS_BASE_URL
- [ ] **Verify BGE-M3 installed**: `pip install FlagEmbedding`
- [ ] **Restart Streamlit** completely
- [ ] **Test with simple query**

---

## 🧪 Test After Fix

Run this to verify configuration:
```bash
cd research_lab
python -c "from config.settings import settings; print('✅ Embeddings:', settings.embeddings_provider); print('✅ Base URL:', settings.openai_base_url); print('✅ Model:', settings.openai_model); print('✅ API Key valid:', len(settings.openai_api_key) > 20 if settings.openai_api_key else False)"
```

---

## 📊 Expected After Fix

- ✅ BGE-M3 loads once (no 404 errors)
- ✅ Agents can make API calls (no RetryError)
- ✅ Papers are found and processed
- ✅ Confidence scores > 0%
- ✅ Workflow completes successfully

---

**Priority**: 🔴 **CRITICAL** - System is completely broken until these are fixed.


