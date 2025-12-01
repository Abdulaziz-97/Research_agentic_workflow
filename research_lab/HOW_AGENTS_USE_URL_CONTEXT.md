# How Agents Use URL Context Tool - Step-by-Step

## 🔄 **Automatic Agent Workflow**

Agents automatically use the URL context tool as part of their research process. Here's exactly how it works:

---

## 📋 **Step-by-Step Process**

### **Step 1: Agent Receives Query**
```python
User Query: "What are recent advances in transformer architectures?"

Agent (AI/ML): Receives query
```

### **Step 2: Agent Searches for Papers**
```python
Agent uses tools:
1. arxiv_search("transformer architectures", max_results=10)
   → Returns: List of papers with URLs
   
2. semantic_scholar_search("transformer architectures")
   → Returns: More papers with URLs
```

**Example Tool Output:**
```
Found 10 papers:

1. **Attention Is All You Need**
   Authors: Vaswani et al.
   URL: https://arxiv.org/abs/1706.03762
   Abstract: We propose a new simple network architecture...

2. **BERT: Pre-training of Deep Bidirectional Transformers**
   URL: https://arxiv.org/abs/1810.04805
   ...
```

### **Step 3: Agent Automatically Uses URL Context Tool**

When the agent sees paper URLs, it **automatically decides** to extract deeper content:

```python
# Agent's internal reasoning:
"I found these papers, but I only have abstracts. 
To provide a comprehensive answer, I should extract 
full content from the most relevant papers."

Agent calls:
extract_url_content(
    url="https://arxiv.org/abs/1706.03762",
    query="What are the key innovations in this paper? 
           What methodology did they use?"
)
```

### **Step 4: DeepSeek Processes Content**

```python
Tool workflow:
1. Scrapes Arxiv page → Gets HTML
2. Extracts text with BeautifulSoup
3. Sends to DeepSeek with query:
   "What are the key innovations? What methodology?"
4. DeepSeek processes and answers:
   "Key innovations: Multi-head attention mechanism,
    positional encoding, encoder-decoder architecture..."
```

### **Step 5: Agent Uses Extracted Content**

```python
Agent now has:
- Abstract (from search tool)
- Full content understanding (from URL context tool)
- DeepSeek's analysis

Agent synthesizes:
"Based on the papers found and their full content:
1. Attention Is All You Need introduced multi-head attention...
2. BERT used bidirectional transformers for pre-training..."
```

### **Step 6: Content Added to RAG**

```python
# Automatically happens in base_agent.py
if new_papers:
    self.add_papers_to_rag(new_papers)
    # Full content (not just abstract) is now in vector store
```

---

## 🎯 **Real Agent Decision Making**

### **Scenario: Literature Review Query**

```
User: "Review recent LLM evaluation methods"

Agent Workflow:

1. SEARCH PHASE
   arxiv_search("LLM evaluation methods")
   → Finds 5 papers with URLs

2. DEEP DIVE PHASE (Agent decides automatically)
   For each relevant paper URL:
   
   extract_url_content(
       url="https://arxiv.org/abs/2301.00001",
       query="What evaluation metrics are used? 
              What are the limitations?"
   )
   
   → DeepSeek extracts: "Uses BLEU, ROUGE, human evaluation.
                         Limitations: Limited to English, 
                         small test set..."

3. SYNTHESIS PHASE
   Agent combines:
   - Abstracts from search
   - Deep insights from URL extraction
   - Cross-paper comparisons
   
   → Comprehensive literature review
```

---

## 🛠️ **Technical Implementation**

### **Tool Availability**

Every agent automatically has access to URL context tool:

```python
# In base_agent.py __init__
self._toolkit = ResearchToolkit()
self._tools = self._toolkit.get_tools_for_field(self.FIELD)

# ResearchToolkit includes:
- arxiv_search
- semantic_scholar_search  
- extract_url_content  ← URL context tool
- read_pdf
- web_search
```

### **Agent Execution Flow**

```python
# In base_agent.py research() method

async def research(self, query):
    # 1. Get context from RAG
    context, papers, confidence = self._rag.get_context_for_query(query)
    
    # 2. Build enhanced input
    enhanced_input = f"""
    Query: {query}
    Context from previous research: {context}
    
    Use your tools to find papers and extract deep content.
    """
    
    # 3. Agent executes (can use any tool, including URL context)
    result = await self._agent_executor.ainvoke({
        "input": enhanced_input,
        "chat_history": chat_history
    })
    
    # 4. Extract papers (including from URL extractions)
    papers = self._extract_papers_from_result(result)
    
    # 5. Add to RAG automatically
    self.add_papers_to_rag(papers)
```

---

## 💡 **When Agents Use URL Context**

Agents **automatically decide** to use URL context when:

1. **They find paper URLs** from search tools
   - "I found this paper URL, let me extract full content"

2. **They need deeper understanding**
   - "The abstract isn't enough, I need methodology details"

3. **User asks specific questions**
   - "What methodology did they use?" → Extract and ask DeepSeek

4. **Cross-referencing needed**
   - "How does this compare to that paper?" → Extract both

5. **Building comprehensive synthesis**
   - "I need full context to write a thorough review"

---

## 📊 **Example: Complete Agent Interaction**

```
USER: "What are the latest transformer architectures?"

AGENT (AI/ML):
┌─────────────────────────────────────────┐
│ 1. Searching Arxiv...                   │
│    → Found 5 papers                     │
│                                          │
│ 2. Extracting content from papers...    │
│    → extract_url_content(paper1_url)    │
│    → DeepSeek: "Key innovation: ..."    │
│    → extract_url_content(paper2_url)    │
│    → DeepSeek: "Uses novel attention"   │
│                                          │
│ 3. Synthesizing findings...             │
│    → Combines abstracts + full content  │
│                                          │
│ 4. Storing in RAG...                    │
│    → Full content saved for future      │
└─────────────────────────────────────────┘

RESPONSE:
"Based on recent papers and their full content:

1. **Paper X** (extracted full content):
   - Innovation: Multi-scale attention
   - Methodology: Pre-training on 1T tokens
   - Results: 15% improvement on benchmarks

2. **Paper Y** (extracted full content):
   - Innovation: Sparse attention patterns
   - Methodology: Dynamic routing
   ..."
```

---

## 🔧 **Agent Prompts Guide Usage**

Agents are instructed in their system prompts:

```python
# Example from AI/ML agent
def _get_system_prompt(self):
    return """
    You are an AI/ML research specialist.
    
    When researching:
    1. Search for papers using arxiv_search, semantic_scholar_search
    2. For important papers, use extract_url_content to get full details
    3. Use read_pdf for PDF papers
    4. Synthesize findings comprehensively
    
    Always extract full content when you need:
    - Methodology details
    - Specific results
    - Implementation details
    - Comparisons between papers
    """
```

---

## 🎯 **Key Points**

1. **Automatic**: Agents decide when to use URL context (no manual trigger)
2. **Intelligent**: DeepSeek processes content, not just scraping
3. **Integrated**: Works seamlessly with other tools
4. **Persistent**: Extracted content goes into RAG for future use
5. **Query-Aware**: Agents can ask specific questions about URLs

---

## 📈 **Impact**

**Before URL Context:**
- Agents: "I found 5 papers with abstracts"
- User: "What methodology did they use?"
- Agent: "I only have abstracts, can't tell"

**After URL Context:**
- Agents: "I found 5 papers, extracting full content..."
- User: "What methodology did they use?"
- Agent: "Based on full content: They used X methodology with Y approach..."

**Result: Deeper, more accurate research!**

