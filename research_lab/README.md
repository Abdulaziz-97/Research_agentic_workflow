# 🔬 Research Lab - Multi-Agent Research Workflow

An AI-powered multi-agent research system that brings together specialized AI scientists to help explore complex research questions across multiple scientific domains.

## Features

- **Multi-Agent Architecture**: 8 domain-specialized agents + 5 support agents
- **RAG-Powered Research**: Retrieve-Reflect-Retry pattern for reliable information
- **Academic Database Integration**: Arxiv, Semantic Scholar, PubMed search
- **Memory Systems**: Short-term and long-term memory with ChromaDB
- **LangGraph Workflow**: Parallel agent execution with state management
- **Streamlit UI**: Modern, responsive interface

## Available Agents

### Domain Agents (Select up to 3)
- 🤖 **AI/ML Agent**
- ⚛️ **Physics Agent**
- 🧬 **Biology Agent**
- ⚗️ **Chemistry Agent**
- 📐 **Mathematics Agent**
- 🧠 **Neuroscience Agent**
- 💊 **Medicine Agent**
- 💻 **Computer Science Agent**

### Support Agents (Always Available)
- 📚 **Literature Reviewer**
- 🔍 **Methodology Critic**
- ✓ **Fact Checker**
- ✍️ **Writing Assistant**
- 🔗 **Cross-Domain Synthesizer**

## Installation

```bash
cd research_lab
pip install -r requirements.txt

# Create .env file and set your preferred LLM provider (Gemini or OpenAI)
# Example for Gemini:
# LLM_PROVIDER=gemini
# GEMINI_API_KEY=your_key_here

streamlit run app.py
```

## Configuration

Create a `.env` file based on `env.example.txt`. Example for Gemini:

```
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.0-pro
```

For OpenAI/Vocareum:

```
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_BASE_URL= # optional custom endpoint
```

## License

MIT License

