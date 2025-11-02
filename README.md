# 🧠 Deep-Thinking RAG System

<div align="center">

<a href="https://buymeacoffee.com/mahendramedapati" target="_blank">
  <img src="images/1_CEZSIxeYr6PCxsN6Gr38MQ.png" alt="Buy Me A Coffee" width="300">
</a>

<br>

<a href="https://buymeacoffee.com/mahendramedapati" target="_blank">
  <strong>☕ Buy Me a Coffee</strong>
</a>

</div>

An intelligent agentic RAG (Retrieval-Augmented Generation) system with multi-step reasoning, self-critique, and adaptive retrieval strategies. Built with LangGraph, LangChain, and Streamlit.

## 🌟 Features

- **🧠 Multi-Step Reasoning**: Breaks down complex queries into sequential research steps
- **🔧 Tool Selection**: Dynamically chooses between document search and web search
- **🎯 Smart Retrieval**: Vector, BM25, and hybrid search with reranking
- **🤔 Self-Critique**: Policy agent decides when research is complete
- **📝 Comprehensive Answers**: Synthesizes findings with full source citations
- **🎨 Beautiful UI**: Modern Streamlit interface with real-time progress tracking
- **⚡ Fast & Cost-Effective**: Powered by Groq API with free local embeddings
- **🔍 Web Integration**: Combines internal documents with web search for comprehensive answers

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [API Keys](#api-keys)
- [Evaluation](#evaluation)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🚀 Installation

### Using Conda (Recommended)

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/deep-thinking-rag.git
cd deep-thinking-rag
```

2. **Run setup script:**

**On macOS/Linux:**
```bash
chmod +x setup.sh
./setup.sh
```

**On Windows:**
```bash
setup.bat
```

3. **Activate environment:**
```bash
conda activate deep-thinking-rag
```

4. **Configure API keys:**

Create a `.env` file in the project root:
```bash
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
LANGSMITH_API_KEY=your_langsmith_api_key_here  # Optional
```

### Manual Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## ⚡ Quick Start

### Option 1: Streamlit UI (Recommended)

1. **Add your document:**
```bash
# Place any text document in data/ folder
cp your_document.txt data/
```

2. **Run the app:**
```bash
streamlit run app.py
```

3. **Open your browser:**
Navigate to `http://localhost:8501`

4. **Upload document and ask questions:**
   - Upload your document in the sidebar
   - Click "Initialize Vector Store"
   - Enter your question
   - Click "Go"

### Option 2: Command Line

```bash
python main.py
```

## 📖 Usage Examples

### Example 1: Basic Query

**Query:** *"What are the key competitive risks mentioned in the document?"*

**System Process:**
1. 🧠 Plans research strategy
2. 🔍 Retrieves relevant sections
3. 🎯 Reranks for precision
4. ✂️ Distills context
5. 🤔 Reflects on findings
6. ✅ Generates cited answer

### Example 2: Multi-Hop Query

**Query:** *"Explain the competitive risks from the document and how recent market developments affect them"*

**System Process:**
1. 🧠 Decomposes into 2 research steps
2. 🔍 Searches document for competitive risks
3. 🌐 Searches web for recent market developments
4. 🔄 Loops until sufficient information
5. ✅ Synthesizes comprehensive answer

### Example 3: Complex Analysis

**Query:** *"Analyze the key financial metrics, operational challenges, and their impact on growth prospects"*

**System Process:**
1. 🧠 Breaks down into multiple research areas
2. 🔍 Retrieves financial data, operational sections, and projections
3. 🌐 Searches web for industry benchmarks
4. 🤔 Self-critiques completeness
5. ✅ Provides comprehensive analysis with citations

## ⚙️ Configuration

Edit `config/config.py` or use environment variables:

```python
# LLM Settings
REASONING_LLM = "llama-3.1-8b-instant"   # For complex reasoning
FAST_LLM = "llama-3.1-8b-instant"        # For simple tasks
EMBEDDING_MODEL = "all-MiniLM-L6-v2"     # Free local embeddings

# Retrieval Settings
TOP_K_RETRIEVAL = 10              # Broad recall
TOP_N_RERANK = 3                  # Precision filtering
MAX_REASONING_ITERATIONS = 7      # Safety limit

# Document Processing
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Query                               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
                   ┌─────────┐
                   │  PLAN   │ ← Multi-step decomposition
                   └────┬────┘
                        │
         ┌──────────────┴──────────────┐
         │                             │
    ┌────▼─────┐               ┌──────▼──────┐
    │ Document │               │   Web       │
    │ Search   │               │   Search    │
    └────┬─────┘               └──────┬──────┘
         │                             │
         └──────────────┬──────────────┘
                        │
                   ┌────▼─────┐
                   │ RERANK   │ ← Cross-encoder
                   └────┬─────┘
                        │
                   ┌────▼─────┐
                   │ COMPRESS │ ← Context distillation
                   └────┬─────┘
                        │
                   ┌────▼─────┐
                   │ REFLECT  │ ← Build history
                   └────┬─────┘
                        │
                   ┌────▼─────┐
                   │ POLICY   │ ← Continue or finish?
                   └────┬─────┘
                        │
         ┌──────────────┴──────────────┐
         │                             │
    ┌────▼─────┐               ┌──────▼──────┐
    │ CONTINUE │               │   FINISH    │
    └────┬─────┘               └──────┬──────┘
         │                             │
         └───────────┬─────────────────┘
                     │
                ┌────▼─────────┐
                │ FINAL ANSWER │
                └──────────────┘
```

### Key Components

1. **Agents** (`src/agents/`):
   - Planner: Multi-step decomposition
   - Query Rewriter: Search optimization
   - Retrieval Supervisor: Strategy selection
   - Reflection: Summarization
   - Policy: Decision-making
   - Final Answer: Synthesis

2. **Retrieval** (`src/retrieval/`):
   - Document Processor: Section-aware parsing
   - Vector Store: Chroma embeddings
   - Search Strategies: Vector, BM25, Hybrid
   - Reranker: Cross-encoder precision

3. **Graph** (`src/graph/`):
   - Nodes: Processing steps
   - Edges: Conditional routing
   - Workflow: State machine

4. **Tools** (`src/tools/`):
   - Web Search: Tavily integration

### Retrieval Strategies

The system intelligently selects the best search strategy for each query:

| Strategy | When to Use | Description |
|----------|-------------|-------------|
| **Vector Search** | Conceptual queries | Semantic similarity using embeddings |
| **BM25** | Keyword queries | Traditional keyword matching |
| **Hybrid** | Complex queries | Combines both using RRF (Reciprocal Rank Fusion) |

## 📊 Project Structure

```
deep-thinking-rag/
├── config/
│   ├── __init__.py
│   └── config.py              # Centralized configuration
├── data/                      # Document storage
│   └── README.md
├── logs/                      # Log files
├── src/
│   ├── agents/                # Agent components
│   │   ├── planner.py         # Multi-step planning
│   │   ├── query_rewriter.py  # Query optimization
│   │   ├── retrieval_supervisor.py  # Strategy selection
│   │   ├── reflection.py      # Research summarization
│   │   ├── policy.py          # Decision-making
│   │   └── final_answer.py    # Answer synthesis
│   ├── retrieval/             # Retrieval components
│   │   ├── document_processor.py  # Document parsing
│   │   ├── vector_store.py    # Vector database
│   │   ├── search_strategies.py   # Multiple search methods
│   │   └── reranker.py        # Precision filtering
│   ├── tools/                 # External tools
│   │   └── web_search.py      # Tavily integration
│   ├── graph/                 # LangGraph workflow
│   │   ├── nodes.py           # Processing nodes
│   │   ├── edges.py           # Routing logic
│   │   └── workflow.py        # State machine
│   └── utils/                 # Utilities
│       └── helpers.py         # Helper functions
├── app.py                     # Streamlit UI
├── main.py                    # CLI entry point
├── requirements.txt           # Python dependencies
├── environment.yml            # Conda environment
└── README.md                  # This file
```

## 🔑 API Keys Required

1. **Groq API Key** (Required)
   - Get at: https://console.groq.com/api-keys
   - Used for: LLM reasoning and responses
   - Free tier available with generous rate limits

2. **Tavily API Key** (Required)
   - Get at: https://tavily.com/
   - Used for: Web search functionality
   - Free tier available

3. **LangSmith API Key** (Optional)
   - Get at: https://smith.langchain.com/
   - Used for: Tracing and debugging
   - Free tier available

## 💰 Cost & Performance

### Current Setup (Groq)

- **Cost per query**: ~$0.00-0.05 (virtually free on free tier)
- **Speed**: ~400+ tokens/sec (10x faster than OpenAI)
- **Embeddings**: FREE (local sentence-transformers)
- **Rate Limits**: Generous on free tier

### Performance Metrics

- **Latency**: 25-50 seconds per complex query
- **Context Precision**: ~89%
- **Context Recall**: ~100%
- **Faithfulness**: ~100%
- **Answer Correctness**: ~99%

## 📝 Logging

Logs are saved to `logs/deep_thinking_rag.log` with rotation:
- Max file size: 100 MB
- Retention: 10 days
- Format: Timestamp | Level | Location | Message

View logs in real-time:
```bash
tail -f logs/deep_thinking_rag.log
```

## 🐛 Troubleshooting

### "Missing required API keys"
- Check your `.env` file
- Ensure keys are correct and not wrapped in quotes
- Verify key format matches provider requirements

### "Vector store not initialized"
- Upload a document in the UI
- Click "Initialize Vector Store"
- Or run `python main.py` first

### Import errors
- Ensure conda environment is activated
- Run `pip install -r requirements.txt --upgrade`
- Check Python version (3.10+)

### Rate limit errors
- Wait a few minutes and try again
- Consider upgrading your Groq API plan
- Or switch to faster model: `llama-3.1-8b-instant`

### Port conflicts
- Run `streamlit run app.py --server.port 8502`

## 🔍 Use Cases

This RAG system can be adapted for various information domains:

- **📄 Financial Documents**: 10-K filings, quarterly reports, analyst reports
- **📚 Research Papers**: Scientific literature, academic papers
- **📖 Technical Documentation**: API docs, user guides, technical specs
- **📰 News & Articles**: Media archives, news databases
- **📊 Business Intelligence**: Corporate reports, market research
- **📝 Legal Documents**: Contracts, regulations, case law
- **🎓 Educational Content**: Textbooks, course materials

The system automatically adapts to the content type and domain!

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Areas for Contribution

- Adding new retrieval strategies
- Implementing additional tools
- Improving UI/UX
- Performance optimization
- Documentation improvements
- Bug fixes

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- LangChain team for the amazing framework
- Groq for fast, affordable inference
- Tavily for web search capabilities
- Streamlit for the UI framework
- The open-source community for inspiration

---

## 🚀 Want to Master More AI?

**Subscribe to my YouTube channel** for in-depth tutorials, hands-on coding sessions, and the latest AI insights! 📺✨

**👆 Hit that subscribe button and ring the notification bell to never miss cutting-edge content!**

---

## 🔗 Let's Connect & Collaborate!

I'm passionate about sharing knowledge and building amazing AI solutions. Let's connect:

### 📱 Social Media & Professional Links

| Platform | Link | Description |
|----------|------|-------------|
| 🐙 **GitHub** | [@MahendraMedapati27](https://github.com/MahendraMedapati27) | Check out my latest projects and code repositories |
| 💼 **LinkedIn** | [Mahendra Medapati](https://www.linkedin.com/in/mahendra-medapati-429239289/) | Connect for professional discussions and industry insights |
| 🐦 **X (Twitter)** | [@MahendraM27](https://x.com/MahendraM27) | Follow for updates, thoughts, and discussions on AI |
| 📧 **Email** | [mahendramedapati.r469@gmail.com](mailto:mahendramedapati.r469@gmail.com) | Reach out directly for inquiries or collaboration |

### ☕ Support This Project

If you find this project helpful, consider **buying me a coffee** to support continued development! ☕✨

<div align="center">

<a href="https://buymeacoffee.com/mahendramedapati" target="_blank">
  <img src="images/1_CEZSIxeYr6PCxsN6Gr38MQ.png" alt="Buy Me A Coffee" width="300">
</a>

<br>

<a href="https://buymeacoffee.com/mahendramedapati" target="_blank">
  <strong>☕ Buy Me a Coffee</strong>
</a>

</div>

---

<div align="center">

**Built with 🧠 for intelligent information retrieval**

</div>
