# Local AI Research Engine

A local, offline-first research assistant that ingests documents, builds knowledge graphs, and answers complex questions with citations using Ollama.

## Features

- **Document Intelligence**: Ingest PDFs, markdown, code, and text files
- **Knowledge Graph**: Automatically extract entities and relationships
- **Hybrid Retrieval**: Vector search + keyword search + graph traversal
- **Cited Answers**: Every answer includes source citations
- **Multi-Document Reasoning**: Synthesize information across multiple sources
- **100% Local**: Runs entirely on Ollama - no API calls, complete privacy

## Prerequisites

1. **Python 3.9+**
2. **Ollama** installed and running
3. **Required Models**:
   ```bash
   ollama pull qwen2.5:7b
   ollama pull nomic-embed-text
   ```

## Installation

1. Clone or download this repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and configure if needed

## Quick Start

1. **Add documents** to `data/documents/`
2. **Run the Streamlit UI**:
   ```bash
   streamlit run ui/streamlit_app.py
   ```
3. **Or use the CLI**:
   ```bash
   python main.py
   ```

## Project Structure

```
local-research-engine/
├── ingest/          # Document loading and chunking
├── index/           # Vector store, keyword index, knowledge graph
├── retrieval/       # Hybrid search and reranking
├── llm/             # Ollama client and prompts
├── ui/              # Streamlit interface
├── data/            # Documents and indexes
└── main.py          # CLI interface
```

## Usage

### Upload Documents

Place your documents in `data/documents/` or use the Streamlit upload interface.

### Ask Questions

The system will:
1. Retrieve relevant chunks using hybrid search
2. Expand context using the knowledge graph
3. Rerank evidence with LLM
4. Generate an answer with inline citations

### Example Query

**Q**: "What is the EM algorithm used for?"

**A**: The EM algorithm is used to estimate HMM parameters by iteratively maximizing the expected log-likelihood [Rabiner1989.pdf §4]. Unlike gradient-based methods, EM guarantees non-decreasing likelihood [Bishop.pdf §9.2].

## Configuration

Edit `config.yaml` to customize:
- Chunk sizes
- Retrieval parameters
- Model selection
- Storage paths

## License

MIT License - See LICENSE file for details
