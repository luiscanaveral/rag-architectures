# RAG Architectures PoC

A Proof of Concept implementing 8 RAG (Retrieval-Augmented Generation) architectures using Python, LangChain, LangGraph, and Docling.

## Architectures Implemented

- **Simple (Naive) RAG** - Basic retrieval + generation
- **Conversational RAG** - With chat history
- **Standard RAG** - With source document tracking
- **Corrective RAG** - Validates retrieved documents
- **Fusion RAG** - Vector + BM25 ensemble
- **Contextual RAG** - Broader context awareness
- **Agentic RAG** - LangGraph multi-step workflow
- **Graph RAG** - Relationship-based retrieval

## Prerequisites

- Python >= 3.9
- [Task](https://taskfile.dev/) (task runner)
- Docker (for external dependencies)
- OpenAI API key
- LangSmith API key (optional, for tracing)

## Dependencies

Core dependencies (from `pyproject.toml`):
- langchain >= 0.3.0
- langchain-openai >= 0.2.0
- langchain-community >= 0.3.0
- langgraph >= 0.2.0
- langsmith >= 0.1.0
- rich >= 13.0.0
- docling >= 2.0.0
- streamlit >= 1.40.0
- sqlalchemy >= 2.0.0
- python-dotenv >= 1.0.0
- chromadb >= 0.5.0

## Setup

### 1. Clone and Configure

```bash
# Copy and edit .env with your API keys
cp .env .env.local  # if needed
```

Edit `.env`:
```
LANGCHAIN_API_KEY=your_langsmith_key
OPENAI_API_KEY=your_openai_key
```

### 2. Create Virtual Environment

```bash
task venv
```

This creates `.venv` and installs all dependencies.

### 3. Start External Dependencies

```bash
task docker
```

Starts ChromaDB (port 8000) and Redis (port 6379).

### 4. Initialize Database

```bash
task db
```

Creates SQLite database with 100 users and 500 orders.

### 5. Add Documents

Place your documents (PDF, DOCX, etc.) in the `data/` folder.

### 6. Ingest Documents

```bash
task ingest
```

Processes documents using Docling and creates vector embeddings.

## Running RAG Architectures

### Run a Specific Architecture

```bash
task run ARCH=simple
task run ARCH=conversational
task run ARCH=standard
task run ARCH=corrective
task run ARCH=fusion
task run ARCH=contextual
task run ARCH=agentic
task run ARCH=graph
```

### Run Streamlit Tests

```bash
task streamlit APP=simple_test
task streamlit APP=conversational_test
```

## Project Structure

```
├── src/
│   ├── db/              # Database initialization
│   ├── ingestion/       # Document ingestion with Docling
│   ├── rag_architectures/
│   │   ├── simple/      # Naive RAG
│   │   ├── conversational/
│   │   ├── standard/
│   │   ├── corrective/
│   │   ├── fusion/
│   │   ├── contextual/
│   │   ├── agentic/     # LangGraph workflow
│   │   └── graph/      # Graph-based RAG
│   └── utils/           # Shared utilities
├── tests/               # Streamlit test apps
├── data/                # Documents for ingestion
├── .env                 # Configuration
├── docker-compose.yml   # External dependencies
├── pyproject.toml       # Python dependencies
├── Taskfile.yml         # Task runner commands
└── DIAGRAMS.md         # Mermaid diagrams
```

## Clean Up

```bash
task clean
```

Removes virtual environment, vectorstore, and cache files.
