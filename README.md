# RAG Architectures PoC

A Proof of Concept implementing 8 RAG (Retrieval-Augmented Generation) architectures using Python, LangChain, LangGraph, and Docling, with observability (Langfuse), semantic caching (Redis), and LLM-as-judge evaluation (DeepEval).

## Features

- **8 RAG architectures** — Simple, Conversational, Standard, Corrective, Fusion, Contextual, Agentic, Graph
- **Langfuse tracing** — full observability via custom REST callback
- **Semantic cache** — Redis-backed, cosine similarity ≥ 0.90 skips the LLM
- **ChromaDB viewer** — Streamlit-based web UI at port 8501
- **LLM-as-judge evaluation** — DeepEval with Faithfulness / Answer Relevancy / Contextual Precision metrics
- **Multi-modal ingestion** — PDF, DOCX, audio/video (Whisper), database content

## Prerequisites

- Python >= 3.9
- [Task](https://taskfile.dev/) (task runner)
- Docker (for Chroma, Redis, Langfuse, PostgreSQL)
- **Either:**
  - OpenAI API key, OR
  - [Ollama](https://ollama.com/) installed locally

## Switching LLM Providers

Edit `.env` to switch between OpenAI and Ollama:

```bash
# Use OpenAI
LLM_PROVIDER=openai
LLM_MODEL=gpt-3.5-turbo
OPENAI_API_KEY=your_key

# Use Ollama (local)
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434
```

Start Ollama:
```bash
ollama serve
ollama pull llama3.2
```

## Dependencies

Core dependencies (from `pyproject.toml`):
- langchain >= 0.3.0, langgraph >= 0.2.0, langchain-openai, langchain-community
- chromadb >= 0.5.0, redis >= 5.0.0
- deepeval >= 3.0.0 (LLM-as-judge evaluation)
- docling >= 2.0.0, openai-whisper (multi-modal ingestion)
- streamlit >= 1.40.0 (ChromaDB viewer)
- psycopg2-binary (database ingestion)
- rich, python-dotenv, sqlalchemy

## Setup

### 1. Clone and Configure

```bash
cp .env .env.local  # if needed
```

Edit `.env` with your API keys and settings.

### 2. Create Virtual Environment

```bash
task venv
```

Creates `.venv` and installs all dependencies.

### 3. Start External Dependencies

```bash
docker compose up -d
```

Starts the full stack:
| Service | Port | Purpose |
|---------|------|---------|
| Chroma | 8000 | Vector database |
| Redis | 6379 | Semantic cache |
| Langfuse | 4000 | LLM observability |
| PostgreSQL | 5432 | Metadata & Langfuse storage |

### 4. Full Pipeline (DB + Ingestion)

```bash
task feed-all
```

Initializes the database (100 users, 500 orders) and ingests documents from `data/` into Chroma.

If starting from scratch with a clean Docker state:
```bash
task clean-start
```

## Running RAG Architectures

### CLI

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

### Streamlit Test Apps

```bash
task streamlit APP=simple_test
task streamlit APP=conversational_test
```

Test apps live in `src/test_apps/`.

## ChromaDB Web Viewer

```bash
task chroma-viewer
```

Opens a Streamlit UI at http://localhost:8501 to browse collections and documents in Chroma.

## Semantic Caching

All architectures include a Redis-backed semantic cache. On each query:
1. The query is embedded and compared against cached entries (cosine similarity)
2. If similarity ≥ 0.90, the cached response is returned instantly (no LLM call)
3. Cache HIT/MISS is recorded in the response metadata as `"cached": true/false`

Clear the cache at any time:
```python
from utils.semantic_cache import SemanticCache
from utils.config import get_embeddings
SemanticCache(embedding_func=get_embeddings()).clear()
```

## Observability (Langfuse)

All LLM calls are traced to Langfuse (http://localhost:4000). Login with `admin@local.dev` / `admin123`.

The `LangfuseRestCallback` in `src/utils/langfuse_tracing.py` sends traces via the Langfuse REST ingestion API — no SDK dependency, compatible with langchain ≥ 1.0.

## Evaluation (DeepEval)

Run LLM-as-judge evaluation across all architectures:

```bash
# Evaluate all architectures against all test cases
task eval

# Evaluate specific architectures
task eval-arch -- simple standard

# View the latest report
task eval-report
```

Reports are saved as JSONL to `.logs/reports/eval.{timestamp}.log`.

Default metrics:
- **Faithfulness** — Is the answer grounded in the retrieved context?
- **Answer Relevancy** — Is the answer relevant to the query?
- **Contextual Precision** — Are relevant documents ranked higher?

The judge LLM is configured via `.env` (`LLM_PROVIDER` / `OLLAMA_MODEL` / `LLM_MODEL`).

## Project Structure

```
├── src/
│   ├── db/                # Database initialization
│   ├── eval/              # DeepEval evaluation (dataset, runner, report)
│   ├── ingestion/         # Document ingestion with Docling + Whisper
│   ├── rag_architectures/ # 8 RAG architectures
│   │   ├── simple/
│   │   ├── conversational/
│   │   ├── standard/
│   │   ├── corrective/
│   │   ├── fusion/
│   │   ├── contextual/
│   │   ├── agentic/       # LangGraph workflow
│   │   └── graph/
│   ├── test_apps/         # Streamlit test apps
│   ├── tools/             # ChromaDB viewer
│   └── utils/             # Config, tracing, semantic cache, logging
├── data/                  # Documents for ingestion
├── .logs/                 # Application logs + eval reports
├── .env                   # Configuration
├── docker-compose.yml     # External services
├── pyproject.toml         # Python dependencies
├── Taskfile.yml           # Task runner commands
└── DIAGRAMS.md            # Mermaid diagrams
```

## All Tasks

| Task | Description |
|------|-------------|
| `task venv` | Create virtual environment |
| `task install` | Install dependencies |
| `task db` | Initialize PostgreSQL with sample data |
| `task ingest` | Ingest documents + DB data to Chroma server |
| `task feed-all` | DB init + ingestion (one-shot) |
| `task clean-start` | Full reset: Docker down -v, rm .local, up, feed |
| `task run ARCH=...` | Run a specific RAG architecture |
| `task chroma-viewer` | Start ChromaDB web viewer (port 8501) |
| `task streamlit APP=...` | Run a Streamlit test app |
| `task eval` | DeepEval evaluation of all architectures |
| `task eval-arch -- ...` | Evaluate specific architectures |
| `task eval-report` | Show latest evaluation report |
