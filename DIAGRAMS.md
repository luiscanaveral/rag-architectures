# RAG Architecture Diagrams

## 1. Simple (Naive) RAG

```mermaid
graph LR
    A[User Query] --> B{Semantic Cache}
    B -->|HIT| C[Return Cached]
    B -->|MISS| D[Vector Search]
    E[Chroma Server] --> D
    D --> F[Retrieved Context]
    F --> G[LLM]
    G --> H[Generated Answer]
    H --> I[Store in Cache]
```

## 2. Conversational RAG

```mermaid
graph LR
    A[User Query] --> B{Semantic Cache}
    B -->|HIT| C[Return Cached]
    B -->|MISS| D[Chat History]
    D --> E[Context + Query]
    E --> F[Vector Search]
    G[Chroma Server] --> F
    F --> H[Retrieved Context]
    H --> I[LLM with History]
    I --> J[Answer]
    J --> K[Store in Cache]
    J --> D
```

## 3. Standard RAG

```mermaid
graph TD
    A[User Query] --> B{Semantic Cache}
    B -->|HIT| C[Return Cached]
    B -->|MISS| D[Embed Query]
    D --> E[Similarity Search]
    F[Chroma Server] --> E
    E --> G[Top K Documents]
    G --> H[Context Assembly]
    H --> I[LLM Prompt]
    I --> J[Answer with Sources]
    J --> K[Store in Cache]
```

## 4. Corrective RAG

```mermaid
graph TD
    A[Query] --> B{Semantic Cache}
    B -->|HIT| C[Return Cached]
    B -->|MISS| D[Retrieve Documents]
    D --> E{Validate Docs}
    E -->|Relevant| F[Generate Answer]
    E -->|Not Relevant| G[Web Search]
    G --> F
    E -->|Uncertain| H[Refine Query]
    H --> D
    F --> I[Answer]
    I --> J[Store in Cache]
```

## 5. Fusion RAG

```mermaid
graph TD
    A[Query] --> B{Semantic Cache}
    B -->|HIT| C[Return Cached]
    B -->|MISS| D[Vector Search]
    A --> E[BM25 Keyword Search]
    D --> F[Ensemble Retriever]
    E --> F
    F --> G[Fused Results]
    G --> H[LLM]
    H --> I[Answer]
    I --> J[Store in Cache]
```

## 6. Contextual RAG

```mermaid
graph TD
    A[Query] --> B{Semantic Cache}
    B -->|HIT| C[Return Cached]
    B -->|MISS| D[Retrieve Context]
    D --> E[Context Window]
    F[Document Metadata] --> E
    E --> G[Context-Aware Prompt]
    G --> H[LLM]
    H --> I[Contextual Answer]
    I --> J[Store in Cache]
```

## 7. Agentic RAG

```mermaid
graph TD
    A[User Query] --> B{Semantic Cache}
    B -->|HIT| C[Return Cached]
    B -->|MISS| D[Agent Planner]
    D --> E{Decision}
    E -->|Vector Search| F[Retrieve Docs]
    E -->|Web Search| G[Search API]
    E -->|SQL Query| H[Query Database]
    F --> I[Synthesize]
    G --> I
    H --> I
    I --> J{Need More?}
    J -->|Yes| D
    J -->|No| K[Final Answer]
    K --> L[Store in Cache]
```

## 8. Graph RAG

```mermaid
graph TD
    A[Query] --> B{Semantic Cache}
    B -->|HIT| C[Return Cached]
    B -->|MISS| D[Extract Entities]
    D --> E[Graph Traversal]
    F[Knowledge Graph] --> E
    E --> G[Relationship Paths]
    G --> H[Path Ranking]
    H --> I[LLM with Graph Context]
    I --> J[Explainable Answer]
    J --> K[Store in Cache]
```

## 9. Semantic Cache — Cross-Cutting

```mermaid
graph TD
    A[Query] --> B[Embed Query]
    B --> C[Redis: cosine similarity]
    C --> D{Score >= 0.90?}
    D -->|Yes| E[Return Cached Response]
    D -->|No| F[Run RAG Pipeline]
    F --> G[Store Embedding + Response]
    G --> H[Return Fresh Response]
```

## 10. Evaluation Pipeline (DeepEval)

```mermaid
graph LR
    A[Test Dataset] --> B[Run RAG Pipeline]
    C[Chroma: retrieve context] --> B
    B --> D[actual_output]
    D --> E[DeepEval Metrics]
    F[Judge LLM] --> E
    E --> G[Faithfulness]
    E --> H[Answer Relevancy]
    E --> I[Contextual Precision]
    G --> J[Report]
    H --> J
    I --> J
```

## Infrastructure

```mermaid
graph TD
    A[User] --> B[Streamlit Viewer :8501]
    A --> C[RAG Apps :CLI]
    A --> D[Langfuse UI :4000]
    C --> E[Chroma Server :8000]
    C --> F[Redis Cache :6379]
    C --> G[LLM: Ollama/OpenAI]
    E --> H[Chroma SQLite]
    F --> I[Semantic Embeddings]
    D --> J[PostgreSQL :5432]
```

## Comparison Summary

| Architecture | Complexity | Use Case |
|-------------|------------|----------|
| Simple | Low | Basic Q&A |
| Conversational | Low | Chatbots with memory |
| Standard | Medium | Production Q&A with citations |
| Corrective | Medium | High accuracy requirements |
| Fusion | Medium | Hybrid search needs |
| Contextual | Medium | Documents with rich metadata |
| Agentic | High | Complex multi-step queries |
| Graph | High | Relationship reasoning |
