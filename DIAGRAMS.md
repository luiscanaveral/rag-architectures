# RAG Architecture Diagrams

## 1. Simple (Naive) RAG

```mermaid
graph LR
    A[User Query] --> B[Vector Search]
    C[Document Store] --> B
    B --> D[Retrieved Context]
    D --> E[LLM]
    E --> F[Generated Answer]
```

## 2. Conversational RAG

```mermaid
graph LR
    A[User Query] --> B[Chat History]
    B --> C[Context + Query]
    C --> D[Vector Search]
    E[Document Store] --> D
    D --> F[Retrieved Context]
    F --> G[LLM with History]
    G --> H[Answer]
    H --> B
```

## 3. Standard RAG

```mermaid
graph TD
    A[User Query] --> B[Embed Query]
    B --> C[Similarity Search]
    D[Vector Store] --> C
    C --> E[Top K Documents]
    E --> F[Context Assembly]
    F --> G[LLM Prompt]
    G --> H[Answer with Sources]
```

## 4. Corrective RAG

```mermaid
graph TD
    A[Query] --> B[Retrieve Documents]
    B --> C{Validate Docs}
    C -->|Relevant| D[Generate Answer]
    C -->|Not Relevant| E[Web Search]
    E --> D
    C -->|Uncertain| F[Refine Query]
    F --> B
    D --> G[Answer]
```

## 5. Fusion RAG

```mermaid
graph TD
    A[Query] --> B[Vector Search]
    A --> C[BM25 Keyword Search]
    B --> D[Ensemble Retriever]
    C --> D
    D --> E[Fused Results]
    E --> F[LLM]
    F --> G[Answer]
```

## 6. Contextual RAG

```mermaid
graph TD
    A[Query] --> B[Retrieve Context]
    B --> C[Context Window]
    D[Document Metadata] --> C
    C --> E[Context-Aware Prompt]
    E --> F[LLM]
    F --> G[Contextual Answer]
```

## 7. Agentic RAG

```mermaid
graph TD
    A[User Query] --> B[Agent Planner]
    B --> C{Decision}
    C -->|Vector Search| D[Retrieve Docs]
    C -->|Web Search| E[Search API]
    C -->|SQL Query| F[Query Database]
    D --> G[Synthesize]
    E --> G
    F --> G
    G --> H{Need More?}
    H -->|Yes| B
    H -->|No| I[Final Answer]
```

## 8. Graph RAG

```mermaid
graph TD
    A[Query] --> B[Extract Entities]
    B --> C[Graph Traversal]
    D[Knowledge Graph] --> C
    C --> E[Relationship Paths]
    E --> F[Path Ranking]
    F --> G[LLM with Graph Context]
    G --> H[Explainable Answer]
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
