from langchain_community.vectorstores import Chroma
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
from langchain_classic.chains import RetrievalQA
from rich.console import Console
from src.utils.config import get_llm, get_embeddings
from dotenv import load_dotenv
import os

load_dotenv()
console = Console()

def run_fusion_rag(query: str):
    embeddings = get_embeddings()
    vectordb = Chroma(persist_directory=os.getenv("VECTOR_STORE_PATH"), embedding_function=embeddings)
    
    # Vector retriever
    vector_retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    
    # BM25 keyword retriever (simplified - would need docs)
    docs = vectordb.similarity_search("", k=50)  # Get some docs for BM25
    bm25_retriever = BM25Retriever.from_documents(docs, k=3)
    
    # Ensemble retriever for fusion
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.7, 0.3]
    )
    
    llm = ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"))
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=ensemble_retriever)
    
    with console.status("[bold green]Processing with fusion..."):
        result = qa_chain.invoke(query)
    
    console.print(f"\n[bold blue]Query:[/bold blue] {query}")
    console.print(f"[bold green]Answer:[/bold green] {result['result']}")

if __name__ == "__main__":
    run_fusion_rag("Explain fusion RAG")
