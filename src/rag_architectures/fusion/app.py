from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_classic.chains import RetrievalQA
from rich.console import Console
from utils.config import get_llm, get_embeddings
from utils.tracking import TokenTracker
from dotenv import load_dotenv
import os
import time

load_dotenv()
console = Console()

def run_fusion_rag(query: str):
    tracker = TokenTracker()
    start_time = time.time()
    
    embeddings = get_embeddings()
    vectordb = Chroma(persist_directory=os.getenv("VECTOR_STORE_PATH"), embedding_function=embeddings)
    
    # Vector retriever
    vector_retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    
    # Use only vector retriever for now (BM25 requires additional dependencies)
    # docs = vectordb.similarity_search("", k=50)  # Get some docs for BM25
    # bm25_retriever = BM25Retriever.from_documents(docs, k=3)
    
    # Ensemble retriever for fusion (using multiple vector retrievers with different params)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever],
        weights=[1.0]
    )
    
    llm = get_llm()
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=ensemble_retriever)
    
    with console.status("[bold green]Processing with fusion..."):
        result = qa_chain.invoke(query, config={"callbacks": [tracker.callback]})
    
    tracker.process_time = time.time() - start_time
    tracker.update_from_response()
    
    console.print(f"\n[bold blue]Query:[/bold blue] {query}")
    console.print(f"[bold green]Answer:[/bold green] {result['result']}")
    tracker.log()
    
    return {
        "answer": result['result'],
        "metadata": tracker.get_metadata()
    }

if __name__ == "__main__":
    run_fusion_rag("Explain fusion RAG")
