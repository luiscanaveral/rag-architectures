from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from rich.console import Console
from utils.config import get_llm, get_embeddings
from utils.tracking import TokenTracker
from dotenv import load_dotenv
import os
import time

load_dotenv()
console = Console()

def run_simple_rag(query: str):
    tracker = TokenTracker()
    start_time = time.time()
    
    embeddings = get_embeddings()
    vectordb = Chroma(persist_directory=os.getenv("VECTOR_STORE_PATH"), embedding_function=embeddings)
    
    llm = get_llm()
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever())
    
    with console.status("[bold green]Processing..."):
        result = qa_chain.invoke(query)
    
    tracker.process_time = time.time() - start_time
    
    if 'response' in result:
        tracker.update_from_llm_response(result.get('response'))
    elif hasattr(result.get('result'), '__dict__'):
        tracker.update_from_llm_response(result.get('result'))
    
    console.print(f"\n[bold blue]Query:[/bold blue] {query}")
    console.print(f"[bold green]Answer:[/bold green] {result['result']}")
    tracker.log()
    
    return {
        "answer": result['result'],
        "metadata": tracker.get_metadata()
    }

if __name__ == "__main__":
    run_simple_rag("What is RAG?")
