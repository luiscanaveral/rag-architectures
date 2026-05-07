from langchain_classic.chains import RetrievalQA
from rich.console import Console
from utils.config import get_llm, get_vectorstore
from utils.tracking import TokenTracker
from utils.langfuse_tracing import LangfuseRestCallback
from dotenv import load_dotenv
import time

load_dotenv()
console = Console()

langfuse_handler = LangfuseRestCallback()

def run_simple_rag(query: str):
    tracker = TokenTracker()
    start_time = time.time()
    
    vectordb = get_vectorstore()
    llm = get_llm()
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever())
    
    with console.status("[bold green]Processing..."):
        result = qa_chain.invoke(query, config={"callbacks": [tracker.callback, langfuse_handler]})
    
    # Calculate process time after invocation
    tracker.process_time = time.time() - start_time
    
    # Update token counts from the callback
    tracker.update_from_response()
    
    tracker.log()
    
    console.print(f"\n[bold blue]Query:[/bold blue] {query}")
    console.print(f"[bold green]Answer:[/bold green] {result['result']}")
    
    return {
        "answer": result['result'],
        "metadata": tracker.get_metadata()
    }

if __name__ == "__main__":
    run_simple_rag("What is RAG?")
