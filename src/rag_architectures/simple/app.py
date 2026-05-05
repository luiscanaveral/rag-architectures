from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.callbacks import StdOutCallbackHandler
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
        result = qa_chain.invoke(query, config={"callbacks": [tracker.callback]})
    
    # Calculate process time after invocation
    tracker.process_time = time.time() - start_time
    
    # Update token counts from the callback
    tracker.update_from_response()
    
    # Also try to get tokens from the LLM response if available
    if 'result' in result:
        # The chain might not expose token usage directly
        # We'll rely on the callback to capture tokens
        pass
    
    tracker.log()
    
    console.print(f"\n[bold blue]Query:[/bold blue] {query}")
    console.print(f"[bold green]Answer:[/bold green] {result['result']}")
    
    return {
        "answer": result['result'],
        "metadata": tracker.get_metadata()
    }

if __name__ == "__main__":
    run_simple_rag("What is RAG?")
