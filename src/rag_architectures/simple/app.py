from langchain_classic.chains import RetrievalQA
from rich.console import Console
from utils.config import get_llm, get_vectorstore, get_embeddings
from utils.tracking import TokenTracker
from utils.langfuse_tracing import LangfuseRestCallback
from utils.semantic_cache import SemanticCache
from dotenv import load_dotenv
import time

load_dotenv()
console = Console()

langfuse_handler = LangfuseRestCallback()
semantic_cache = SemanticCache(
    embedding_func=get_embeddings(),
    redis_host="localhost",
    redis_port=6379,
)

def run_simple_rag(query: str):
    cached = semantic_cache.lookup(query)
    if cached:
        console.print("[bold yellow]Cache HIT[/bold yellow]")
        console.print(f"\n[bold blue]Query:[/bold blue] {query}")
        console.print(f"[bold green]Answer:[/bold green] {cached}")
        return {"answer": cached, "metadata": {"cached": True}}

    tracker = TokenTracker()
    start_time = time.time()

    vectordb = get_vectorstore()
    llm = get_llm()
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever())

    with console.status("[bold green]Processing..."):
        result = qa_chain.invoke(query, config={"callbacks": [tracker.callback, langfuse_handler]})

    tracker.process_time = time.time() - start_time
    tracker.update_from_response()

    semantic_cache.store(query, result["result"])

    tracker.log()

    console.print(f"\n[bold blue]Query:[/bold blue] {query}")
    console.print(f"[bold green]Answer:[/bold green] {result['result']}")

    return {
        "answer": result["result"],
        "metadata": tracker.get_metadata(),
    }

if __name__ == "__main__":
    run_simple_rag("What is RAG?")
