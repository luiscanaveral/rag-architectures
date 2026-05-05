from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from rich.console import Console
from src.utils.config import get_llm, get_embeddings
from dotenv import load_dotenv
import os

load_dotenv()
console = Console()

def run_simple_rag(query: str):
    embeddings = get_embeddings()
    vectordb = Chroma(persist_directory=os.getenv("VECTOR_STORE_PATH"), embedding_function=embeddings)
    
    llm = get_llm()
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever())
    
    with console.status("[bold green]Processing..."):
        result = qa_chain.invoke(query)
    
    console.print(f"\n[bold blue]Query:[/bold blue] {query}")
    console.print(f"[bold green]Answer:[/bold green] {result['result']}")
    return result['result']

if __name__ == "__main__":
    run_simple_rag("What is RAG?")
