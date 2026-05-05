from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from rich.console import Console
from dotenv import load_dotenv
import os

load_dotenv()
console = Console()

def run_simple_rag(query: str):
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=os.getenv("VECTOR_STORE_PATH"), embedding_function=embeddings)
    
    llm = ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"))
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever())
    
    with console.status("[bold green]Processing..."):
        result = qa_chain.invoke(query)
    
    console.print(f"\n[bold blue]Query:[/bold blue] {query}")
    console.print(f"[bold green]Answer:[/bold green] {result['result']}")

if __name__ == "__main__":
    run_simple_rag("What is RAG?")
