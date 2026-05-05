from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from rich.console import Console
from dotenv import load_dotenv
import os

load_dotenv()
console = Console()

def run_graph_rag(query: str):
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=os.getenv("VECTOR_STORE_PATH"), embedding_function=embeddings)
    
    llm = ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"))
    retriever = vectordb.as_retriever()
    
    # GraphRAG uses relationship-based retrieval (simplified)
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        return_source_documents=True
    )
    
    with console.status("[bold green]Graph RAG processing..."):
        result = qa_chain.invoke(query)
    
    console.print(f"\n[bold blue]Query:[/bold blue] {query}")
    console.print(f"[bold green]Answer:[/bold green] {result['result']}")
    console.print("\n[bold yellow]Note:[/bold yellow] Full GraphRAG requires knowledge graph construction")

if __name__ == "__main__":
    run_graph_rag("How are entities related in GraphRAG?")
