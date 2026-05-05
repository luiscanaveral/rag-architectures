from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from rich.console import Console
from dotenv import load_dotenv
import os

load_dotenv()
console = Console()

def run_standard_rag(query: str, k: int = 4):
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=os.getenv("VECTOR_STORE_PATH"), embedding_function=embeddings)
    
    llm = ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"))
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, return_source_documents=True)
    
    with console.status("[bold green]Processing..."):
        result = qa_chain.invoke(query)
    
    console.print(f"\n[bold blue]Query:[/bold blue] {query}")
    console.print(f"[bold green]Answer:[/bold green] {result['result']}")
    console.print("\n[bold yellow]Sources:[/bold yellow]")
    for doc in result['source_documents']:
        console.print(f"  - {doc.page_content[:100]}...")

if __name__ == "__main__":
    run_standard_rag("Explain standard RAG architecture")
