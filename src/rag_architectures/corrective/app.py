from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from rich.console import Console
from src.utils.config import get_llm, get_embeddings
from dotenv import load_dotenv
import os

load_dotenv()
console = Console()

def run_corrective_rag(query: str):
    embeddings = get_embeddings()
    vectordb = Chroma(persist_directory=os.getenv("VECTOR_STORE_PATH"), embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    
    llm = get_llm()
    
    # Corrective prompt that validates retrieved docs
    prompt = PromptTemplate(
        template="""
        Based on the following context, answer the question.
        If the context is not relevant or seems incorrect, state that you cannot answer accurately.
        
        Context: {context}
        Question: {question}
        
        Answer:""",
        input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm, 
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    
    with console.status("[bold green]Processing with correction..."):
        result = qa_chain.invoke(query)
    
    console.print(f"\n[bold blue]Query:[/bold blue] {query}")
    console.print(f"[bold green]Answer:[/bold green] {result['result']}")

if __name__ == "__main__":
    run_corrective_rag("What is corrective RAG?")
