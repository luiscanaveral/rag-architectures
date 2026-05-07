from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from rich.console import Console
from utils.config import get_llm, get_embeddings
from utils.tracking import TokenTracker
from utils.semantic_cache import SemanticCache
from dotenv import load_dotenv
import os
import time

load_dotenv()
console = Console()

semantic_cache = SemanticCache(embedding_func=get_embeddings())

def run_corrective_rag(query: str):
    cached = semantic_cache.lookup(query)
    if cached:
        console.print("[bold yellow]Cache HIT[/bold yellow]")
        console.print(f"\n[bold blue]Query:[/bold blue] {query}")
        console.print(f"[bold green]Answer:[/bold green] {cached}")
        return {"answer": cached, "metadata": {"cached": True}}

    tracker = TokenTracker()
    start_time = time.time()
    
    embeddings = get_embeddings()
    vectordb = Chroma(persist_directory=os.getenv("VECTOR_STORE_PATH"), embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    
    llm = get_llm()
    
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
        result = qa_chain.invoke(query, config={"callbacks": [tracker.callback]})
    
    tracker.process_time = time.time() - start_time
    tracker.update_from_response()

    semantic_cache.store(query, result["result"])
    
    console.print(f"\n[bold blue]Query:[/bold blue] {query}")
    console.print(f"[bold green]Answer:[/bold green] {result['result']}")
    tracker.log()
    
    return {
        "answer": result['result'],
        "metadata": tracker.get_metadata()
    }

if __name__ == "__main__":
    run_corrective_rag("What is corrective RAG?")
