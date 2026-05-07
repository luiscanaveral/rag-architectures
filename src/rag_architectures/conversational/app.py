from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
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

def run_conversational_rag(query: str):
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
    retriever = vectordb.as_retriever()
    
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the context to answer questions."),
        ("human", "{input}"),
    ])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the context to answer questions.\nContext: {context}"),
        ("human", "{input}"),
    ])
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    history = ChatMessageHistory()
    chain_with_history = RunnableWithMessageHistory(
        retrieval_chain,
        lambda session_id: history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )
    
    with console.status("[bold green]Processing..."):
        result = chain_with_history.invoke(
            {"input": query},
            config={"configurable": {"session_id": "default"}, "callbacks": [tracker.callback]}
        )
    
    tracker.process_time = time.time() - start_time
    tracker.update_from_response()

    semantic_cache.store(query, result["answer"])
    
    console.print(f"\n[bold blue]Query:[/bold blue] {query}")
    console.print(f"[bold green]Answer:[/bold green] {result['answer']}")
    tracker.log()
    
    return {
        "answer": result['answer'],
        "metadata": {**tracker.get_metadata(), "cached": False}
    }

if __name__ == "__main__":
    run_conversational_rag("What is RAG?")
