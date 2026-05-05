from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import HumanMessage, AIMessage
from rich.console import Console
from utils.config import get_llm, get_embeddings
from utils.tracking import TokenTracker
from dotenv import load_dotenv
import os
import time

load_dotenv()
console = Console()

class AgentState(TypedDict):
    query: str
    context: str
    messages: Sequence

def run_agentic_rag(query: str):
    tracker = TokenTracker()
    start_time = time.time()
    
    embeddings = get_embeddings()
    vectordb = Chroma(persist_directory=os.getenv("VECTOR_STORE_PATH"), embedding_function=embeddings)
    retriever = vectordb.as_retriever()
    
    llm = get_llm()
    
    # Define agent workflow
    workflow = StateGraph(AgentState)
    
    def retrieve(state):
        query = state["query"]
        docs = retriever.invoke(query)
        context = "\n".join([d.page_content for d in docs])
        return {"context": context}
    
    def generate(state):
        response = llm.invoke([
            HumanMessage(content=f"Context: {state['context']}\n\nQuestion: {state['query']}")
        ])
        tracker.update_from_llm_response(response)
        return {"messages": [response]}
    
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    app = workflow.compile()
    
    with console.status("[bold green]Agentic processing..."):
        result = app.invoke({"query": query, "messages": []})
    
    tracker.process_time = time.time() - start_time
    
    console.print(f"\n[bold blue]Query:[/bold blue] {query}")
    console.print(f"[bold green]Answer:[/bold green] {result['messages'][-1].content}")
    tracker.log()
    
    return {
        "answer": result['messages'][-1].content,
        "metadata": tracker.get_metadata()
    }

if __name__ == "__main__":
    run_agentic_rag("Explain agentic RAG")
