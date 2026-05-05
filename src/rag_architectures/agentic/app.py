from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import HumanMessage, AIMessage
from rich.console import Console
from dotenv import load_dotenv
import os

load_dotenv()
console = Console()

class AgentState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], "messages"]
    query: str
    context: str

def run_agentic_rag(query: str):
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=os.getenv("VECTOR_STORE_PATH"), embedding_function=embeddings)
    retriever = vectordb.as_retriever()
    
    llm = ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"))
    
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
        return {"messages": [response]}
    
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    app = workflow.compile()
    
    with console.status("[bold green]Agentic processing..."):
        result = app.invoke({"query": query, "messages": []})
    
    console.print(f"\n[bold blue]Query:[/bold blue] {query}")
    console.print(f"[bold green]Answer:[/bold green] {result['messages'][-1].content}")

if __name__ == "__main__":
    run_agentic_rag("Explain agentic RAG")
