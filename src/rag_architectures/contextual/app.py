from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from rich.console import Console
from dotenv import load_dotenv
import os

load_dotenv()
console = Console()

def run_contextual_rag(query: str):
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=os.getenv("VECTOR_STORE_PATH"), embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    
    llm = ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"))
    
    # Contextual prompt that considers document context
    prompt = PromptTemplate(
        template="""
        You are given context from multiple documents. Consider the broader context of each document section.
        
        Context: {context}
        Question: {question}
        
        Provide a comprehensive answer that considers the full context:""",
        input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )
    
    with console.status("[bold green]Processing with context..."):
        result = qa_chain.invoke(query)
    
    console.print(f"\n[bold blue]Query:[/bold blue] {query}")
    console.print(f"[bold green]Answer:[/bold green] {result['result']}")

if __name__ == "__main__":
    run_contextual_rag("Explain contextual RAG")
