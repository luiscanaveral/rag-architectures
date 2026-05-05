from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

def get_llm():
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    
    if provider == "ollama":
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )
    else:
        return ChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE")
        )

def get_embeddings():
    provider = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
    
    if provider == "ollama":
        return OllamaEmbeddings(model=os.getenv("OLLAMA_MODEL", "llama3.2"))
    else:
        return OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        )

def print_config():
    from rich.console import Console
    console = Console()
    console.print(f"[bold]LLM Provider:[/bold] {os.getenv('LLM_PROVIDER')}")
    console.print(f"[bold]LLM Model:[/bold] {os.getenv('LLM_MODEL') if os.getenv('LLM_PROVIDER') == 'openai' else os.getenv('OLLAMA_MODEL')}")
