from dotenv import load_dotenv
import os

load_dotenv()

def setup_tracing():
    provider = os.getenv("TRACING_PROVIDER", "langsmith").lower()
    
    if provider == "langfuse":
        os.environ["LANGFUSE_TRACING_ENABLED"] = "true"
        os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
        os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY", "")
        print(f"Tracing: LangFuse enabled at {os.getenv('LANGFUSE_HOST')}")
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
        print(f"Tracing: LangSmith enabled")

if __name__ == "__main__":
    setup_tracing()
