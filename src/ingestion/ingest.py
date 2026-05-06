from docling.document_converter import DocumentConverter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from utils.config import get_embeddings
from dotenv import load_dotenv
import os
import psycopg2
from pathlib import Path

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH", "./data")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./vectorstore")
DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/rag")

def ingest_documents():
    converter = DocumentConverter()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    embeddings = get_embeddings()
    
    docs = []
    data_dir = Path(DATA_PATH)
    supported_extensions = {'.pdf', '.docx', '.pptx', '.html', '.md', '.csv', '.xlsx', '.txt'}
    
    # Ingest files
    for file_path in data_dir.glob("*.*"):
        if file_path.suffix.lower() not in supported_extensions:
            continue
        print(f"Processing: {file_path}")
        result = converter.convert(str(file_path))
        text = result.document.export_to_markdown()
        chunks = splitter.split_text(text)
        docs.extend(chunks)
    
    # Ingest database content
    print(f"Ingesting database content from {DB_URL}")
    try:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()
        
        # Ingest users
        cursor.execute("SELECT * FROM users")
        users = cursor.fetchall()
        for user in users:
            content = f"User: ID={user[0]}, Name={user[1]}, Email={user[2]}, Created={user[3]}"
            docs.append(content)
        
        # Ingest orders
        cursor.execute("SELECT * FROM orders")
        orders = cursor.fetchall()
        for order in orders:
            content = f"Order: ID={order[0]}, User_ID={order[1]}, Product={order[2]}, Amount={order[3]}, Date={order[4]}, Status={order[5]}"
            docs.append(content)
        
        conn.close()
        print(f"Ingested {len(users)} users and {len(orders)} orders")
    except Exception as e:
        print(f"Warning: Could not ingest database content: {e}")
    
    print(f"Total chunks: {len(docs)}")
    documents = [Document(page_content=chunk) for chunk in docs]
    vectordb = Chroma.from_documents(documents, embeddings, persist_directory=VECTOR_STORE_PATH)
    print(f"Documents ingested to {VECTOR_STORE_PATH}")

if __name__ == "__main__":
    ingest_documents()
