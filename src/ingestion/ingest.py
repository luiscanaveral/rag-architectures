from docling.document_converter import DocumentConverter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH", "./data")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./vectorstore")

def ingest_documents():
    converter = DocumentConverter()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    embeddings = OpenAIEmbeddings()
    
    docs = []
    data_dir = Path(DATA_PATH)
    
    for file_path in data_dir.glob("*.*"):
        print(f"Processing: {file_path}")
        result = converter.convert(str(file_path))
        text = result.document.export_to_markdown()
        chunks = splitter.split_text(text)
        docs.extend(chunks)
    
    print(f"Total chunks: {len(docs)}")
    vectordb = Chroma.from_texts(docs, embeddings, persist_directory=VECTOR_STORE_PATH)
    print(f"Documents ingested to {VECTOR_STORE_PATH}")

if __name__ == "__main__":
    ingest_documents()
