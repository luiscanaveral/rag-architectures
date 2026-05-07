import sys
from pathlib import Path
import tempfile
import subprocess

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from docling.document_converter import DocumentConverter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from src.utils.config import get_embeddings
from dotenv import load_dotenv
import whisper
import os
import psycopg2
from pathlib import Path

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH", "./data")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", ".local/vectorstore")
DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/rag")

def transcribe_video(video_path):
    """Extract audio from video and transcribe using Whisper"""
    print(f"  Transcribing video: {video_path}")
    
    # Extract audio from video using ffmpeg
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        audio_path = tmp_file.name
    
    try:
        # Extract audio
        subprocess.run([
            'ffmpeg', '-i', str(video_path), '-vn', '-acodec', 'pcm_s16le', 
            '-ar', '16000', '-ac', '1', audio_path, '-y'
        ], capture_output=True, check=True)
        
        # Load Whisper model and transcribe
        model = whisper.load_model("turbo")
        result = model.transcribe(audio_path)
        
        return result["text"]
    finally:
        # Clean up temp file
        if Path(audio_path).exists():
            Path(audio_path).unlink()

def ingest_documents():
    # Converter for regular documents (PDF, DOCX, etc.)
    doc_converter = DocumentConverter()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    embeddings = get_embeddings()

    docs = []
    data_dir = Path(DATA_PATH).resolve()  # Get absolute path
    document_extensions = {'.pdf', '.docx', '.pptx', '.html', '.md', '.csv', '.xlsx', '.txt'}
    media_extensions = {'.mp4', '.avi', '.mov', '.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac'}

    # Ingest files
    for file_path in data_dir.glob("*.*"):
        file_path = file_path.resolve()  # Ensure absolute path
        ext = file_path.suffix.lower()
        if ext not in document_extensions and ext not in media_extensions:
            continue

        print(f"Processing: {file_path}")
        
        if not file_path.exists():
            print(f"  Warning: File not found, skipping: {file_path}")
            continue
            
        try:
            if ext in media_extensions:
                # Transcribe audio/video files using Whisper
                text = transcribe_video(str(file_path))
            else:
                # Use regular converter for documents
                result = doc_converter.convert(str(file_path))
                text = result.document.export_to_markdown()

            chunks = splitter.split_text(text)
            docs.extend(chunks)
            print(f"  Successfully processed: {len(chunks)} chunks")
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
            continue
    
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
