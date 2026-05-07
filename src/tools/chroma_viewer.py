import streamlit as st
import chromadb
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="ChromaDB Viewer", page_icon="", layout="wide")
st.title(" ChromaDB Viewer")

host = os.getenv("CHROMA_SERVER_HOST", "localhost")
port = int(os.getenv("CHROMA_SERVER_PORT", "8000"))


@st.cache_resource
def get_client():
    return chromadb.HttpClient(host=host, port=port)


client = get_client()

st.sidebar.header("Connection")
st.sidebar.text(f"Host: {host}:{port}")

try:
    hb = client.heartbeat()
    st.sidebar.success(f"Connected (heartbeat: {hb:.2f}s)")
except Exception as e:
    st.sidebar.error(f"Connection failed: {e}")
    st.stop()

collections = client.list_collections()
collection_names = [c.name for c in collections]

if not collection_names:
    st.warning("No collections found.")
    st.stop()

selected = st.sidebar.selectbox("Collection", collection_names)
col = next(c for c in collections if c.name == selected)
count = col.count()

st.header(f"Collection: `{selected}`")
st.caption(f"Total documents: {count}")

page_size = st.sidebar.slider("Page size", 5, 100, 20)
page = st.sidebar.number_input("Page", 0, max((count - 1) // page_size, 0), 0, step=1)
offset = page * page_size

st.sidebar.markdown(f"**Showing** {offset + 1}–{min(offset + page_size, count)} of {count}")

if count > 0:
    results = col.get(limit=page_size, offset=offset)
    for i, (doc_id, doc, meta) in enumerate(
        zip(results["ids"], results["documents"], results["metadatas"])
    ):
        label = f"#{offset + i + 1}: {doc_id}"
        if meta and "source" in meta:
            label += f" — {meta['source']}"
        with st.expander(label):
            if doc:
                st.text_area("Content", doc, height=120, key=f"doc_{i}")
            if meta:
                st.json(meta)
else:
    st.info("Collection is empty.")
