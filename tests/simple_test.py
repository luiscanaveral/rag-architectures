import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from rag_architectures.simple.app import run_simple_rag

st.title("Simple RAG Test")
query = st.text_input("Enter your question:")
if st.button("Submit"):
    with st.spinner("Processing..."):
        result = run_simple_rag(query)
        st.write(result)
