import streamlit as st
from src.rag_architectures.simple.app import run_simple_rag

st.title("Simple RAG Test")
query = st.text_input("Enter your question:")
if st.button("Submit"):
    with st.spinner("Processing..."):
        run_simple_rag(query)
