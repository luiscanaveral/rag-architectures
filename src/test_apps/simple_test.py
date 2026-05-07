import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from rag_architectures.simple.app import run_simple_rag

st.title("Simple RAG Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "metadata" in message and message["role"] == "assistant":
            with st.expander("Metadata"):
                meta = message["metadata"]
                st.text(f"Prompt tokens: {meta.get('prompt_tokens', 0)}")
                st.text(f"Completion tokens: {meta.get('completion_tokens', 0)}")
                st.text(f"Total tokens: {meta.get('total_tokens', 0)}")
                st.text(f"Process time: {meta.get('process_time', 0)}s")

if prompt := st.chat_input("What is your question?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            result = run_simple_rag(prompt)
            answer = result["answer"] if isinstance(result, dict) else result
            st.markdown(answer)
            
            if isinstance(result, dict) and "metadata" in result:
                meta = result["metadata"]
                with st.expander("Metadata"):
                    st.text(f"Prompt tokens: {meta.get('prompt_tokens', 0)}")
                    st.text(f"Completion tokens: {meta.get('completion_tokens', 0)}")
                    st.text(f"Total tokens: {meta.get('total_tokens', 0)}")
                    st.text(f"Process time: {meta.get('process_time', 0)}s")
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "metadata": result.get("metadata", {}) if isinstance(result, dict) else {}
            })
