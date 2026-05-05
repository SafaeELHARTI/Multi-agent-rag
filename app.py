import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Multi-Agent RAG",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Multi-Agent RAG Assistant")
st.caption("Powered by Mistral · pgvector · LangGraph")

with st.sidebar:
    st.header("About")
    st.markdown("""
    **Architecture:**
    - 🔍 Retriever Agent — semantic search
    - 📝 Summarizer Agent — context condensation  
    - 💡 Answer Agent — grounded generation
    
    **Stack:**
    - Mistral API (embeddings + LLM)
    - pgvector (vector store)
    - LangGraph (orchestration)
    - FastAPI (backend)
    """)
    
    st.divider()
    
    if st.button("Check API Health"):
        try:
            r = requests.get(f"{API_URL}/health")
            if r.status_code == 200:
                st.success("API is running")
            else:
                st.error("API error")
        except:
            st.error("API unreachable — start uvicorn first")

st.divider()

question = st.text_input(
    "Ask a question about your documents",
    placeholder="e.g. What is the attention mechanism?"
)

if st.button("Ask", type="primary") and question:
    with st.spinner("Running multi-agent pipeline..."):
        try:
            response = requests.post(
                f"{API_URL}/ask",
                json={"question": question},
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                
                st.success("Done!")
                
                st.subheader("Answer")
                st.markdown(data["answer"])
                
                with st.expander(f"Retrieved {data['chunks_found']} chunks"):
                    st.markdown(data["summary"])
                    
            else:
                st.error(f"Error {response.status_code}: {response.json()}")
                
        except requests.exceptions.Timeout:
            st.error("Request timed out — Mistral API may be slow")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API — make sure uvicorn is running")

st.divider()
st.caption("Multi-Agent RAG · Built for Lyha technical interview")