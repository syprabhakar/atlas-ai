"""Streamlit UI. Start FastAPI first, then run this module with Streamlit."""
import os
import requests
import streamlit as st

BASE = os.getenv("ATLAS_API_URL", "http://127.0.0.1:8000")
st.set_page_config(page_title="ATLAS AI", page_icon="📚", layout="wide")
st.title("ATLAS AI — Enterprise Knowledge Assistant")
st.caption("Day 1: PDF ingestion • BM25 + semantic search • RRF • reranking • citations")
with st.sidebar:
    st.header("Documents")
    uploaded = st.file_uploader("Upload a text-based PDF", type=["pdf"])
    if uploaded and st.button("Index document"):
        try:
            r = requests.post(f"{BASE}/documents", files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}, timeout=180)
            r.raise_for_status()
            st.success(r.json())
        except requests.RequestException as e:
            st.error(f"Upload failed: {e}")
    try:
        st.metric("Indexed chunks", requests.get(f"{BASE}/health", timeout=5).json()["chunks"])
    except Exception:
        st.warning("Start the FastAPI server first.")
question = st.text_input("Ask a question about your documents")
top_k = st.slider("Number of source chunks", 1, 10, 5)
if st.button("Ask ATLAS", type="primary") and question.strip():
    try:
        with st.spinner("Searching and generating answer..."):
            response = requests.post(f"{BASE}/ask", json={"question": question, "top_k": top_k}, timeout=180)
            response.raise_for_status()
            result = response.json()
        st.subheader("Answer")
        st.write(result["answer"])
        st.caption(f"Answer mode: {result['mode']}")
        st.subheader("Retrieved sources")
        for source in result["sources"]:
            with st.expander(f"{source['citation']} {source['filename']} — page {source['page']}"):
                st.write(source["text"])
                st.caption(f"RRF score: {source['rrf_score']}")
    except requests.RequestException as e:
        st.error(f"Request failed: {e}")
