
"""Single Streamlit demo for RAG ingestion and agent workflows."""
import os
import uuid
import requests
import streamlit as st

BASE = os.getenv("ATLAS_API_URL", "http://127.0.0.1:8000").rstrip("/")
st.set_page_config(page_title="ATLAS AI", page_icon="🤖", layout="wide")
st.title("ATLAS AI")
st.caption("Hybrid RAG · LangGraph agent · Human approval · Evaluation")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = []


def api(method, path, **kwargs):
    url = f"{BASE}{path}"

    print(f"[ATLAS DEBUG] API URL: {url}")
    print(f"[ATLAS DEBUG] Request: {kwargs.get('json')}")

    response = requests.request(
        method,
        url,
        timeout=180,
        **kwargs
    )

    print(f"[ATLAS DEBUG] Status: {response.status_code}")
    print(f"[ATLAS DEBUG] Response: {response.text[:1000]}")

    response.raise_for_status()
    return response.json()


def handle_agent_result(result):
    if result["status"] == "approval_required":
        st.session_state.pending = result.get("approval", [])
    else:
        st.session_state.pending = []
        st.session_state.messages.append({
            "role": "assistant",
            "content": result.get("answer", "")
        })


with st.sidebar:
    st.subheader("Documents")
    uploaded = st.file_uploader("Upload a text-based PDF", type=["pdf"])

    if uploaded is not None and st.button("Index PDF"):
        try:
            result = api(
                "POST",
                "/documents",
                files={
                    "file": (
                        uploaded.name,
                        uploaded.getvalue(),
                        "application/pdf"
                    )
                }
            )
            st.success(f"{result['status']}: {result['chunks']} new chunks")
        except requests.RequestException as exc:
            st.error(f"Upload failed: {exc}")

    try:
        st.metric("Indexed chunks", api("GET", "/health")["chunks"])
    except (requests.RequestException, KeyError):
        st.warning("Start FastAPI first: uvicorn atlas.api:app --reload")

    st.divider()
    st.subheader("Conversation")
    st.code(st.session_state.thread_id)

    if st.button("New conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.pending = []
        st.rerun()

    st.caption("Local development demo only. Do not upload sensitive documents.")


tab_rag, tab_agent = st.tabs(["Document Q&A", "Agent & approvals"])

with tab_rag:
    question = st.text_input(
        "Question about indexed documents",
        key="rag_question"
    )
    top_k = st.slider("Source chunks", 1, 10, 5)

    if st.button("Ask documents", disabled=not question.strip()):
        try:
            with st.spinner("Retrieving and generating..."):
                result = api(
                    "POST",
                    "/ask",
                    json={"question": question, "top_k": top_k}
                )

            st.markdown(result["answer"])
            st.caption(f"Mode: {result['mode']}")

            for source in result["sources"]:
                with st.expander(
                    f"{source['citation']} "
                    f"{source['filename']} · page {source['page']}"
                ):
                    st.write(source["text"])

        except requests.RequestException as exc:
            st.error(f"Question failed: {exc}")


with tab_agent:
    st.caption(
        "Try: Search the WCAG documents, summarize conformance "
        "levels, and prepare a compliance ticket for approval."
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.session_state.pending:
        st.warning("Human approval required — review the proposed action")

        for action in st.session_state.pending:
            st.markdown(f"**Action:** {action.get('action', '')}")
            st.markdown(f"**Title:** {action.get('title', '')}")
            st.text_area(
                "Description",
                action.get("description", ""),
                disabled=True,
                key=f"desc_{action.get('tool_call_id', '')}"
            )

        a, b = st.columns(2)
        approve = a.button("Approve", type="primary")
        reject = b.button("Reject")

        if approve or reject:
            try:
                with st.spinner("Resuming workflow..."):
                    result = api(
                        "POST",
                        "/agent/approve",
                        json={
                            "thread_id": st.session_state.thread_id,
                            "approved": approve
                        }
                    )
                    handle_agent_result(result)
                st.rerun()
            except requests.RequestException as exc:
                st.error(f"Approval failed: {exc}")

    prompt = st.chat_input(
        "Ask the agent to search, calculate, or prepare a ticket",
        disabled=bool(st.session_state.pending)
    )

    if prompt:
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        try:
            with st.spinner("Agent is working..."):
                result = api(
                    "POST",
                    "/agent/chat",
                    json={
                        "thread_id": st.session_state.thread_id,
                        "message": prompt
                    }
                )
                handle_agent_result(result)

        except requests.RequestException as exc:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Request failed: {exc}"
            })

        st.rerun()
