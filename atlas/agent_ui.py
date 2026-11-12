
import uuid

import requests
import streamlit as st


API_BASE = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="ATLAS AI Agent",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 ATLAS AI")
st.caption("LangGraph Agent • RAG • Tools • Human Approval")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_approval" not in st.session_state:
    st.session_state.pending_approval = None


def call_api(endpoint, payload):
    response = requests.post(
        f"{API_BASE}{endpoint}",
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def process_result(result):
    if result["status"] == "approval_required":
        st.session_state.pending_approval = result["approval"]
    else:
        st.session_state.pending_approval = None
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
            }
        )


with st.sidebar:
    st.subheader("Conversation")
    st.code(st.session_state.thread_id)

    if st.button("New conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.pending_approval = None
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


pending = st.session_state.pending_approval

if pending:
    st.warning("Human approval required")

    for action in pending:
        st.write("**Action:**", action.get("action"))
        st.write("**Title:**", action.get("title"))
        st.write("**Description:**", action.get("description"))

    col1, col2 = st.columns(2)

    with col1:
        approve = st.button(
            "Approve",
            type="primary",
            use_container_width=True,
        )

    with col2:
        reject = st.button(
            "Reject",
            use_container_width=True,
        )

    if approve or reject:
        try:
            with st.spinner("Resuming agent..."):
                result = call_api(
                    "/agent/approve",
                    {
                        "thread_id": st.session_state.thread_id,
                        "approved": approve,
                    },
                )
            process_result(result)
            st.rerun()

        except requests.RequestException as exc:
            st.error(f"Approval request failed: {exc}")


prompt = st.chat_input(
    "Ask ATLAS AI...",
    disabled=bool(pending),
)

if prompt:
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    try:
        with st.spinner("ATLAS AI is thinking..."):
            result = call_api(
                "/agent/chat",
                {
                    "thread_id": st.session_state.thread_id,
                    "message": prompt,
                },
            )

        process_result(result)

    except requests.RequestException as exc:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"API request failed: {exc}",
            }
        )

    st.rerun()
