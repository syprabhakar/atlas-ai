
"""FastAPI routes for the ATLAS AI agent."""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from atlas.agent.runtime import get_agent


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["Agent"])


class ChatRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1)


class ApprovalRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=128)
    approved: bool


def get_config(thread_id: str):
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 12,
    }


def format_result(result):
    interrupts = result.get("__interrupt__", ())

    if interrupts:
        return {
            "status": "approval_required",
            "approval": [
                item.value for item in interrupts
            ],
        }

    return {
        "status": "completed",
        "answer": str(result["messages"][-1].content),
    }


@router.post("/chat")
def chat(request: ChatRequest):
    graph = get_agent()
    config = get_config(request.thread_id)

    state = graph.get_state(config)

    if state.next:
        raise HTTPException(
            status_code=409,
            detail="This conversation has a pending workflow.",
        )

    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content=request.message)]},
            config=config,
        )
    except Exception as exc:
        logger.exception("Agent chat failed")
        raise HTTPException(
            503,
            "Agent execution failed; see server logs."
        ) from exc

    return format_result(result)


@router.post("/approve")
def approve(request: ApprovalRequest):
    graph = get_agent()
    config = get_config(request.thread_id)

    state = graph.get_state(config)

    if "approval" not in state.next:
        raise HTTPException(
            status_code=409,
            detail="No pending approval exists.",
        )

    try:
        result = graph.invoke(
            Command(resume=request.approved),
            config=config
        )
    except Exception as exc:
        logger.exception("Agent approval resume failed")
        raise HTTPException(
            503,
            "Workflow resume failed; see server logs."
        ) from exc

    return format_result(result)


@router.get("/history/{thread_id}")
def history(thread_id: str):
    graph = get_agent()
    state = graph.get_state(get_config(thread_id))

    messages = state.values.get("messages", [])

    return {
        "thread_id": thread_id,
        "messages": [
            {
                "type": message.type,
                "content": str(message.content),
            }
            for message in messages
        ],
    }
