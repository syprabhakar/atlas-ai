
import uuid

import pytest
from langchain_core.messages import HumanMessage

from atlas.agent.runtime import get_agent


@pytest.fixture(scope="module")
def graph():
    """Initialize the existing LangGraph agent once."""
    return get_agent()


def run_agent(graph, question):
    """Execute a request with a fresh conversation thread."""
    config = {
        "configurable": {
            "thread_id": f"regression-{uuid.uuid4()}"
        },
        "recursion_limit": 12,
    }

    return graph.invoke(
        {"messages": [HumanMessage(content=question)]},
        config=config,
    )


def get_tool_calls(result):
    """Extract tool names selected by the agent."""
    calls = []

    for message in result.get("messages", []):
        for call in getattr(message, "tool_calls", None) or []:
            calls.append(call["name"])

    return calls


def test_calculator_tool_selection(graph):
    """The agent should choose the calculator tool."""
    result = run_agent(
        graph,
        "Use the calculator to calculate 12500 * 18 / 100",
    )

    assert "calculator" in get_tool_calls(result)


def test_calculator_result(graph):
    """The calculator should return the correct result."""
    result = run_agent(
        graph,
        "Use the calculator to calculate 12500 * 18 / 100",
    )

    tool_messages = [
        message
        for message in result.get("messages", [])
        if message.type == "tool"
        and message.name == "calculator"
    ]

    assert tool_messages, "Calculator was not executed"

    assert any(
        abs(float(message.content) - 2250.0) < 0.001
        for message in tool_messages
    ), "Incorrect calculator result"


def test_rag_tool_selection(graph):
    """The agent should use document search for WCAG questions."""
    result = run_agent(
        graph,
        "Search the uploaded documents for the "
        "three WCAG conformance levels.",
    )

    assert "search_documents" in get_tool_calls(result)


def test_rag_answer_contains_expected_levels(graph):
    """The final answer should include A, AA, and AAA."""
    result = run_agent(
        graph,
        "Search the uploaded documents for the "
        "three WCAG conformance levels.",
    )

    ai_messages = [
        message
        for message in result.get("messages", [])
        if message.type == "ai"
        and message.content
        and not getattr(message, "tool_calls", None)
    ]

    assert ai_messages, "No final AI answer generated"

    answer = str(ai_messages[-1].content).lower()

    for level in ["level a", "level aa", "level aaa"]:
        assert level in answer, f"Missing WCAG {level}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
