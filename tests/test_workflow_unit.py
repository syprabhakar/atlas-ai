
"""Fast, offline checks of agent routing and approval safety."""

from unittest.mock import patch
from langchain_core.messages import AIMessage
from langgraph.graph import END

from atlas.agent.graph import route_tools, approval_node


def test_route_search_to_ordinary_tools():
    msg = AIMessage(
        content="",
        tool_calls=[{
            "name": "search_documents",
            "args": {"query": "WCAG"},
            "id": "s1"
        }]
    )

    assert route_tools({"messages": [msg]}) == "ordinary_tools"


def test_route_ticket_to_approval():
    msg = AIMessage(
        content="",
        tool_calls=[{
            "name": "create_ticket",
            "args": {
                "title": "Audit",
                "description": "Review WCAG"
            },
            "id": "t1"
        }]
    )

    assert route_tools({"messages": [msg]}) == "approval"


def test_no_tool_ends_workflow():
    assert route_tools({
        "messages": [AIMessage(content="Done")]
    }) == END


def test_ticket_rejection_does_not_create_ticket():
    msg = AIMessage(
        content="",
        tool_calls=[{
            "name": "create_ticket",
            "args": {
                "title": "Audit",
                "description": "Review WCAG"
            },
            "id": "t1"
        }]
    )

    with patch("atlas.agent.graph.interrupt", return_value=False):
        output = approval_node({"messages": [msg]})

    assert "rejected" in output["messages"][0].content.lower()
    assert "Mock ticket created" not in output["messages"][0].content


def test_ticket_approval_creates_mock_ticket():
    msg = AIMessage(
        content="",
        tool_calls=[{
            "name": "create_ticket",
            "args": {
                "title": "Audit",
                "description": "Review WCAG"
            },
            "id": "t1"
        }]
    )

    with patch("atlas.agent.graph.interrupt", return_value=True):
        output = approval_node({"messages": [msg]})

    assert "Mock ticket created: ATLAS-" in output["messages"][0].content
