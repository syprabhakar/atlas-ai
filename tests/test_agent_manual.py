
"""Manual end-to-end tests for ATLAS AI Day 2."""

import uuid

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from atlas.agent.runtime import get_agent

graph = get_agent()


def run(question, thread_id):
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 12,
    }

    result = graph.invoke(
        {"messages": [HumanMessage(content=question)]},
        config=config,
    )

    if result.get("__interrupt__"):
        print("\nAPPROVAL REQUIRED")
        for item in result["__interrupt__"]:
            print(item.value)
        return config, True

    print("\nANSWER:")
    print(result["messages"][-1].content)
    return config, False


print("\n=== TEST 1: Calculator ===")
run("What is 18% of 12500?", "calculator-demo")

print("\n=== TEST 2: RAG ===")
run(
    "What are the three levels of WCAG conformance "
    "according to the uploaded documents?",
    "rag-demo",
)

print("\n=== TEST 3: Conversation memory ===")
run("My project name is ATLAS AI.", "memory-demo")
run("What is my project name?", "memory-demo")

print("\n=== TEST 4: Human approval ===")
thread_id = f"ticket-{uuid.uuid4().hex}"

config, interrupted = run(
    "Create a compliance ticket titled "
    "'Missing image alt text' with description "
    "'Important images lack alternative text'.",
    thread_id,
)

if interrupted:
    choice = input(
        "\nApprove mock ticket creation? (yes/no): "
    ).strip().lower()

    result = graph.invoke(
        Command(resume=(choice == "yes")),
        config=config,
    )

    print("\nFINAL RESULT:")
    if result.get("__interrupt__"):
        print("Another approval is required:", result["__interrupt__"])
    else:
        print(result["messages"][-1].content)
