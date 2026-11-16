
import json
import time
import uuid
from pathlib import Path

from langchain_core.messages import HumanMessage
from atlas.agent.runtime import get_agent


OUTPUT_PATH = Path("evals/traces.json")


def trace_question(question):
    graph = get_agent()
    thread_id = f"trace-{uuid.uuid4()}"

    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 12,
    }

    trace = {
        "thread_id": thread_id,
        "question": question,
        "nodes": [],
        "tool_calls": [],
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "latency_seconds": 0,
        "status": "running",
    }

    start = time.perf_counter()

    try:
        for event in graph.stream(
            {"messages": [HumanMessage(content=question)]},
            config=config,
            stream_mode="updates",
        ):
            for node_name, update in event.items():
                trace["nodes"].append(node_name)

                if node_name == "__interrupt__":
                    trace["status"] = "approval_required"
                    continue

                if not isinstance(update, dict):
                    continue

                for message in update.get("messages", []):
                    # Capture tool calls
                    for call in (
                        getattr(message, "tool_calls", None) or []
                    ):
                        trace["tool_calls"].append({
                            "name": call["name"],
                            "args": call["args"],
                        })

                    # Capture LLM token usage
                    usage = getattr(
                        message, "usage_metadata", None
                    )

                    if usage:
                        trace["input_tokens"] += usage.get(
                            "input_tokens", 0
                        )
                        trace["output_tokens"] += usage.get(
                            "output_tokens", 0
                        )

                    # Capture final AI answer
                    if (
                        message.type == "ai"
                        and message.content
                        and not getattr(
                            message, "tool_calls", None
                        )
                    ):
                        trace["answer"] = str(message.content)

        if trace["status"] == "running":
            trace["status"] = "completed"

    except Exception as exc:
        trace["status"] = "failed"
        trace["error"] = str(exc)

    finally:
        trace["latency_seconds"] = round(
            time.perf_counter() - start, 3
        )

        trace["total_tokens"] = (
            trace["input_tokens"]
            + trace["output_tokens"]
        )

    return trace


def main():
    questions = [
        "Use the calculator to calculate 12500 * 18 / 100",
        "Search the documents for the three WCAG conformance levels.",
    ]

    results = []

    for question in questions:
        print("\n" + "=" * 60)
        print("Question:", question)

        trace = trace_question(question)
        results.append(trace)

        print("Status:", trace["status"])
        print("Nodes:", trace["nodes"])

        print(
            "Tools:",
            [tool["name"] for tool in trace["tool_calls"]],
        )

        print("Input tokens:", trace["input_tokens"])
        print("Output tokens:", trace["output_tokens"])
        print("Total tokens:", trace["total_tokens"])

        print(
            "Latency:",
            trace["latency_seconds"],
            "seconds",
        )

        if "answer" in trace:
            print("Answer:", trace["answer"][:500])

        if "error" in trace:
            print("Error:", trace["error"])

    OUTPUT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    OUTPUT_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("Traces saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
