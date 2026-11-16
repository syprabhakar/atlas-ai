
import json
import uuid
from pathlib import Path

from langchain_core.messages import HumanMessage

from atlas.agent.runtime import get_agent


DATASET_PATH = Path("evals/dataset.json")


def extract_tool_calls(messages):
    calls = []

    for message in messages:
        for call in getattr(message, "tool_calls", []) or []:
            calls.append(call["name"])

    return calls


def run_evaluation():
    dataset = json.loads(DATASET_PATH.read_text())
    graph = get_agent()

    results = []

    for case in dataset:
        thread_id = f"eval-{uuid.uuid4()}"

        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 12,
        }

        print(f"\nEvaluating: {case['id']}")

        result = graph.invoke(
            {
                "messages": [
                    HumanMessage(content=case["question"])
                ]
            },
            config=config,
        )

        messages = result.get("messages", [])
        actual_tools = extract_tool_calls(messages)

        expected_tool = case["expected_tool"]
        tool_passed = expected_tool in actual_tools

        interrupted = bool(result.get("__interrupt__"))

        status_passed = (
            interrupted
            if case.get("expected_status") == "approval_required"
            else True
        )

        passed = tool_passed and status_passed

        results.append(
            {
                "id": case["id"],
                "expected_tool": expected_tool,
                "actual_tools": actual_tools,
                "tool_passed": tool_passed,
                "status_passed": status_passed,
                "passed": passed,
            }
        )

        print("Expected tool:", expected_tool)
        print("Actual tools:", actual_tools)
        print("Result:", "PASS" if passed else "FAIL")

    passed_count = sum(item["passed"] for item in results)
    total = len(results)

    print("\n========== EVALUATION SUMMARY ==========")
    print(f"Passed: {passed_count}/{total}")
    print(f"Pass rate: {passed_count / total:.1%}")

    output = Path("evals/results.json")
    output.write_text(json.dumps(results, indent=2))

    print(f"Results saved to {output}")


if __name__ == "__main__":
    run_evaluation()
