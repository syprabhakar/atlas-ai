
"""Tools available to the ATLAS AI agent."""

import ast
import operator
from functools import lru_cache

from langchain_core.tools import tool
from atlas.core import AtlasRAG


@lru_cache(maxsize=1)
def get_rag() -> AtlasRAG:
    return AtlasRAG()


@tool
def search_documents(query: str) -> str:
    """Search uploaded documents for relevant facts and evidence."""
    results = get_rag().search(query, top_k=5)

    if not results:
        return "NO_EVIDENCE: No indexed document evidence found."

    return "\n\n".join(
        f"[{i}] {r['filename']} page {r['page']}\n"
        f"Chunk ID: {r['id']}\n{r['text']}"
        for i, r in enumerate(results, 1)
    )


OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def evaluate(node):
    if isinstance(node, ast.Expression):
        return evaluate(node.body)

    if isinstance(node, ast.Constant):
        if type(node.value) in (int, float):
            if abs(node.value) > 1e12:
                raise ValueError("Number too large")
            return node.value

    if isinstance(node, ast.UnaryOp) and isinstance(
        node.op, (ast.UAdd, ast.USub)
    ):
        value = evaluate(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value

    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        left = evaluate(node.left)
        right = evaluate(node.right)
        result = OPERATORS[type(node.op)](left, right)
        if abs(result) > 1e15:
            raise ValueError("Result too large")
        return result

    raise ValueError("Unsupported mathematical expression")


@tool
def calculator(expression: str) -> str:
    """Calculate arithmetic using +, -, *, / and parentheses.

    Example: (12500 * 18) / 100
    """
    try:
        if len(expression) > 200:
            raise ValueError("Expression too long")
        tree = ast.parse(expression, mode="eval")
        return str(evaluate(tree))
    except (ValueError, SyntaxError, ZeroDivisionError,
            OverflowError, RecursionError) as exc:
        return f"Calculation error: {exc}"


@tool
def create_ticket(title: str, description: str) -> str:
    """Propose creation of a compliance ticket.

    Requires human approval before execution.
    """
    # Actual creation is handled by the graph's protected node.
    return (
        "Ticket request prepared: "
        f"{title}. Description: {description}"
    )
