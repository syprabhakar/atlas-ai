
import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from atlas.core import AtlasRAG

load_dotenv()

client = OpenAI()
rag = AtlasRAG()

CASES = [
    {
        "question": "What are the three WCAG conformance levels?",
        "required_facts": [
            "Level A",
            "Level AA",
            "Level AAA",
        ],
    },
    {
        "question": "What are the four WCAG POUR principles?",
        "required_facts": [
            "Perceivable",
            "Operable",
            "Understandable",
            "Robust",
        ],
    },
]


def judge(question, chunks, required_facts):
    payload = {
        "question": question,
        "chunks": [
            {"id": c["id"], "text": c["text"]}
            for c in chunks
        ],
        "required_facts": required_facts,
    }

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Evaluate retrieved document evidence. "
                    "Use only the supplied chunks. "
                    "For each chunk, decide whether it contains "
                    "information useful for answering the question. "
                    "For each required fact, decide whether it is "
                    "supported by at least one chunk. "
                    "Return JSON with keys chunk_relevance "
                    "(array of booleans in chunk order) and "
                    "fact_coverage (array of booleans in fact order). "
                    "Do not use outside knowledge."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(payload),
            },
        ],
    )

    return json.loads(response.choices[0].message.content)


def main():
    results = []

    for case in CASES:
        question = case["question"]
        chunks = rag.search(question, top_k=5)

        if not chunks:
            print(f"\n{question}: No chunks retrieved")
            continue

        judgment = judge(
            question,
            chunks,
            case["required_facts"],
        )

        relevance = judgment["chunk_relevance"]
        coverage = judgment["fact_coverage"]

        if len(relevance) != len(chunks):
            raise ValueError("Invalid chunk relevance count")

        if len(coverage) != len(case["required_facts"]):
            raise ValueError("Invalid fact coverage count")

        if any(type(v) is not bool for v in relevance + coverage):
            raise ValueError("Judge must return boolean values")

        precision = sum(relevance) / len(chunks)
        context_coverage = sum(coverage) / len(coverage)

        result = {
            "question": question,
            "precision_at_k": precision,
            "context_coverage": context_coverage,
            "retrieved_count": len(chunks),
            "chunk_relevance": relevance,
            "fact_coverage": coverage,
        }

        results.append(result)

        print("\nQuestion:", question)
        print("Precision@K:", round(precision, 3))
        print("Context coverage:", round(context_coverage, 3))
        print("Relevant chunks:", sum(relevance), "/", len(chunks))
        print("Covered facts:", sum(coverage), "/", len(coverage))

    output = Path("evals/retrieval_results.json")
    output.write_text(json.dumps(results, indent=2))
    print("\nResults saved to", output)


if __name__ == "__main__":
    main()
