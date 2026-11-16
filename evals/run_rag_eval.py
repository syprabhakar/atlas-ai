
import json
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

from atlas.core import AtlasRAG

load_dotenv()

client = OpenAI()
rag = AtlasRAG()

CASES = [
    {
        "question": "What are the three WCAG conformance levels?",
        "reference": "The three levels are A, AA, and AAA.",
    },
    {
        "question": "What is the purpose of WCAG?",
        "reference": (
            "WCAG provides guidelines for making web "
            "content accessible to people with disabilities."
        ),
    },
]


def evaluate(question, answer, contexts, reference):
    prompt = f"""
You are evaluating a retrieval-augmented generation system.

QUESTION:
{question}

REFERENCE ANSWER:
{reference}

RETRIEVED CONTEXT:
{contexts}

GENERATED ANSWER:
{answer}

Evaluate these two metrics:

1. faithfulness:
   How well are the generated answer's factual claims
   supported by the retrieved context?

2. answer_relevancy:
   How well does the generated answer address the question?

Return JSON with:
- faithfulness: number between 0 and 1
- answer_relevancy: number between 0 and 1
- explanation: short explanation

Do not use outside knowledge to judge faithfulness.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    return json.loads(response.choices[0].message.content)


def main():
    results = []

    for case in CASES:
        question = case["question"]

        # Retrieve context using existing hybrid search
        retrieved = rag.search(question, top_k=5)

        contexts = "\n\n".join(
            chunk["text"] for chunk in retrieved
        )

        # Generate answer using existing RAG pipeline
        response = rag.answer(question)

        answer = response["answer"]

        scores = evaluate(
            question,
            answer,
            contexts,
            case["reference"],
        )

        results.append(
            {
                "question": question,
                "answer": answer,
                "scores": scores,
            }
        )

        print("\nQuestion:", question)
        print("Answer:", answer)
        print("Faithfulness:", scores["faithfulness"])
        print("Relevancy:", scores["answer_relevancy"])
        print("Explanation:", scores["explanation"])

    output = Path("evals/rag_results.json")
    output.write_text(json.dumps(results, indent=2))

    print("\nResults saved to", output)


if __name__ == "__main__":
    main()
