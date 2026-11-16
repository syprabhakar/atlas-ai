
from atlas.core import AtlasRAG

rag = AtlasRAG()

questions = [
    "What are the three WCAG conformance levels?",
    "What are the four WCAG POUR principles?",
]

for question in questions:
    print("\n" + "=" * 70)
    print("QUESTION:", question)

    chunks = rag.search(question, top_k=5)

    for rank, chunk in enumerate(chunks, start=1):
        print(f"\nRANK: {rank}")
        print("SOURCE:", chunk.get("filename"))
        print("PAGE:", chunk.get("page"))
        print("RRF SCORE:", chunk.get("rrf_score"))
        print("TEXT:")
        print(chunk["text"][:500])
        print("-" * 70)
