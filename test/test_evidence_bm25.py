from datasets import load_dataset

from src.retrieval.evidence_bm25 import (
    build_evidence_retriever,
)


dataset = load_dataset(
    "allenai/qasper",
    split="train",
)

selected_papers = [
    dataset[0],
    dataset[1],
    dataset[2],
]

retriever = build_evidence_retriever(
    selected_papers
)

results = retriever.search(
    query="What are the results?",
    top_k=5,
)

for result in results:
    print("\nRank:", result["rank"])
    print("Paper:", result["title"])
    print("Section:", result["section"])
    print("Paragraph:", result["text"])
    print("Score:", result["score"])

assert len(results) > 0
assert "paragraph_id" in results[0]
assert "paper_id" in results[0]
assert "text" in results[0]