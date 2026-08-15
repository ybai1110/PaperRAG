from src.retrieval.paper_bm25 import build_paper_retriever
from datasets import load_dataset

dataset = load_dataset(
    "allenai/qasper",
    split="train",
)

retriever = build_paper_retriever(dataset)

query = "How big are improvements of supervszed learning results trained on smalled labeled data enhanced with proposed approach copared to basic approach"

results = retriever.search(
    query = query,
    top_k=5,
)

for result in results:
    print(result["rank"])
    print(result["title"])
    print(result["score"])
    print()

