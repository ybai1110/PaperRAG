from datasets import load_dataset

from src.retrieval.paper_bm25 import build_paper_retriever


dataset = load_dataset(
    "allenai/qasper",
    split="train",
)

retriever = build_paper_retriever(dataset)

queries = [
    "multilingual RAG evaluation",
    "child language learning",
    "discourse relation detection",
]

for query in queries:
    print("\n" + "=" * 80)
    print("QUERY:", query)

    results = retriever.search(
        query=query,
        top_k=5,
    )

    for result in results:
        print(
            result["rank"],
            result["title"],
            result["score"],
        )