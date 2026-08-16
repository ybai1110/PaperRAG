import os

from datasets import load_dataset
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv() 

from src.retrieval.paper_bm25 import (
    build_paper_retriever,
)

from src.agent.graph import build_graph


dataset = load_dataset(
    "allenai/qasper",
    split="train",
)

retriever = build_paper_retriever(
    dataset
)

router_model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

graph = build_graph(
    retriever=retriever,
    router_model=router_model,
)

queries = [
    "multilingual RAG evaluation",

    "How does CamemBERT compare with multilingual BERT?",

    "What dataset did they use?",
]


for query in queries:

    print("\n" + "=" * 80)
    print("QUERY:", query)

    result = graph.invoke(
        {
            "user_query": query
        }
    )

    print("ROUTE:", result["route"])

    print(
        "CONFIDENCE:",
        result["route_confidence"],
    )

    print(
        "REASON:",
        result["route_reason"],
    )
