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


paper_store = {
    str(paper["id"]): paper
    for paper in dataset
}

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
    paper_store=paper_store,
)

# Test queries
