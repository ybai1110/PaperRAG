from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel
from datasets import load_dataset

from src.retrieval.paper_bm25 import build_paper_retriever


retriever = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever

    dataset = load_dataset(
        "allenai/qasper",
        split="train",
    )

    retriever = build_paper_retriever(dataset)
    yield


app = FastAPI(lifespan=lifespan)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


@app.post("/search")
def search(request: SearchRequest):
    results = retriever.search(
        query=request.query,
        top_k=request.top_k,
    )

    return {
        "query": request.query,
        "route": "discovery",
        "results": [
            {
                "paper_id": result["paper_id"],
                "title": result["title"],
                "abstract": result["abstract"],
                "score": result["score"],
            }
            for result in results
        ],
    }