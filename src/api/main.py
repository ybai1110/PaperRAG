import os
from contextlib import asynccontextmanager

from datasets import load_dataset
from fastapi import FastAPI, HTTPException
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from pathlib import Path
from dotenv import load_dotenv

from src.agent.graph import build_graph
from src.retrieval.paper_bm25 import build_paper_retriever

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

graph = None

graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph

    dataset = load_dataset(
        "allenai/qasper",
        split="train",
    )

    paper_store = {
        str(paper["id"]): paper
        for paper in dataset
    }

    retriever = build_paper_retriever(dataset)

    model_name = os.environ["OPENAI_MODEL"]

    model = ChatOpenAI(
        model=model_name,
        temperature=0,
    )

    graph = build_graph(
        retriever=retriever,
        router_model=model,
        paper_store=paper_store,
    )

    yield

    graph = None


app = FastAPI(
    title="PaperRAG API",
    version="1.0.0",
    lifespan=lifespan,
)


class SearchRequest(BaseModel):
    query: str = Field(
        min_length=2,
        max_length=1000,
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "graph_ready": graph is not None,
    }


@app.post("/search")
def search(request: SearchRequest):
    if graph is None:
        raise HTTPException(
            status_code=503,
            detail="PaperRAG is still loading.",
        )

    result = graph.invoke(
        {
            "user_query": request.query.strip(),
        }
    )

    papers = [
        {
            "paper_id": paper["paper_id"],
            "rank": paper.get("rank"),
            "title": paper["title"],
            "abstract": paper.get("abstract", ""),
            "score": paper.get("score", 0.0),
        }
        for paper in result.get(
            "paper_candidates",
            [],
        )
    ]

    return {
        "query": request.query.strip(),
        "route": result.get("route"),
        "route_confidence": result.get(
            "route_confidence"
        ),
        "route_reason": result.get("route_reason"),
        "results": papers,
        "evidence": result.get(
            "evidence_candidates",
            [],
        ),
        "answer": result.get("answer"),
        "citations": result.get("citations", []),
        "clarification": result.get(
            "clarification_question"
        ),
        "abstained": result.get(
            "abstained",
            False,
        ),
        "abstention_reason": result.get(
            "abstention_reason",
            "",
        ),
    }