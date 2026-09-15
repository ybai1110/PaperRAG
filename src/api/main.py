import os
from contextlib import asynccontextmanager

from datasets import load_dataset
from fastapi import FastAPI, HTTPException
from langchain_openai import ChatOpenAI
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from pathlib import Path
from dotenv import load_dotenv

from src.agent.graph import build_graph
from src.retrieval.paper_bm25 import build_paper_retriever

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

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

    title_weight = int(os.getenv("PAPER_TITLE_WEIGHT", "1"))
    paper_min_score = float(os.getenv("PAPER_MIN_SCORE", "0"))
    score_drop_value = os.getenv("PAPER_MAX_SCORE_DROP")
    max_score_drop = (
        float(score_drop_value) if score_drop_value else None
    )
    retriever = build_paper_retriever(
        dataset,
        title_weight=title_weight,
        min_score=paper_min_score,
        max_score_drop=max_score_drop,
    )

    model_mode = os.getenv("PAPERRAG_MODEL_MODE", "openai").casefold()
    if model_mode == "deterministic":
        from src.agent.deterministic_model import DeterministicEvaluationModel

        model = DeterministicEvaluationModel()
    elif model_mode == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is required unless "
                "PAPERRAG_MODEL_MODE=deterministic is explicitly set."
            )
        model = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
        )
    else:
        raise ValueError(
            "PAPERRAG_MODEL_MODE must be 'openai' or 'deterministic'."
        )

    per_paper_limit_value = os.getenv("EVIDENCE_PER_PAPER_LIMIT")
    per_paper_limit = (
        int(per_paper_limit_value) if per_paper_limit_value else None
    )

    graph = build_graph(
        retriever=retriever,
        router_model=model,
        paper_store=paper_store,
        surface_top_k=int(os.getenv("PAPER_TOP_K", "5")),
        evidence_max_papers=int(os.getenv("EVIDENCE_MAX_PAPERS", "3")),
        evidence_top_k=int(os.getenv("EVIDENCE_TOP_K", "5")),
        evidence_per_paper_limit=per_paper_limit,
        evidence_retriever_mode=os.getenv(
            "EVIDENCE_RETRIEVER_MODE", "hybrid"
        ).casefold(),
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

    selected_paper_id: str | None = None
    selected_paper_title: str | None = None
    question_context_mode: Literal[
        "standalone",
        "paper_anchored",
        "missing_context",
    ] = "standalone"

    @model_validator(mode="after")
    def validate_paper_context(self):
        has_selection = bool(
            (self.selected_paper_id or "").strip()
            or (self.selected_paper_title or "").strip()
        )
        if has_selection and self.question_context_mode == "standalone":
            self.question_context_mode = "paper_anchored"
        if self.question_context_mode == "paper_anchored" and not has_selection:
            raise ValueError(
                "paper_anchored mode requires selected_paper_id or "
                "selected_paper_title"
            )
        if self.question_context_mode == "missing_context" and has_selection:
            raise ValueError(
                "missing_context mode cannot include a selected paper"
            )
        return self


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

    initial_state = {
        "user_query": request.query.strip(),
        "question_context_mode": request.question_context_mode,
        "paper_context_source": (
            "user_selected_session"
            if request.question_context_mode == "paper_anchored"
            else "original_user_query"
        ),
    }
    if request.selected_paper_id:
        initial_state["selected_paper_id"] = request.selected_paper_id.strip()
    if request.selected_paper_title:
        initial_state["selected_paper_title"] = request.selected_paper_title.strip()

    result = graph.invoke(initial_state)

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
        "question_context_mode": result.get("question_context_mode"),
        "paper_context_source": result.get("paper_context_source"),
        "selected_paper_id": result.get("selected_paper_id"),
        "selected_paper_title": result.get("selected_paper_title"),
        "paper_context_error": result.get("paper_context_error"),
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
        "citation_scope_violations": result.get(
            "citation_scope_violations",
            [],
        ),
    }
