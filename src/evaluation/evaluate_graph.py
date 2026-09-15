"""Context-aware evaluation for PaperRAG.

The evaluator keeps raw open-domain, paper-anchored, gold-paper oracle, and
gold-evidence oracle conditions separate. Gold paper data is never added to a
raw graph input. Normal retrieval correctness is scored only for manually
reviewed questions whose final context label is sufficiently standalone.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np
import pandas as pd
from datasets import load_dataset
from langchain_openai import ChatOpenAI
from rank_bm25 import BM25Okapi

from src.agent.deterministic_model import DeterministicEvaluationModel
from src.agent.generator import make_answer_generation_node
from src.agent.graph import build_graph
from src.config import PARAGRAPHS_CSV, QUESTIONS_CSV, RANDOM_SEED, create_directories
from src.data.question_context_audit import DEFAULT_OUTPUT as CONTEXT_AUDIT_CSV
from src.data.question_context_audit import VALID_LABELS
from src.evaluation.metrics import (
    abstention_precision_recall,
    evidence_ranking_metrics,
    grounded_answer_f1,
    max_answer_f1,
    paper_mrr_at_k,
    paper_recall_at_k,
)
from src.retrieval.bm25 import parse_json_list, tokenize
from src.retrieval.paper_bm25 import build_paper_retriever


DEFAULT_METRICS_PATH = Path("results/metrics/context_aware_graph_evaluation.json")
DEFAULT_LOG_PATH = Path("results/logs/context_aware_graph_evaluation.jsonl")
STANDALONE_LABELS = {"standalone_discovery", "standalone_targeted"}


def _mean(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return mean(present) if present else None


def _percentile(values: list[float], percentile: int) -> float | None:
    return float(np.percentile(values, percentile)) if values else None


def _truthy(value: object) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes", "y"}


def _load_audit(audit_path: Path | None) -> pd.DataFrame:
    if audit_path is None or not audit_path.exists():
        return pd.DataFrame()
    audit = pd.read_csv(
        audit_path,
        dtype={"question_id": "string", "paper_id": "string"},
    ).fillna("")
    required = {
        "question_id",
        "proposed_context_label",
        "final_context_label",
        "expected_route",
        "manual_review_complete",
    }
    missing = required - set(audit.columns)
    if missing:
        raise ValueError(
            f"Context audit {audit_path} is missing columns: {sorted(missing)}"
        )
    completed = audit["manual_review_complete"].map(_truthy)
    completed_labels = set(
        audit.loc[completed, "final_context_label"].astype(str).str.strip()
    )
    invalid = sorted(completed_labels - VALID_LABELS)
    if invalid:
        raise ValueError("Invalid completed final context labels: " + ", ".join(invalid))
    if (
        audit.loc[~completed, "final_context_label"].astype(str).str.strip() != ""
    ).any():
        raise ValueError(
            "An audit row has a final label without manual_review_complete=true."
        )
    return audit


def _select_questions(
    questions: pd.DataFrame,
    *,
    split: str,
    max_questions: int,
    audit: pd.DataFrame,
) -> pd.DataFrame:
    available = questions.loc[questions["data_split"] == split].copy()
    if available.empty:
        raise ValueError(f"No processed questions found for split {split!r}.")
    if not audit.empty:
        audit_ids = audit["question_id"].astype(str).tolist()
        order = {question_id: index for index, question_id in enumerate(audit_ids)}
        selected = available.loc[
            available["question_id"].astype(str).isin(order)
        ].copy()
        selected["_audit_order"] = selected["question_id"].astype(str).map(order)
        selected = selected.sort_values("_audit_order", kind="stable").drop(
            columns="_audit_order"
        )
    else:
        selected = available.sample(
            n=min(max_questions, len(available)), random_state=RANDOM_SEED
        )
    return selected.head(max_questions).copy()


def _base_rows(frame: pd.DataFrame, audit: pd.DataFrame) -> list[dict[str, Any]]:
    audit_by_id = (
        audit.set_index(audit["question_id"].astype(str)).to_dict("index")
        if not audit.empty
        else {}
    )
    rows = []
    for _, row in frame.iterrows():
        question_id = str(row["question_id"])
        audit_row = audit_by_id.get(question_id, {})
        reviewed = _truthy(audit_row.get("manual_review_complete", False))
        final_label = (
            str(audit_row.get("final_context_label", "")).strip() if reviewed else ""
        )
        rows.append(
            {
                "question_id": question_id,
                "question": str(row["question"]).strip(),
                "gold_paper_id": str(row["gold_paper_id"]),
                "gold_paper_title": str(row["gold_paper_title"]),
                "gold_paragraph_ids": parse_json_list(row["gold_paragraph_ids_json"]),
                "gold_answers": parse_json_list(row["gold_answers_json"]),
                "gold_unanswerable": not _truthy(row["answerable"]),
                "proposed_context_label": str(
                    audit_row.get("proposed_context_label", "")
                ).strip(),
                "final_context_label": final_label,
                "manual_review_complete": reviewed,
                # A proposed route is never copied into expected_route.
                "reviewed_expected_route": (
                    str(audit_row.get("expected_route", "")).strip()
                    if reviewed
                    else None
                ),
            }
        )
    return rows


def _make_conditions(
    base_rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    raw = []
    anchored = []
    oracle_paper = []
    oracle_evidence = []
    for row in base_rows:
        standalone = row["final_context_label"] in STANDALONE_LABELS
        raw.append(
            {
                **row,
                "condition": "raw_open_domain",
                "condition_is_diagnostic": False,
                "input_context_mode": "standalone",
                "paper_context_source": "original_user_query",
                "paper_metric_eligible": standalone,
                "evidence_metric_eligible": standalone
                and bool(row["gold_paragraph_ids"]),
            }
        )
        if (
            row["manual_review_complete"]
            and row["final_context_label"] == "paper_dependent_repairable"
        ):
            anchored.append(
                {
                    **row,
                    "condition": "paper_anchored",
                    "condition_is_diagnostic": True,
                    "input_context_mode": "paper_anchored",
                    "paper_context_source": "user_selected_session",
                    "paper_metric_eligible": True,
                    "evidence_metric_eligible": bool(row["gold_paragraph_ids"]),
                    "reviewed_expected_route": "targeted_qa",
                }
            )
        oracle_paper.append(
            {
                **row,
                "condition": "oracle_paper",
                "condition_is_diagnostic": True,
                "input_context_mode": "paper_anchored",
                "paper_context_source": "diagnostic_gold_paper_oracle",
                "paper_metric_eligible": True,
                "evidence_metric_eligible": bool(row["gold_paragraph_ids"]),
                "reviewed_expected_route": "targeted_qa",
            }
        )
        oracle_evidence.append(
            {
                **row,
                "condition": "oracle_evidence",
                "condition_is_diagnostic": True,
                "input_context_mode": "paper_anchored",
                "paper_context_source": "diagnostic_gold_paper_oracle",
                "paper_metric_eligible": False,
                "evidence_metric_eligible": bool(row["gold_paragraph_ids"]),
                "reviewed_expected_route": "targeted_qa",
            }
        )
    return {
        "raw_open_domain": raw,
        "paper_anchored": anchored,
        "oracle_paper": oracle_paper,
        "oracle_evidence": oracle_evidence,
    }


def _ranking_metrics(
    gold_paragraph_ids: list[str], retrieved_paragraph_ids: list[str]
) -> dict[str, float]:
    return {
        **evidence_ranking_metrics(gold_paragraph_ids, retrieved_paragraph_ids, 5),
        **evidence_ranking_metrics(gold_paragraph_ids, retrieved_paragraph_ids, 10),
    }


def _citation_checks(
    citations: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    intended_paper_id: str,
) -> dict[str, bool | None]:
    evidence_by_id = {str(item["paragraph_id"]): item for item in evidence}
    citation_ids = [str(item.get("paragraph_id", "")) for item in citations]
    return {
        "citation_ids_valid": all(item_id in evidence_by_id for item_id in citation_ids),
        "citation_passages_match": all(
            item_id in evidence_by_id
            and citation.get("passage") == evidence_by_id[item_id].get("text")
            for item_id, citation in zip(citation_ids, citations)
        ),
        "citation_ids_unique": len(citation_ids) == len(set(citation_ids)),
        "citations_from_intended_paper": (
            all(
                str(citation.get("paper_id")) == intended_paper_id
                for citation in citations
            )
            if citations and intended_paper_id
            else None
        ),
    }


def _record_from_result(
    item: dict[str, Any],
    result: dict[str, Any],
    *,
    variant: str,
    latency_ms: float,
    graph_input: dict[str, Any],
) -> dict[str, Any]:
    evidence = result.get("evidence_candidates", [])
    citations = result.get("citations", [])
    retrieved_paper_ids = [
        str(value["paper_id"]) for value in result.get("paper_candidates", [])
    ]
    retrieved_paragraph_ids = [str(value["paragraph_id"]) for value in evidence]
    answer = str(result.get("answer") or "")
    return {
        **item,
        "system_variant": variant,
        # Persisting the exact graph input makes leakage checks mechanical.
        "graph_input": graph_input,
        "route": result.get("route"),
        "route_confidence": result.get("route_confidence"),
        "output_context_mode": result.get("question_context_mode"),
        "output_context_source": result.get("paper_context_source"),
        "retrieved_paper_ids": retrieved_paper_ids,
        "retrieved_paragraph_ids": retrieved_paragraph_ids,
        "answer": answer,
        "citations": citations,
        "clarification": result.get("clarification_question"),
        "abstained": bool(result.get("abstained", False)),
        "citation_scope_violations": result.get("citation_scope_violations", []),
        "latency_ms": latency_ms,
        "answer_f1": max_answer_f1(answer, item["gold_answers"]),
        "grounded_answer_f1": grounded_answer_f1(
            answer,
            item["gold_answers"],
            item["gold_paragraph_ids"],
            retrieved_paragraph_ids,
        ),
        **_ranking_metrics(item["gold_paragraph_ids"], retrieved_paragraph_ids),
        **_citation_checks(citations, evidence, item["gold_paper_id"]),
    }


def _run_graph_condition(
    graph, rows: list[dict[str, Any]], variant: str
) -> list[dict[str, Any]]:
    records = []
    for item in rows:
        graph_input: dict[str, Any] = {
            "user_query": item["question"],
            "question_context_mode": item["input_context_mode"],
            "paper_context_source": item["paper_context_source"],
        }
        if item["input_context_mode"] == "paper_anchored":
            graph_input.update(
                {
                    "selected_paper_id": item["gold_paper_id"],
                    "selected_paper_title": item["gold_paper_title"],
                }
            )
        started = time.perf_counter()
        result = graph.invoke(graph_input)
        records.append(
            _record_from_result(
                item,
                result,
                variant=variant,
                latency_ms=(time.perf_counter() - started) * 1000,
                graph_input=graph_input,
            )
        )
    return records


def _oracle_evidence_records(
    rows: list[dict[str, Any]],
    paragraphs: pd.DataFrame,
    model,
    variant: str,
) -> list[dict[str, Any]]:
    paragraph_by_id = {
        str(row["paragraph_id"]): row for _, row in paragraphs.iterrows()
    }
    generator = make_answer_generation_node(model)
    records = []
    for item in rows:
        evidence = []
        for paragraph_id in item["gold_paragraph_ids"]:
            paragraph = paragraph_by_id.get(str(paragraph_id))
            if paragraph is None:
                continue
            evidence.append(
                {
                    "paragraph_id": str(paragraph["paragraph_id"]),
                    "paper_id": str(paragraph["paper_id"]),
                    "title": str(paragraph["paper_title"]),
                    "section": str(paragraph["section_name"]),
                    "text": str(paragraph["paragraph_text"]),
                    "rank": len(evidence) + 1,
                    "score": None,
                }
            )
        graph_input = {
            "user_query": item["question"],
            "question_context_mode": "paper_anchored",
            "paper_context_source": "diagnostic_gold_paper_oracle",
            "selected_paper_id": item["gold_paper_id"],
            "selected_paper_title": item["gold_paper_title"],
            "evidence_candidates": evidence,
            "intended_paper_id": item["gold_paper_id"],
        }
        started = time.perf_counter()
        generated = generator(graph_input)
        result = {
            **graph_input,
            **generated,
            "route": "targeted_qa",
            "route_confidence": 1.0,
            "paper_candidates": [
                {
                    "paper_id": item["gold_paper_id"],
                    "title": item["gold_paper_title"],
                    "rank": 1,
                    "score": None,
                }
            ],
        }
        records.append(
            _record_from_result(
                item,
                result,
                variant=variant,
                latency_ms=(time.perf_counter() - started) * 1000,
                graph_input=graph_input,
            )
        )
    return records


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    paper_rows = [row for row in records if row["paper_metric_eligible"]]
    evidence_rows = [row for row in records if row["evidence_metric_eligible"]]
    conditional_rows = [
        row
        for row in evidence_rows
        if paper_recall_at_k(row["gold_paper_id"], row["retrieved_paper_ids"], 5)
    ]
    route_rows = [
        row for row in records if row.get("reviewed_expected_route") is not None
    ]
    unanswerable = [row for row in records if row["gold_unanswerable"]]
    abstention = (
        abstention_precision_recall(
            [row["gold_unanswerable"] for row in records],
            [row["abstained"] for row in records],
        )
        if unanswerable
        else {"abstention_precision": None, "abstention_recall": None}
    )
    metrics: dict[str, Any] = {
        "number_of_questions": len(records),
        "condition_is_diagnostic": (
            bool(records[0]["condition_is_diagnostic"]) if records else None
        ),
        "manual_route_labeled_count": len(route_rows),
        "route_accuracy": _mean(
            float(row["route"] == row["reviewed_expected_route"])
            for row in route_rows
        ),
        "paper_retrieval_availability": _mean(
            float(bool(row["retrieved_paper_ids"])) for row in records
        ),
        "paper_retrieval_scored_count": len(paper_rows),
        "paper_recall_at_1": _mean(
            paper_recall_at_k(row["gold_paper_id"], row["retrieved_paper_ids"], 1)
            for row in paper_rows
        ),
        "paper_recall_at_5": _mean(
            paper_recall_at_k(row["gold_paper_id"], row["retrieved_paper_ids"], 5)
            for row in paper_rows
        ),
        "paper_recall_at_10": _mean(
            paper_recall_at_k(row["gold_paper_id"], row["retrieved_paper_ids"], 10)
            for row in paper_rows
        ),
        "paper_mrr_at_10": _mean(
            paper_mrr_at_k(row["gold_paper_id"], row["retrieved_paper_ids"], 10)
            for row in paper_rows
        ),
        "evidence_scored_count": len(evidence_rows),
        "conditional_evidence_given_gold_paper_at_5_count": len(conditional_rows),
        "answer_f1": _mean(row["answer_f1"] for row in evidence_rows),
        "grounded_answer_f1": _mean(
            row["grounded_answer_f1"] for row in evidence_rows
        ),
        "citation_id_validity": _mean(
            float(row["citation_ids_valid"]) for row in records
        ),
        "citation_passage_validity": _mean(
            float(row["citation_passages_match"]) for row in records
        ),
        "citations_from_intended_paper": _mean(
            None
            if row["citations_from_intended_paper"] is None
            else float(row["citations_from_intended_paper"])
            for row in records
        ),
        "clarification_nonempty": _mean(
            float(bool(row["clarification"]))
            for row in route_rows
            if row["reviewed_expected_route"] == "needs_context"
        ),
        "avoids_unsupported_answer": _mean(
            float(not bool(row["answer"]) or row["abstained"])
            for row in route_rows
            if row["reviewed_expected_route"] == "needs_context"
        ),
        "latency_p50_ms": _percentile([row["latency_ms"] for row in records], 50),
        "latency_p95_ms": _percentile([row["latency_ms"] for row in records], 95),
        **abstention,
    }
    for k in (5, 10):
        for name in ("precision", "recall", "f1", "success", "mrr", "ndcg"):
            key = f"evidence_{name}_at_{k}"
            metrics[key] = _mean(row[key] for row in evidence_rows)
            metrics[f"conditional_{key}_given_gold_paper_at_5"] = _mean(
                row[key] for row in conditional_rows
            )
    return metrics


def _direct_bm25(
    paragraphs: pd.DataFrame,
    rows: list[dict[str, Any]],
    *,
    top_k: int,
    contextual: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    indexed_text = paragraphs["paragraph_text"].astype(str)
    variant = "original_direct_paragraph_bm25"
    if contextual:
        indexed_text = (
            paragraphs["paper_title"].astype(str)
            + " "
            + paragraphs["section_name"].astype(str)
            + " "
            + indexed_text
        )
        variant = "contextual_direct_paragraph_bm25"
    bm25 = BM25Okapi([tokenize(text) for text in indexed_text])
    records = []
    for item in rows:
        started = time.perf_counter()
        scores = bm25.get_scores(tokenize(item["question"]))
        indices = np.argsort(-scores, kind="stable")[: min(top_k, len(scores))]
        retrieved = paragraphs.iloc[indices]
        paper_ids = retrieved["paper_id"].astype(str).tolist()
        paragraph_ids = retrieved["paragraph_id"].astype(str).tolist()
        records.append(
            {
                **item,
                "system_variant": variant,
                "retrieved_paper_ids": paper_ids,
                "retrieved_paragraph_ids": paragraph_ids,
                "latency_ms": (time.perf_counter() - started) * 1000,
                **_ranking_metrics(item["gold_paragraph_ids"], paragraph_ids),
            }
        )
    paper_rows = [row for row in records if row["paper_metric_eligible"]]
    evidence_rows = [row for row in records if row["evidence_metric_eligible"]]
    metrics: dict[str, Any] = {
        "number_of_questions": len(records),
        "paper_retrieval_availability": _mean(
            float(bool(row["retrieved_paper_ids"])) for row in records
        ),
        "paper_retrieval_scored_count": len(paper_rows),
        "paper_recall_at_1": _mean(
            paper_recall_at_k(row["gold_paper_id"], row["retrieved_paper_ids"], 1)
            for row in paper_rows
        ),
        "paper_recall_at_5": _mean(
            paper_recall_at_k(row["gold_paper_id"], row["retrieved_paper_ids"], 5)
            for row in paper_rows
        ),
        "paper_recall_at_10": _mean(
            paper_recall_at_k(row["gold_paper_id"], row["retrieved_paper_ids"], 10)
            for row in paper_rows
        ),
        "paper_mrr_at_10": _mean(
            paper_mrr_at_k(row["gold_paper_id"], row["retrieved_paper_ids"], 10)
            for row in paper_rows
        ),
        "evidence_scored_count": len(evidence_rows),
        "latency_p50_ms": _percentile([row["latency_ms"] for row in records], 50),
        "latency_p95_ms": _percentile([row["latency_ms"] for row in records], 95),
    }
    for k in (5, 10):
        for name in ("precision", "recall", "f1", "success", "mrr", "ndcg"):
            key = f"evidence_{name}_at_{k}"
            metrics[key] = _mean(row[key] for row in evidence_rows)
    return records, metrics


def evaluate(
    *,
    paragraphs_path: Path,
    questions_path: Path,
    audit_path: Path | None,
    output_metrics_path: Path,
    output_log_path: Path,
    split: str,
    max_questions: int,
    direct_top_k: int,
    title_weights: list[int],
    evidence_top_k_values: list[int],
    per_paper_limits: list[int | None],
    model_mode: str,
    evidence_retriever_mode: str = "hybrid",
) -> dict[str, Any]:
    create_directories()
    paragraphs = pd.read_csv(
        paragraphs_path,
        dtype={"paper_id": "string", "paragraph_id": "string"},
    ).fillna("")
    questions = pd.read_csv(
        questions_path,
        dtype={"question_id": "string", "gold_paper_id": "string"},
    ).fillna("")
    audit = _load_audit(audit_path)
    selected = _select_questions(
        questions,
        split=split,
        max_questions=max_questions,
        audit=audit,
    )
    base_rows = _base_rows(selected, audit)
    conditions = _make_conditions(base_rows)

    all_records: list[dict[str, Any]] = []
    direct_metrics = {}
    for contextual in (False, True):
        records, metrics = _direct_bm25(
            paragraphs,
            conditions["raw_open_domain"],
            top_k=max(direct_top_k, 10),
            contextual=contextual,
        )
        all_records.extend(records)
        direct_metrics[records[0]["system_variant"] if records else str(contextual)] = metrics

    dataset = list(load_dataset("allenai/qasper", split=split))
    paper_store = {str(paper["id"]): paper for paper in dataset}
    if model_mode == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for --model-mode openai")
        model = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0
        )
    else:
        model = DeterministicEvaluationModel()

    graph_metrics: dict[str, dict[str, dict[str, Any]]] = {}
    for title_weight in title_weights:
        retriever = build_paper_retriever(dataset, title_weight=title_weight)
        for evidence_top_k in evidence_top_k_values:
            for per_paper_limit in per_paper_limits:
                limit_label = "none" if per_paper_limit is None else str(per_paper_limit)
                variant = (
                    f"two_stage_graph_title_w{title_weight}_paper_k10_"
                    f"evidence_k{evidence_top_k}_limit{limit_label}_"
                    f"retriever_{evidence_retriever_mode}_{model_mode}"
                )
                graph = build_graph(
                    retriever=retriever,
                    router_model=model,
                    paper_store=paper_store,
                    surface_top_k=10,
                    evidence_top_k=evidence_top_k,
                    evidence_per_paper_limit=per_paper_limit,
                    evidence_retriever_mode=evidence_retriever_mode,
                )
                graph_metrics[variant] = {}
                for condition in ("raw_open_domain", "paper_anchored", "oracle_paper"):
                    records = _run_graph_condition(graph, conditions[condition], variant)
                    all_records.extend(records)
                    graph_metrics[variant][condition] = _summarize(records)
                oracle_records = _oracle_evidence_records(
                    conditions["oracle_evidence"], paragraphs, model, variant
                )
                all_records.extend(oracle_records)
                graph_metrics[variant]["oracle_evidence"] = _summarize(oracle_records)

    raw_metric_candidates = [
        (variant, values["raw_open_domain"])
        for variant, values in graph_metrics.items()
        if values["raw_open_domain"]["evidence_f1_at_10"] is not None
    ]
    selected_variant = (
        max(raw_metric_candidates, key=lambda item: item[1]["evidence_f1_at_10"])[0]
        if raw_metric_candidates
        else None
    )
    pending_count = sum(not row["manual_review_complete"] for row in base_rows)
    result = {
        "metadata": {
            "random_seed": RANDOM_SEED,
            "split": split,
            "max_questions": max_questions,
            "selected_question_ids": [row["question_id"] for row in base_rows],
            "context_audit_path": str(audit_path) if audit_path else None,
            "manual_review_pending_count": pending_count,
            "model_mode": model_mode,
            "evidence_retriever_mode": evidence_retriever_mode,
            "raw_leakage_policy": (
                "raw_open_domain graph_input contains only the original question, "
                "standalone mode, and original_user_query source; gold title/ID are "
                "retained only outside graph_input for post-hoc scoring."
            ),
            "normal_scoring_policy": (
                "Raw paper/evidence correctness is calculated only for manually "
                "reviewed standalone_discovery or standalone_targeted questions."
            ),
            "conditional_evidence_definition": (
                "Evidence metrics restricted to scored questions where the gold "
                "paper occurs in surface-search top 5."
            ),
            "semantic_citation_support": (
                "Not measured. Structural ID/passage checks do not establish that "
                "a cited passage semantically supports the generated claim."
            ),
            "condition_definitions": {
                "raw_open_domain": "No paper ID/title supplied to the graph; headline condition.",
                "paper_anchored": (
                    "Reviewed paper-dependent questions with simulated user-selected "
                    "session context; diagnostic and never open-domain."
                ),
                "oracle_paper": (
                    "Gold paper supplied as selected context for every question; diagnostic."
                ),
                "oracle_evidence": (
                    "Gold paragraphs supplied directly to answer generation; diagnostic."
                ),
            },
        },
        "headline_open_domain": {
            "selected_variant": selected_variant,
            "metrics": (
                graph_metrics[selected_variant]["raw_open_domain"]
                if selected_variant
                else None
            ),
            "status": (
                "scored"
                if selected_variant
                else "pending manual context review; no proposed labels were used as gold"
            ),
        },
        "direct_retrieval": direct_metrics,
        "graph_conditions_separate": graph_metrics,
    }
    output_log_path.parent.mkdir(parents=True, exist_ok=True)
    with output_log_path.open("w", encoding="utf-8") as handle:
        for record in all_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    output_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    output_metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Per-query log: {output_log_path}")
    print(f"Aggregate metrics: {output_metrics_path}")
    return result


def _parse_limits(values: list[str]) -> list[int | None]:
    return [None if value.casefold() == "none" else int(value) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paragraphs", type=Path, default=PARAGRAPHS_CSV)
    parser.add_argument("--questions", type=Path, default=QUESTIONS_CSV)
    parser.add_argument("--audit", type=Path, default=CONTEXT_AUDIT_CSV)
    parser.add_argument("--output-metrics", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--output-log", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--split", default="validation", choices=["validation", "train"])
    parser.add_argument("--max-questions", type=int, default=50)
    parser.add_argument("--direct-top-k", type=int, default=10)
    parser.add_argument("--title-weights", nargs="+", type=int, default=[1])
    parser.add_argument("--evidence-top-k-values", nargs="+", type=int, default=[10])
    parser.add_argument("--per-paper-limits", nargs="+", default=["none"])
    parser.add_argument(
        "--model-mode", choices=["deterministic", "openai"], default="deterministic"
    )
    parser.add_argument(
        "--evidence-retriever-mode",
        choices=["bm25", "hybrid"],
        default="hybrid",
    )
    args = parser.parse_args()
    evaluate(
        paragraphs_path=args.paragraphs,
        questions_path=args.questions,
        audit_path=args.audit,
        output_metrics_path=args.output_metrics,
        output_log_path=args.output_log,
        split=args.split,
        max_questions=args.max_questions,
        direct_top_k=args.direct_top_k,
        title_weights=args.title_weights,
        evidence_top_k_values=args.evidence_top_k_values,
        per_paper_limits=_parse_limits(args.per_paper_limits),
        model_mode=args.model_mode,
        evidence_retriever_mode=args.evidence_retriever_mode,
    )


if __name__ == "__main__":
    main()
