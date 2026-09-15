"""Preliminary within-paper BM25 versus hybrid-RRF validation ablation.

Question IDs are recovered from an existing oracle-paper graph log. The oracle
paper ID is used only to select the already-known paper corpus. Queries are the
raw user questions. Gold paragraph IDs are read only after each retrieval call
returns and are then used to calculate metrics.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import numpy as np
from datasets import load_dataset

from src.evaluation.metrics import evidence_ranking_metrics
from src.retrieval.evidence_bm25 import build_evidence_retriever
from src.retrieval.evidence_hybrid import build_hybrid_evidence_retriever


DEFAULT_SOURCE_LOG = Path(
    "results/logs/evidence_ablation_baseline_validation_42.jsonl"
)
DEFAULT_OUTPUT_METRICS = Path(
    "results/metrics/evidence_hybrid_rrf_ablation_validation_42.json"
)
DEFAULT_OUTPUT_LOG = Path(
    "results/logs/evidence_hybrid_rrf_ablation_validation_42.jsonl"
)
EXPECTED_QUESTION_COUNT = 42


def _mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _percentile(values: list[float], percentile: int) -> float:
    return float(np.percentile(values, percentile)) if values else 0.0


def load_fixed_oracle_rows(source_log: Path) -> list[dict[str, Any]]:
    """Load the exact evidence-scored IDs from one existing oracle run."""
    rows_by_id: dict[str, dict[str, Any]] = {}
    with source_log.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if (
                row.get("condition") != "oracle_paper"
                or not row.get("evidence_metric_eligible")
            ):
                continue
            question_id = str(row["question_id"])
            rows_by_id.setdefault(
                question_id,
                {
                    "question_id": question_id,
                    "question": str(row["question"]),
                    "oracle_scope_paper_id": str(row["gold_paper_id"]),
                    "gold_paragraph_ids": [
                        str(value) for value in row["gold_paragraph_ids"]
                    ],
                },
            )
    rows = list(rows_by_id.values())
    if len(rows) != EXPECTED_QUESTION_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_QUESTION_COUNT} evidence-scored oracle-paper "
            f"questions, found {len(rows)} in {source_log}."
        )
    return rows


def _evaluate_variant(
    rows: list[dict[str, Any]],
    paper_store: dict[str, dict[str, Any]],
    *,
    name: str,
    retriever_factory: Callable[[list[dict[str, Any]]], Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = []
    wall_started = time.perf_counter()
    for item in rows:
        paper = paper_store[item["oracle_scope_paper_id"]]
        build_started = time.perf_counter()
        retriever = retriever_factory([paper])
        build_ms = (time.perf_counter() - build_started) * 1000

        # The retrieval input is exactly the raw question. No title, ID,
        # answer, or gold evidence is appended or exposed to either ranker.
        retrieval_query = item["question"]
        search_started = time.perf_counter()
        results = retriever.search(retrieval_query, top_k=10)
        search_ms = (time.perf_counter() - search_started) * 1000
        retrieved_ids = [str(result["paragraph_id"]) for result in results]

        # Gold evidence enters only after ranking has completed.
        metrics = {
            **evidence_ranking_metrics(
                item["gold_paragraph_ids"], retrieved_ids, 5
            ),
            **evidence_ranking_metrics(
                item["gold_paragraph_ids"], retrieved_ids, 10
            ),
        }
        records.append(
            {
                "variant": name,
                "question_id": item["question_id"],
                "retrieval_query": retrieval_query,
                "oracle_scope_paper_id": item["oracle_scope_paper_id"],
                "retrieved_paragraph_ids": retrieved_ids,
                "gold_paragraph_ids_for_post_retrieval_scoring": item[
                    "gold_paragraph_ids"
                ],
                "build_ms": build_ms,
                "search_ms": search_ms,
                "retrieval_total_ms": build_ms + search_ms,
                **metrics,
            }
        )
    wall_ms = (time.perf_counter() - wall_started) * 1000
    totals = [record["retrieval_total_ms"] for record in records]
    summary: dict[str, Any] = {
        "questions": len(records),
        "wall_runtime_ms": wall_ms,
        "summed_retrieval_runtime_ms": sum(totals),
        "mean_retrieval_runtime_ms": _mean(totals),
        "retrieval_runtime_p50_ms": _percentile(totals, 50),
        "retrieval_runtime_p95_ms": _percentile(totals, 95),
    }
    for key in (
        "evidence_recall_at_5",
        "evidence_f1_at_5",
        "evidence_success_at_5",
        "evidence_mrr_at_10",
        "evidence_ndcg_at_10",
    ):
        summary[key] = _mean([float(record[key]) for record in records])
    return records, summary


def run_ablation(
    *,
    source_log: Path,
    output_metrics: Path,
    output_log: Path,
    candidate_pool: int,
    rrf_k: int,
    dense_dimensions: int,
    lexical_weight: float,
    dense_weight: float,
    lexical_k1: float,
    lexical_b: float,
) -> dict[str, Any]:
    rows = load_fixed_oracle_rows(source_log)
    dataset = list(load_dataset("allenai/qasper", split="validation"))
    paper_store = {str(paper["id"]): paper for paper in dataset}
    missing_papers = sorted(
        {
            row["oracle_scope_paper_id"]
            for row in rows
            if row["oracle_scope_paper_id"] not in paper_store
        }
    )
    if missing_papers:
        raise ValueError(f"Oracle-scope papers missing from validation: {missing_papers}")

    baseline_records, baseline_metrics = _evaluate_variant(
        rows,
        paper_store,
        name="unchanged_bm25",
        retriever_factory=build_evidence_retriever,
    )
    hybrid_records, hybrid_metrics = _evaluate_variant(
        rows,
        paper_store,
        name="bm25_lsa_rrf",
        retriever_factory=lambda papers: build_hybrid_evidence_retriever(
            papers,
            candidate_pool=candidate_pool,
            rrf_k=rrf_k,
            dense_dimensions=dense_dimensions,
            lexical_weight=lexical_weight,
            dense_weight=dense_weight,
            lexical_k1=lexical_k1,
            lexical_b=lexical_b,
        ),
    )
    baseline_ids = [record["question_id"] for record in baseline_records]
    hybrid_ids = [record["question_id"] for record in hybrid_records]
    if baseline_ids != hybrid_ids:
        raise AssertionError("Baseline and hybrid question IDs differ.")

    comparison_keys = (
        "evidence_recall_at_5",
        "evidence_f1_at_5",
        "evidence_success_at_5",
        "evidence_mrr_at_10",
        "evidence_ndcg_at_10",
    )
    comparison = {
        key: {
            "before": baseline_metrics[key],
            "after": hybrid_metrics[key],
            "absolute_delta": hybrid_metrics[key] - baseline_metrics[key],
        }
        for key in comparison_keys
    }
    result = {
        "metadata": {
            "description": (
                "Preliminary within-paper oracle-paper validation ablation; "
                "not open-domain performance."
            ),
            "split": "validation",
            "question_count": len(rows),
            "question_ids": baseline_ids,
            "source_oracle_log": str(source_log),
            "query_policy": (
                "Raw question only. No gold evidence, answer, paper title, or "
                "paper ID is added to the query."
            ),
            "scope_policy": (
                "The existing oracle-paper ID selects the within-paper corpus "
                "but is not a ranking feature."
            ),
            "baseline_rank_fields": (
                "Existing unchanged implementation: paper title + section + paragraph."
            ),
            "hybrid_rank_fields": "Section + paragraph only; no title or ID.",
            "gold_evidence_policy": "Used only after retrieval for metric calculation.",
            "hybrid_configuration": {
                "lexical": "BM25",
                "dense": "TF-IDF + deterministic truncated SVD (LSA)",
                "fusion": "Reciprocal Rank Fusion",
                "candidate_pool_per_branch": candidate_pool,
                "rrf_k": rrf_k,
                "dense_dimensions_cap": dense_dimensions,
                "lexical_weight": lexical_weight,
                "dense_weight": dense_weight,
                "lexical_k1": lexical_k1,
                "lexical_b": lexical_b,
            },
        },
        "unchanged_baseline": baseline_metrics,
        "hybrid_rrf": hybrid_metrics,
        "before_vs_after": comparison,
    }
    output_log.parent.mkdir(parents=True, exist_ok=True)
    with output_log.open("w", encoding="utf-8") as handle:
        for record in baseline_records + hybrid_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    output_metrics.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Per-query log: {output_log}")
    print(f"Aggregate metrics: {output_metrics}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-log", type=Path, default=DEFAULT_SOURCE_LOG)
    parser.add_argument("--output-metrics", type=Path, default=DEFAULT_OUTPUT_METRICS)
    parser.add_argument("--output-log", type=Path, default=DEFAULT_OUTPUT_LOG)
    parser.add_argument("--candidate-pool", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=10)
    parser.add_argument("--dense-dimensions", type=int, default=64)
    parser.add_argument("--lexical-weight", type=float, default=2.5)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--lexical-k1", type=float, default=2.0)
    parser.add_argument("--lexical-b", type=float, default=1.0)
    args = parser.parse_args()
    run_ablation(
        source_log=args.source_log,
        output_metrics=args.output_metrics,
        output_log=args.output_log,
        candidate_pool=args.candidate_pool,
        rrf_k=args.rrf_k,
        dense_dimensions=args.dense_dimensions,
        lexical_weight=args.lexical_weight,
        dense_weight=args.dense_weight,
        lexical_k1=args.lexical_k1,
        lexical_b=args.lexical_b,
    )


if __name__ == "__main__":
    main()
