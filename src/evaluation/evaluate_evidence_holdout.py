"""Evaluate frozen evidence retrievers on untouched validation-audit IDs.

The 42 evidence-scored oracle-paper questions used for configuration selection
are development data. This evaluator subtracts them from the fixed validation
audit before identifying eligible held-out questions. It does not expose any
hybrid parameter overrides.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset

from src.config import QUESTIONS_CSV
from src.data.question_context_audit import DEFAULT_OUTPUT as VALIDATION_AUDIT
from src.evaluation.evaluate_evidence_ablation import (
    DEFAULT_SOURCE_LOG,
    _evaluate_variant,
    load_fixed_oracle_rows,
)
from src.retrieval.bm25 import parse_json_list
from src.retrieval.evidence_bm25 import build_evidence_retriever
from src.retrieval.evidence_hybrid import (
    FROZEN_HYBRID_CONFIG,
    build_frozen_hybrid_evidence_retriever,
)


DEFAULT_OUTPUT_METRICS = Path(
    "results/metrics/evidence_holdout_validation_bm25_vs_frozen_hybrid.json"
)
DEFAULT_OUTPUT_LOG = Path(
    "results/logs/evidence_holdout_validation_bm25_vs_frozen_hybrid.jsonl"
)


def _question_id_fingerprint(question_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(question_ids).encode()).hexdigest()


def _load_heldout_rows(
    *,
    audit_path: Path,
    questions_path: Path,
    development_log: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    audit = pd.read_csv(audit_path, dtype={"question_id": "string"}).fillna("")
    audit_ids = audit["question_id"].astype(str).tolist()
    if len(audit_ids) != len(set(audit_ids)):
        raise ValueError("Validation audit contains duplicate question IDs.")

    development_rows = load_fixed_oracle_rows(development_log)
    development_ids = [row["question_id"] for row in development_rows]
    unexpected_development_ids = sorted(set(development_ids) - set(audit_ids))
    if unexpected_development_ids:
        raise ValueError(
            "Development IDs are absent from the fixed validation audit: "
            f"{unexpected_development_ids}"
        )
    heldout_audit_ids = [
        question_id for question_id in audit_ids if question_id not in development_ids
    ]
    heldout_order = {
        question_id: index for index, question_id in enumerate(heldout_audit_ids)
    }

    questions = pd.read_csv(
        questions_path,
        dtype={"question_id": "string", "gold_paper_id": "string"},
    ).fillna("")
    heldout = questions.loc[
        (questions["data_split"] == "validation")
        & questions["question_id"].astype(str).isin(heldout_order)
    ].copy()
    heldout["_heldout_order"] = (
        heldout["question_id"].astype(str).map(heldout_order)
    )
    heldout = heldout.sort_values("_heldout_order", kind="stable")
    found_ids = heldout["question_id"].astype(str).tolist()
    missing_ids = [question_id for question_id in heldout_audit_ids if question_id not in found_ids]
    if missing_ids:
        raise ValueError(f"Held-out audit IDs missing from validation questions: {missing_ids}")

    rows = []
    for _, row in heldout.iterrows():
        gold_paragraph_ids = parse_json_list(row["gold_paragraph_ids_json"])
        if not gold_paragraph_ids:
            continue
        rows.append(
            {
                "question_id": str(row["question_id"]),
                "question": str(row["question"]),
                "oracle_scope_paper_id": str(row["gold_paper_id"]),
                "gold_paragraph_ids": [
                    str(paragraph_id) for paragraph_id in gold_paragraph_ids
                ],
            }
        )
    metadata = {
        "fixed_validation_audit_count": len(audit_ids),
        "development_evidence_scored_count": len(development_ids),
        "remaining_heldout_audit_count": len(heldout_audit_ids),
        "eligible_heldout_evidence_scored_count": len(rows),
        "ineligible_heldout_count": len(heldout_audit_ids) - len(rows),
        "development_question_ids_sha256": _question_id_fingerprint(
            development_ids
        ),
        "heldout_question_ids_sha256": _question_id_fingerprint(
            [row["question_id"] for row in rows]
        ),
        "heldout_question_ids": [row["question_id"] for row in rows],
    }
    return rows, metadata


def _add_success_count(metrics: dict[str, Any], records: list[dict[str, Any]]) -> None:
    metrics["evidence_success_at_5_count"] = int(
        sum(record["evidence_success_at_5"] for record in records)
    )
    metrics["evidence_success_at_5_denominator"] = len(records)


def evaluate_holdout(
    *,
    audit_path: Path,
    questions_path: Path,
    development_log: Path,
    output_metrics: Path,
    output_log: Path,
) -> dict[str, Any]:
    rows, sample_metadata = _load_heldout_rows(
        audit_path=audit_path,
        questions_path=questions_path,
        development_log=development_log,
    )
    if not rows:
        raise ValueError("No eligible held-out evidence-scored questions remain.")

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
        raise ValueError(f"Held-out oracle-scope papers are missing: {missing_papers}")

    baseline_records, baseline_metrics = _evaluate_variant(
        rows,
        paper_store,
        name="graph_retriever_mode_bm25",
        retriever_factory=build_evidence_retriever,
    )
    hybrid_records, hybrid_metrics = _evaluate_variant(
        rows,
        paper_store,
        name="graph_retriever_mode_hybrid_frozen",
        retriever_factory=build_frozen_hybrid_evidence_retriever,
    )
    if [record["question_id"] for record in baseline_records] != [
        record["question_id"] for record in hybrid_records
    ]:
        raise AssertionError("Baseline and hybrid held-out IDs differ.")
    _add_success_count(baseline_metrics, baseline_records)
    _add_success_count(hybrid_metrics, hybrid_records)

    metric_keys = (
        "evidence_recall_at_5",
        "evidence_f1_at_5",
        "evidence_success_at_5",
        "evidence_mrr_at_10",
        "evidence_ndcg_at_10",
        "mean_retrieval_runtime_ms",
        "retrieval_runtime_p95_ms",
    )
    result = {
        "metadata": {
            "description": (
                "Untouched held-out within-paper validation comparison; "
                "not open-domain performance."
            ),
            "split": "validation",
            "test_split_used": False,
            "parameters_frozen_before_holdout_run": True,
            "development_log": str(development_log),
            "validation_audit": str(audit_path),
            "query_policy": (
                "Original question only; no gold evidence, answer, title, or ID "
                "is appended to the query."
            ),
            "scope_policy": (
                "Existing oracle-paper ID selects the within-paper corpus only."
            ),
            "baseline_rank_fields": (
                "Original graph BM25 implementation: paper title + section + "
                "paragraph. The title is ordinary corpus content, not injected "
                "from an evaluation label."
            ),
            "hybrid_rank_fields": "Section + paragraph only; no title or ID.",
            "gold_evidence_policy": (
                "Gold paragraph IDs are used only after retrieval returns."
            ),
            "frozen_hybrid_configuration": {
                "candidate_pool": FROZEN_HYBRID_CONFIG.candidate_pool,
                "dense_dimensions": FROZEN_HYBRID_CONFIG.dense_dimensions,
                "rrf_k": FROZEN_HYBRID_CONFIG.rrf_k,
                "lexical_weight": FROZEN_HYBRID_CONFIG.lexical_weight,
                "dense_weight": FROZEN_HYBRID_CONFIG.dense_weight,
                "lexical_k1": FROZEN_HYBRID_CONFIG.lexical_k1,
                "lexical_b": FROZEN_HYBRID_CONFIG.lexical_b,
            },
            **sample_metadata,
        },
        "original_bm25": baseline_metrics,
        "frozen_hybrid": hybrid_metrics,
        "before_vs_after": {
            key: {
                "before": baseline_metrics[key],
                "after": hybrid_metrics[key],
                "absolute_delta": hybrid_metrics[key] - baseline_metrics[key],
            }
            for key in metric_keys
        },
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
    parser.add_argument("--audit", type=Path, default=VALIDATION_AUDIT)
    parser.add_argument("--questions", type=Path, default=QUESTIONS_CSV)
    parser.add_argument("--development-log", type=Path, default=DEFAULT_SOURCE_LOG)
    parser.add_argument("--output-metrics", type=Path, default=DEFAULT_OUTPUT_METRICS)
    parser.add_argument("--output-log", type=Path, default=DEFAULT_OUTPUT_LOG)
    args = parser.parse_args()
    evaluate_holdout(
        audit_path=args.audit,
        questions_path=args.questions,
        development_log=args.development_log,
        output_metrics=args.output_metrics,
        output_log=args.output_log,
    )


if __name__ == "__main__":
    main()
