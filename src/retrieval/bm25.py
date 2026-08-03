"""
Phase 5: BM25 paragraph retrieval baseline.

This script:
1. Loads processed paragraphs and questions.
2. Optionally keeps only manually audited standalone questions.
3. Retrieves top-k paragraphs using BM25.
4. Saves a per-query JSONL log.
5. Reports paper recall, evidence F1, and latency.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from statistics import mean, median
from typing import Any

import pandas as pd
from rank_bm25 import BM25Okapi

from src.config import (
    BM25_LOG_JSONL,
    BM25_METRICS_JSON,
    DEFAULT_TOP_K,
    PARAGRAPHS_CSV,
    QUESTION_AUDIT_CSV,
    QUESTIONS_CSV,
    create_directories,
)
from src.evaluation.metrics import (
    evidence_precision_recall_f1,
    paper_recall_at_k,
)
from src.evaluation.query_log import QueryLogRecord, write_jsonl


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")


def tokenize(text: str) -> list[str]:
    """Simple lowercase tokenizer suitable for a transparent BM25 baseline."""
    return TOKEN_PATTERN.findall(str(text).lower())


def parse_json_list(value: Any) -> list[Any]:
    """Safely parse a JSON-list column from CSV."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return value
    parsed = json.loads(str(value))
    return parsed if isinstance(parsed, list) else [parsed]


def load_standalone_ids(audit_path: Path) -> set[str]:
    if not audit_path.exists():
        raise FileNotFoundError(
            f"Audit file not found: {audit_path}"
        )

    audit = pd.read_csv(audit_path).fillna("")

    final_rows = audit.loc[
        (
            audit["question_validity"]
            .str.strip()
            .str.lower()
            .eq("standalone")
        )
        &
        (
            audit["review_status"]
            .str.strip()
            .str.lower()
            .isin(["completed", "finished"])
        ),
        "question_id",
    ]

    return set(final_rows.astype(str))


def prepare_questions(
    questions: pd.DataFrame,
    split: str | None,
    audit_path: Path | None,
    max_questions: int | None,
) -> pd.DataFrame:
    """Apply the evaluation filters used by the BM25 experiment."""
    filtered = questions.copy()

    if split is not None:
        filtered = filtered.loc[filtered["data_split"] == split]

    # Paragraph-level BM25 cannot retrieve figure/table-only evidence.
    filtered = filtered.loc[filtered["has_text_evidence"] == True]  # noqa: E712

    if audit_path is not None:
        standalone_ids = load_standalone_ids(audit_path)
        filtered = filtered.loc[
            filtered["question_id"].astype(str).isin(standalone_ids)
        ]

    if max_questions is not None:
        filtered = filtered.head(max_questions)

    if filtered.empty:
        raise ValueError("No questions remain after filtering.")

    return filtered.reset_index(drop=True)


def run_bm25(
    paragraphs_path: Path = PARAGRAPHS_CSV,
    questions_path: Path = QUESTIONS_CSV,
    output_log_path: Path = BM25_LOG_JSONL,
    output_metrics_path: Path = BM25_METRICS_JSON,
    top_k: int = DEFAULT_TOP_K,
    split: str | None = None,
    audit_path: Path | None = None,
    max_questions: int | None = None,
) -> dict[str, Any]:
    """Run BM25 retrieval and save query-level logs and aggregate metrics."""
    create_directories()

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    paragraphs = pd.read_csv(paragraphs_path).fillna("")
    questions = pd.read_csv(questions_path)
    questions = prepare_questions(
        questions,
        split=split,
        audit_path=audit_path,
        max_questions=max_questions,
    )

    corpus_tokens = [
        tokenize(text)
        for text in paragraphs["paragraph_text"].astype(str)
    ]
    bm25 = BM25Okapi(corpus_tokens)

    records: list[QueryLogRecord] = []
    recall_at_1: list[float] = []
    recall_at_5: list[float] = []
    recall_at_10: list[float] = []
    evidence_f1_at_k: list[float] = []
    latencies: list[float] = []

    effective_k = min(top_k, len(paragraphs))

    for index, question_row in questions.iterrows():
        question = str(question_row["question"])
        gold_paper_id = str(question_row["gold_paper_id"])
        gold_paragraph_ids = parse_json_list(
            question_row["gold_paragraph_ids_json"]
        )
        gold_answers = parse_json_list(question_row["gold_answers_json"])

        start = time.perf_counter()
        scores = bm25.get_scores(tokenize(question))
        top_indices = scores.argsort()[::-1][:effective_k]
        latency_ms = (time.perf_counter() - start) * 1000

        retrieved = paragraphs.iloc[top_indices]
        retrieved_paragraph_ids = (
            retrieved["paragraph_id"].astype(str).tolist()
        )
        retrieved_paper_ids = retrieved["paper_id"].astype(str).tolist()
        retrieved_scores = [float(scores[i]) for i in top_indices]

        recall_at_1.append(
            paper_recall_at_k(gold_paper_id, retrieved_paper_ids, 1)
        )
        recall_at_5.append(
            paper_recall_at_k(gold_paper_id, retrieved_paper_ids, min(5, top_k))
        )
        recall_at_10.append(
            paper_recall_at_k(
                gold_paper_id,
                retrieved_paper_ids,
                min(10, top_k),
            )
        )
        evidence_metrics = evidence_precision_recall_f1(
            gold_paragraph_ids,
            retrieved_paragraph_ids,
        )
        evidence_f1_at_k.append(evidence_metrics["evidence_f1"])
        latencies.append(latency_ms)

        records.append(
            QueryLogRecord(
                question_id=str(question_row["question_id"]),
                system_variant=f"bm25_top_{top_k}",
                question=question,
                gold_paper_id=gold_paper_id,
                retrieved_paper_ids=retrieved_paper_ids,
                gold_paragraph_ids=gold_paragraph_ids,
                retrieved_paragraph_ids=retrieved_paragraph_ids,
                retrieval_scores=retrieved_scores,
                gold_answers=gold_answers,
                retrieval_latency_ms=latency_ms,
                metadata={
                    "data_split": str(question_row["data_split"]),
                },
            )
        )

        if (index + 1) % 50 == 0:
            print(f"Retrieved {index + 1}/{len(questions)} questions")

    write_jsonl(records, output_log_path)

    metrics = {
        "system_variant": f"bm25_top_{top_k}",
        "number_of_questions": len(records),
        "paper_recall_at_1": mean(recall_at_1),
        "paper_recall_at_5": mean(recall_at_5),
        "paper_recall_at_10": mean(recall_at_10),
        f"evidence_f1_at_{top_k}": mean(evidence_f1_at_k),
        "mean_retrieval_latency_ms": mean(latencies),
        "median_retrieval_latency_ms": median(latencies),
    }

    output_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with output_metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Query log: {output_log_path}")
    print(f"Metrics:   {output_metrics_path}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the BM25 baseline.")
    parser.add_argument("--paragraphs", type=Path, default=PARAGRAPHS_CSV)
    parser.add_argument("--questions", type=Path, default=QUESTIONS_CSV)
    parser.add_argument("--output-log", type=Path, default=BM25_LOG_JSONL)
    parser.add_argument(
        "--output-metrics",
        type=Path,
        default=BM25_METRICS_JSON,
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument(
        "--audit",
        type=Path,
        default=None,
        help=(
            "Optional completed Phase 2 audit CSV. "
            "Only standalone questions will be evaluated."
        ),
    )
    parser.add_argument("--max-questions", type=int, default=None)
    args = parser.parse_args()

    run_bm25(
        paragraphs_path=args.paragraphs,
        questions_path=args.questions,
        output_log_path=args.output_log,
        output_metrics_path=args.output_metrics,
        top_k=args.top_k,
        split=args.split,
        audit_path=args.audit,
        max_questions=args.max_questions,
    )


if __name__ == "__main__":
    main()
