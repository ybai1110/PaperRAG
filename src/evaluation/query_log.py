"""Phase 4: Standard JSONL logging for every question and system variant."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass
class QueryLogRecord:
    question_id: str
    system_variant: str
    question: str

    gold_paper_id: str
    retrieved_paper_ids: list[str] = field(default_factory=list)

    gold_paragraph_ids: list[str] = field(default_factory=list)
    retrieved_paragraph_ids: list[str] = field(default_factory=list)
    retrieval_scores: list[float] = field(default_factory=list)

    gold_answers: list[str] = field(default_factory=list)
    generated_answer: str = ""
    abstained: bool = False

    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0

    metadata: dict[str, Any] = field(default_factory=dict)


def append_jsonl(record: QueryLogRecord, output_path: Path) -> None:
    """Append one query record to a JSON Lines file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as file:
        json.dump(asdict(record), file, ensure_ascii=False)
        file.write("\n")


def write_jsonl(
    records: list[QueryLogRecord],
    output_path: Path,
) -> None:
    """Overwrite a JSON Lines file with a list of records."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            json.dump(asdict(record), file, ensure_ascii=False)
            file.write("\n")


def read_jsonl(input_path: Path) -> Iterator[dict[str, Any]]:
    """Yield records from a JSON Lines file."""
    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {input_path}"
                ) from exc
