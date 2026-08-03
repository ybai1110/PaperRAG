from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from datasets import DatasetDict, load_dataset

from src.config import PARAGRAPHS_CSV, QUESTIONS_CSV, create_directories


def normalize_text(text: Any) -> str:
    """Normalize whitespace so evidence text can be matched to paragraphs."""
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def stable_short_hash(text: str) -> str:
    """Create a short stable ID for unmatched evidence such as figures/tables."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def ensure_list(value: Any) -> list[Any]:
    """Return a list while safely handling None, tuples, and scalar values."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def get_nested_list(container: dict[str, Any], key: str) -> list[Any]:
    value = container.get(key, [])
    return ensure_list(value)


def answer_to_text(answer: dict[str, Any]) -> str:
    """Convert one QASPER answer annotation into a readable answer string."""
    if not isinstance(answer, dict):
        return ""

    if bool(answer.get("unanswerable", False)):
        return "UNANSWERABLE"

    extractive = [
        normalize_text(x)
        for x in ensure_list(answer.get("extractive_spans"))
        if normalize_text(x)
    ]
    if extractive:
        return " ".join(extractive)

    free_form = normalize_text(answer.get("free_form_answer"))
    if free_form:
        return free_form

    # In some dataset versions, yes_no may be None when it is not the answer type.
    yes_no = answer.get("yes_no")
    if isinstance(yes_no, bool):
        return "Yes" if yes_no else "No"

    return ""


def flatten_answers(raw_answers: Any) -> list[dict[str, Any]]:
    """
    Normalize the possible Hugging Face representations of QASPER answers.

    A question usually stores:
    {
        "annotation_id": [...],
        "answer": [{...}, {...}],
        "worker_id": [...]
    }
    """
    if raw_answers is None:
        return []

    if isinstance(raw_answers, dict):
        answers = raw_answers.get("answer", [])
        return [a for a in ensure_list(answers) if isinstance(a, dict)]

    if isinstance(raw_answers, list):
        flattened: list[dict[str, Any]] = []
        for item in raw_answers:
            if isinstance(item, dict) and "answer" in item:
                flattened.extend(
                    a for a in ensure_list(item.get("answer")) if isinstance(a, dict)
                )
            elif isinstance(item, dict):
                flattened.append(item)
        return flattened

    return []


def build_paragraph_rows(
    row: dict[str, Any],
    split: str,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Flatten one paper into paragraph rows and return a text-to-ID map."""
    paper_id = normalize_text(row.get("id"))
    title = normalize_text(row.get("title"))
    abstract = normalize_text(row.get("abstract"))

    full_text = row.get("full_text") or {}
    section_names = ensure_list(full_text.get("section_name"))
    section_paragraphs = ensure_list(full_text.get("paragraphs"))

    paragraph_rows: list[dict[str, Any]] = []
    text_to_ids: dict[str, list[str]] = defaultdict(list)

    for section_index, paragraphs in enumerate(section_paragraphs):
        section_name = (
            normalize_text(section_names[section_index])
            if section_index < len(section_names)
            else ""
        )

        for paragraph_index, paragraph_text in enumerate(ensure_list(paragraphs)):
            paragraph_text = normalize_text(paragraph_text)
            if not paragraph_text:
                continue

            paragraph_id = (
                f"{paper_id}_s{section_index:03d}_p{paragraph_index:03d}"
            )

            paragraph_rows.append(
                {
                    "paper_id": paper_id,
                    "paper_title": title,
                    "paper_abstract": abstract,
                    "section_index": section_index,
                    "section_name": section_name,
                    "paragraph_index": paragraph_index,
                    "paragraph_id": paragraph_id,
                    "paragraph_text": paragraph_text,
                    "data_split": split,
                }
            )
            text_to_ids[paragraph_text].append(paragraph_id)

    return paragraph_rows, text_to_ids


def map_evidence_to_paragraph_ids(
    evidence_items: Iterable[Any],
    text_to_ids: dict[str, list[str]],
) -> tuple[list[str], list[str]]:
    """
    Map QASPER paragraph-level evidence text back to generated paragraph IDs.

    Figures and tables often begin with 'FLOAT SELECTED'. They are kept in
    unmatched_evidence because this starter project initially retrieves text
    paragraphs only.
    """
    matched_ids: list[str] = []
    unmatched: list[str] = []

    for item in evidence_items:
        evidence_text = normalize_text(item)
        if not evidence_text:
            continue

        ids = text_to_ids.get(evidence_text, [])
        if ids:
            matched_ids.extend(ids)
        else:
            unmatched.append(evidence_text)

    # Keep order while removing duplicates.
    matched_ids = list(dict.fromkeys(matched_ids))
    unmatched = list(dict.fromkeys(unmatched))
    return matched_ids, unmatched


def build_question_rows(
    row: dict[str, Any],
    split: str,
    text_to_ids: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """Flatten all questions and answer annotations for one paper."""
    paper_id = normalize_text(row.get("id"))
    title = normalize_text(row.get("title"))
    qas = row.get("qas") or {}

    questions = get_nested_list(qas, "question")
    question_ids = get_nested_list(qas, "question_id")
    answer_groups = get_nested_list(qas, "answers")

    rows: list[dict[str, Any]] = []

    for question_index, question in enumerate(questions):
        question_text = normalize_text(question)
        question_id = (
            normalize_text(question_ids[question_index])
            if question_index < len(question_ids)
            else f"{paper_id}_q{question_index:04d}"
        )

        raw_answer_group = (
            answer_groups[question_index]
            if question_index < len(answer_groups)
            else []
        )
        answer_annotations = flatten_answers(raw_answer_group)

        gold_answers: list[str] = []
        all_evidence: list[str] = []
        unanswerable_votes: list[bool] = []

        for annotation in answer_annotations:
            answer_text = answer_to_text(annotation)
            if answer_text:
                gold_answers.append(answer_text)

            unanswerable_votes.append(
                bool(annotation.get("unanswerable", False))
            )
            all_evidence.extend(ensure_list(annotation.get("evidence")))

        gold_answers = list(dict.fromkeys(gold_answers))
        gold_paragraph_ids, unmatched_evidence = map_evidence_to_paragraph_ids(
            all_evidence,
            text_to_ids,
        )

        # Majority vote is used only as a starter field. Preserve every answer
        # annotation in gold_answers_json for later official evaluation.
        answerable = not (
            unanswerable_votes
            and sum(unanswerable_votes) > len(unanswerable_votes) / 2
        )

        rows.append(
            {
                "question_id": question_id,
                "question": question_text,
                "gold_paper_id": paper_id,
                "gold_paper_title": title,
                "gold_paragraph_ids_json": json.dumps(
                    gold_paragraph_ids,
                    ensure_ascii=False,
                ),
                "gold_answers_json": json.dumps(
                    gold_answers,
                    ensure_ascii=False,
                ),
                "primary_gold_answer": gold_answers[0] if gold_answers else "",
                "answerable": answerable,
                "has_text_evidence": bool(gold_paragraph_ids),
                "unmatched_evidence_json": json.dumps(
                    unmatched_evidence,
                    ensure_ascii=False,
                ),
                "data_split": split,
            }
        )

    return rows


def prepare_qasper(
    paragraphs_output: Path = PARAGRAPHS_CSV,
    questions_output: Path = QUESTIONS_CSV,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load QASPER and save flattened paragraph and question tables."""
    create_directories()

    dataset = load_dataset("allenai/qasper")
    if not isinstance(dataset, DatasetDict):
        raise TypeError("Expected QASPER to load as a DatasetDict.")

    all_paragraph_rows: list[dict[str, Any]] = []
    all_question_rows: list[dict[str, Any]] = []

    for split, split_dataset in dataset.items():
        print(f"Processing split: {split} ({len(split_dataset)} papers)")

        for row in split_dataset:
            paragraph_rows, text_to_ids = build_paragraph_rows(row, split)
            question_rows = build_question_rows(row, split, text_to_ids)

            all_paragraph_rows.extend(paragraph_rows)
            all_question_rows.extend(question_rows)

    paragraphs_df = pd.DataFrame(all_paragraph_rows)
    questions_df = pd.DataFrame(all_question_rows)

    paragraphs_output.parent.mkdir(parents=True, exist_ok=True)
    questions_output.parent.mkdir(parents=True, exist_ok=True)

    paragraphs_df.to_csv(paragraphs_output, index=False)
    questions_df.to_csv(questions_output, index=False)

    print(f"Saved {len(paragraphs_df):,} paragraphs to {paragraphs_output}")
    print(f"Saved {len(questions_df):,} questions to {questions_output}")
    print(
        "Questions with mapped text evidence:",
        f"{questions_df['has_text_evidence'].mean():.1%}",
    )

    return paragraphs_df, questions_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the QASPER dataset.")
    parser.add_argument(
        "--paragraphs-output",
        type=Path,
        default=PARAGRAPHS_CSV,
    )
    parser.add_argument(
        "--questions-output",
        type=Path,
        default=QUESTIONS_CSV,
    )
    args = parser.parse_args()

    prepare_qasper(args.paragraphs_output, args.questions_output)


if __name__ == "__main__":
    main()
