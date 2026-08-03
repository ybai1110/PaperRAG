from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import (
    QUESTION_AUDIT_CSV,
    QUESTIONS_CSV,
    RANDOM_SEED,
    create_directories,
)

VALID_LABELS = {"standalone", "repairable", "ambiguous"}


def create_audit_file(
    questions_path: Path = QUESTIONS_CSV,
    output_path: Path = QUESTION_AUDIT_CSV,
    sample_size: int = 100,
    split: str | None = None,
    require_text_evidence: bool = True,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Sample questions and create a blank manual-review CSV."""
    create_directories()

    if not questions_path.exists():
        raise FileNotFoundError(
            f"{questions_path} does not exist. Run phase1_prepare_qasper.py first."
        )

    questions = pd.read_csv(questions_path)

    if split is not None:
        questions = questions.loc[questions["data_split"] == split].copy()

    if require_text_evidence:
        questions = questions.loc[questions["has_text_evidence"] == True].copy()  # noqa: E712

    if questions.empty:
        raise ValueError("No questions remain after applying the filters.")

    actual_sample_size = min(sample_size, len(questions))
    sample = questions.sample(
        n=actual_sample_size,
        random_state=random_seed,
    ).copy()

    audit = sample[["question_id", "question", "data_split"]].copy()
    audit["question_validity"] = ""
    audit["reason"] = ""
    audit["rewritten_question"] = ""
    audit["reviewer"] = ""
    audit["review_status"] = "pending"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_path, index=False)

    print(f"Created audit file with {len(audit)} questions:")
    print(output_path)
    print()
    print("Fill question_validity with one of:")
    print("  standalone | repairable | ambiguous")

    return audit


def summarize_audit(
    audit_path: Path = QUESTION_AUDIT_CSV,
) -> pd.DataFrame:
    """Validate completed labels and print a simple audit summary."""
    if not audit_path.exists():
        raise FileNotFoundError(f"Audit file not found: {audit_path}")

    audit = pd.read_csv(audit_path).fillna("")
    labels = audit["question_validity"].str.strip().str.lower()

    invalid = sorted(
        {
            label
            for label in labels
            if label and label not in VALID_LABELS
        }
    )
    if invalid:
        raise ValueError(
            "Invalid question_validity labels found: "
            + ", ".join(invalid)
        )

    summary = (
        labels.replace("", "unreviewed")
        .value_counts(dropna=False)
        .rename_axis("question_validity")
        .reset_index(name="count")
    )
    summary["percentage"] = summary["count"] / len(audit)

    print(summary.to_string(index=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or summarize question audit.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create")
    create_parser.add_argument("--questions", type=Path, default=QUESTIONS_CSV)
    create_parser.add_argument("--output", type=Path, default=QUESTION_AUDIT_CSV)
    create_parser.add_argument("--sample-size", type=int, default=100)
    create_parser.add_argument("--split", type=str, default=None)
    create_parser.add_argument(
        "--include-without-text-evidence",
        action="store_true",
    )

    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--audit", type=Path, default=QUESTION_AUDIT_CSV)

    args = parser.parse_args()

    if args.command == "create":
        create_audit_file(
            questions_path=args.questions,
            output_path=args.output,
            sample_size=args.sample_size,
            split=args.split,
            require_text_evidence=not args.include_without_text_evidence,
        )
    else:
        summarize_audit(args.audit)


if __name__ == "__main__":
    main()
