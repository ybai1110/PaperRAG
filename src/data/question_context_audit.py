"""Create and validate a fixed QASPER question-context audit sample.

Automatic labels are proposals only. Evaluation may consume ``final_context_label``
only when ``manual_review_complete`` is true.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from src.config import AUDIT_DIR, QUESTIONS_CSV, RANDOM_SEED, create_directories


DEFAULT_OUTPUT = AUDIT_DIR / "question_context_audit_validation.csv"
VALID_LABELS = {
    "standalone_discovery",
    "standalone_targeted",
    "paper_dependent_repairable",
    "ambiguous_needs_context",
    "unanswerable",
}
LABEL_TO_ROUTE = {
    "standalone_discovery": "discovery",
    "standalone_targeted": "targeted_qa",
    "paper_dependent_repairable": "needs_context",
    "ambiguous_needs_context": "needs_context",
    "unanswerable": "targeted_qa",
}
REFERENT_PATTERN = re.compile(
    r"\b(they|their|them|this paper|the paper|this work|this approach|"
    r"the approach|this method|the method|the model|this model|it|here|"
    r"these tasks|these results|the proposed)\b",
    re.IGNORECASE,
)
DISCOVERY_PATTERN = re.compile(
    r"\b(which|what) (papers|work|methods|approaches|studies|datasets)\b|"
    r"\bpapers? (about|on|for)\b|\bliterature (about|on)\b",
    re.IGNORECASE,
)


def _truthy(value: object) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes", "y"}


def _contains_explicit_entity(question: str, paper_title: str) -> bool:
    question_terms = set(re.findall(r"\b[A-Za-z][A-Za-z0-9-]{2,}\b", question))
    title_terms = {
        term.casefold()
        for term in re.findall(r"\b[A-Za-z][A-Za-z0-9-]{3,}\b", paper_title)
    }
    named_or_acronym = any(
        (term.isupper() and len(term) >= 2)
        or (term[:1].isupper() and term.casefold() in title_terms)
        for term in question_terms
    )
    title_overlap = len(
        {term.casefold() for term in question_terms} & title_terms
    ) >= 2
    return named_or_acronym or title_overlap


def propose_context(row: pd.Series) -> dict[str, object]:
    question = str(row["question"]).strip()
    paper_title = str(row["gold_paper_title"]).strip()
    explicit = _contains_explicit_entity(question, paper_title)
    unresolved = bool(REFERENT_PATTERN.search(question)) and not explicit

    if not _truthy(row["answerable"]):
        label = "unanswerable"
        reason = "QASPER marks every available annotation as unanswerable."
        repairable = False
    elif DISCOVERY_PATTERN.search(question) and not unresolved:
        label = "standalone_discovery"
        reason = "The wording explicitly asks for papers, methods, or literature."
        repairable = False
    elif unresolved:
        label = "paper_dependent_repairable"
        reason = "The wording contains an unresolved paper/model/method referent."
        repairable = True
    elif explicit:
        label = "standalone_targeted"
        reason = "The wording contains a named entity or meaningful title overlap."
        repairable = False
    else:
        label = "ambiguous_needs_context"
        reason = "No reliable paper/entity anchor was detected in the question alone."
        repairable = True

    return {
        "proposed_context_label": label,
        "contains_explicit_paper_or_entity": explicit,
        "repairable_with_title": repairable,
        "expected_route": LABEL_TO_ROUTE[label],
        "review_reason": reason,
    }


def create_audit(
    questions_path: Path = QUESTIONS_CSV,
    output_path: Path = DEFAULT_OUTPUT,
    sample_size: int = 100,
    split: str = "validation",
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    create_directories()
    questions = pd.read_csv(
        questions_path,
        dtype={"question_id": "string", "gold_paper_id": "string"},
    ).fillna("")
    eligible = questions.loc[questions["data_split"] == split].copy()
    if eligible.empty:
        raise ValueError(f"No questions found for split {split!r}.")
    sample = eligible.sample(
        n=min(sample_size, len(eligible)),
        random_state=random_seed,
    ).sort_values("question_id", kind="stable")

    records = []
    for _, row in sample.iterrows():
        proposal = propose_context(row)
        records.append(
            {
                "question_id": str(row["question_id"]),
                "paper_id": str(row["gold_paper_id"]),
                "question": str(row["question"]),
                **proposal,
                "final_context_label": "",
                "manual_review_complete": False,
            }
        )
    columns = [
        "question_id",
        "paper_id",
        "question",
        "proposed_context_label",
        "final_context_label",
        "contains_explicit_paper_or_entity",
        "repairable_with_title",
        "expected_route",
        "review_reason",
        "manual_review_complete",
    ]
    audit = pd.DataFrame(records)[columns]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_path, index=False)
    metadata = {
        "split": split,
        "sample_size": len(audit),
        "random_seed": random_seed,
        "tuning_split_policy": "Validation only; final test split is not used.",
        "label_status": (
            "proposed_context_label and expected_route are heuristic proposals; "
            "only manually reviewed final_context_label values are evaluation gold."
        ),
        "selected_question_ids": audit["question_id"].tolist(),
    }
    output_path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return audit


def validate_audit(audit_path: Path = DEFAULT_OUTPUT) -> dict[str, int]:
    audit = pd.read_csv(audit_path, dtype=str).fillna("")
    completed = audit["manual_review_complete"].map(_truthy)
    final_labels = audit["final_context_label"].str.strip()
    invalid = sorted(set(final_labels[completed]) - VALID_LABELS)
    if invalid:
        raise ValueError("Invalid completed final labels: " + ", ".join(invalid))
    if (final_labels[~completed] != "").any():
        raise ValueError(
            "Rows with final_context_label must set manual_review_complete=true."
        )
    summary = {
        "rows": len(audit),
        "manual_review_complete": int(completed.sum()),
        "pending_manual_review": int((~completed).sum()),
    }
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--questions", type=Path, default=QUESTIONS_CSV)
    create.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    create.add_argument("--sample-size", type=int, default=100)
    create.add_argument("--split", default="validation", choices=["validation"])
    validate = subparsers.add_parser("validate")
    validate.add_argument("--audit", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "create":
        frame = create_audit(
            questions_path=args.questions,
            output_path=args.output,
            sample_size=args.sample_size,
            split=args.split,
        )
        print(f"Created {len(frame)} proposed rows at {args.output}")
    else:
        validate_audit(args.audit)


if __name__ == "__main__":
    main()
