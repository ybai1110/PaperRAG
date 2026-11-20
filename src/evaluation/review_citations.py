"""Prepare and tally human citation-support judgments from saved graph logs."""

import argparse
import hashlib
import json
from pathlib import Path

VERDICTS = ("supported", "partial", "unsupported")


def review_citations(log_path: Path, review_path: Path, condition: str) -> dict:
    log = log_path.read_bytes()
    fingerprint = hashlib.sha256(log).hexdigest()
    rows = []
    for line in log.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if (
            record.get("condition") != condition
            or record.get("route") != "targeted_qa"
            or record.get("abstained")
            or not str(record.get("answer") or "").strip()
        ):
            continue
        rows.append({
            "question_id": record["question_id"],
            "system_variant": record["system_variant"],
            "question": record["question"],
            "answer": record["answer"],
            "citations": record.get("citations", []),
            "verdict": None,
            "notes": "",
        })

    if review_path.exists():
        review = json.loads(review_path.read_text(encoding="utf-8"))
        if review["log_sha256"] != fingerprint or review["condition"] != condition:
            raise ValueError("The review file was prepared for a different log or condition.")
        judgments = review["rows"]
        original_fields = ("question_id", "system_variant", "question", "answer", "citations")
        if len(judgments) != len(rows) or any(
            saved[key] != original[key]
            for saved, original in zip(judgments, rows)
            for key in original_fields
        ):
            raise ValueError("The review rows changed. Edit only verdict and notes.")
    else:
        with review_path.open("x", encoding="utf-8") as output:
            json.dump(
                {"log_sha256": fingerprint, "condition": condition, "rows": rows},
                output, indent=2, ensure_ascii=False,
            )
            output.write("\n")
        judgments = rows

    counts = dict.fromkeys(VERDICTS, 0)
    for row in judgments:
        verdict = row["verdict"]
        if verdict is None:
            continue
        if verdict not in VERDICTS:
            raise ValueError(f"Invalid verdict for {row['question_id']}: {verdict!r}")
        if verdict in {"supported", "partial"} and not row["citations"]:
            raise ValueError(f"A response with no citations cannot earn support: {row['question_id']}")
        counts[verdict] += 1

    answered = len(rows)
    reviewed = sum(counts.values())
    cited = sum(bool(row["citations"]) for row in rows)
    return {
        "condition": condition,
        "review_denominator": answered,
        "reviewed_count": reviewed,
        "review_coverage": reviewed / answered if answered else None,
        "cited_answer_count": cited,
        "citation_coverage": cited / answered if answered else None,
        "verdict_counts": counts,
        "supported_rate_among_reviewed": counts["supported"] / reviewed if reviewed else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--review", type=Path, required=True, help="JSON review file to create or tally.")
    parser.add_argument("--condition", default="raw_open_domain")
    args = parser.parse_args()
    print(json.dumps(review_citations(args.log, args.review, args.condition), indent=2))


if __name__ == "__main__":
    main()
