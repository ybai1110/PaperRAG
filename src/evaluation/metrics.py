"""Phase 3: Reusable evaluation metrics for retrieval and later generation."""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable, Sequence


def paper_recall_at_k(
    gold_paper_id: str,
    retrieved_paper_ids: Sequence[str],
    k: int,
) -> float:
    """Return 1.0 when the gold paper occurs in the first k results."""
    if k <= 0:
        raise ValueError("k must be greater than 0.")
    return float(gold_paper_id in list(retrieved_paper_ids)[:k])


def evidence_precision_recall_f1(
    gold_paragraph_ids: Iterable[str],
    retrieved_paragraph_ids: Iterable[str],
) -> dict[str, float]:
    """Compute set-based paragraph evidence precision, recall, and F1."""
    gold = set(gold_paragraph_ids)
    retrieved = set(retrieved_paragraph_ids)

    overlap = len(gold & retrieved)

    precision = overlap / len(retrieved) if retrieved else 0.0
    recall = overlap / len(gold) if gold else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "evidence_precision": precision,
        "evidence_recall": recall,
        "evidence_f1": f1,
    }


def normalize_answer(text: str) -> str:
    """
    Apply a SQuAD-style normalization for a simple starter Answer F1.

    For final research reporting, compare this implementation against the
    official QASPER evaluation code.
    """
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, gold_answer: str) -> float:
    """Calculate token overlap F1 between one prediction and one gold answer."""
    prediction_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold_answer).split()

    if not prediction_tokens and not gold_tokens:
        return 1.0
    if not prediction_tokens or not gold_tokens:
        return 0.0

    common = Counter(prediction_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(prediction_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def max_answer_f1(
    prediction: str,
    gold_answers: Sequence[str],
) -> float:
    """Use the highest F1 across multiple human gold answers."""
    if not gold_answers:
        return 0.0
    return max(token_f1(prediction, gold) for gold in gold_answers)


def abstention_precision_recall(
    gold_unanswerable: Sequence[bool],
    predicted_abstained: Sequence[bool],
) -> dict[str, float]:
    """Evaluate whether abstentions occurred on truly unanswerable questions."""
    if len(gold_unanswerable) != len(predicted_abstained):
        raise ValueError("Gold and prediction lists must have equal length.")

    true_positive = sum(
        gold and predicted
        for gold, predicted in zip(gold_unanswerable, predicted_abstained)
    )
    predicted_positive = sum(predicted_abstained)
    actual_positive = sum(gold_unanswerable)

    precision = (
        true_positive / predicted_positive
        if predicted_positive
        else 0.0
    )
    recall = true_positive / actual_positive if actual_positive else 0.0

    return {
        "abstention_precision": precision,
        "abstention_recall": recall,
    }


def grounded_answer_f1(
    prediction: str,
    gold_answers: Sequence[str],
    gold_paragraph_ids: Sequence[str],
    retrieved_paragraph_ids: Sequence[str],
) -> float | None:
    """
    Return Answer F1 only when at least one gold evidence paragraph was found.

    None means the question is excluded from the conditional metric because
    retrieval did not recover any gold evidence.
    """
    evidence_found = bool(
        set(gold_paragraph_ids) & set(retrieved_paragraph_ids)
    )
    if not evidence_found:
        return None
    return max_answer_f1(prediction, gold_answers)
