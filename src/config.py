"""Shared paths and basic settings for the first PaperRAG phases."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
AUDIT_DIR = DATA_DIR / "audits"

RESULTS_DIR = PROJECT_ROOT / "results"
LOG_DIR = RESULTS_DIR / "logs"
METRICS_DIR = RESULTS_DIR / "metrics"

PARAGRAPHS_CSV = PROCESSED_DIR / "qasper_paragraphs.csv"
QUESTIONS_CSV = PROCESSED_DIR / "qasper_questions.csv"
QUESTION_AUDIT_CSV = AUDIT_DIR / "question_audit.csv"

BM25_LOG_JSONL = LOG_DIR / "bm25_results.jsonl"
BM25_METRICS_JSON = METRICS_DIR / "bm25_metrics.json"

RANDOM_SEED = 42
DEFAULT_TOP_K = 10


def create_directories() -> None:
    """Create all folders needed by the starter pipeline."""
    for directory in [
        PROCESSED_DIR,
        AUDIT_DIR,
        LOG_DIR,
        METRICS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
