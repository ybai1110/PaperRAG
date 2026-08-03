"""
Convenience runner for the first PaperRAG phases.

The manual Phase 2 audit cannot be completed automatically, so this runner:
1. Runs Phase 1.
2. Creates the Phase 2 audit file if it does not exist.
3. Stops and asks you to fill the audit CSV.
4. After the audit contains standalone labels, runs the BM25 baseline.
"""

from src.config import QUESTION_AUDIT_CSV
from src.data.prepare_qasper import prepare_qasper
from src.data.question_audit import create_audit_file
from src.retrieval.bm25 import run_bm25


def main() -> None:
    prepare_qasper()

    if not QUESTION_AUDIT_CSV.exists():
        create_audit_file(sample_size=100)
        print()
        print("STOP: Open the following file and complete question_validity:")
        print(QUESTION_AUDIT_CSV)
        return

    run_bm25(
        top_k=10,
        audit_path=QUESTION_AUDIT_CSV,
    )


if __name__ == "__main__":
    main()
