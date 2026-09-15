from pydantic import BaseModel

from src.agent.state import PaperRAGState


class GeneratedAnswer(BaseModel):
    answer: str
    cited_paragraph_ids: list[str]
    insufficient_evidence: bool


GENERATOR_PROMPT = """
You answer questions about scientific papers using only the
provided evidence passages.

Rules:

1. Use only information in the supplied evidence.
2. Do not use outside knowledge.
3. Give a concise but complete answer.
4. Cite the paragraph IDs that support the answer.
5. Only cite paragraph IDs present in the evidence.
6. If the evidence cannot answer the question, set
   insufficient_evidence to true.
"""


def format_evidence(
    evidence_candidates: list[dict],
) -> str:

    passages = []

    for evidence in evidence_candidates:
        passages.append(
            f"""
Paragraph ID: {evidence["paragraph_id"]}
Paper: {evidence["title"]}
Section: {evidence["section"]}
Text: {evidence["text"]}
""".strip()
        )

    return "\n\n".join(passages)


def make_answer_generation_node(model):

    structured_model = model.with_structured_output(
        GeneratedAnswer
    )

    def generate_answer_node(
        state: PaperRAGState,
    ) -> dict:

        evidence_candidates = state.get(
            "evidence_candidates",
            [],
        )

        if not evidence_candidates:
            return {
                "answer": (
                    "I could not find sufficient evidence "
                    "to answer this question."
                ),
                "citations": [],
                "abstained": True,
                "abstention_reason": (
                    "No evidence passages were retrieved."
                ),
            }

        evidence_text = format_evidence(
            evidence_candidates
        )

        decision = structured_model.invoke(
            [
                {
                    "role": "system",
                    "content": GENERATOR_PROMPT,
                },
                {
                    "role": "user",
                    "content": (
                        f"Question:\n"
                        f"{state['user_query']}\n\n"
                        f"Evidence:\n"
                        f"{evidence_text}"
                    ),
                },
            ]
        )

        evidence_by_id = {
            evidence["paragraph_id"]: evidence
            for evidence in evidence_candidates
        }

        valid_citation_ids = []
        citation_scope_violations = []
        intended_paper_id = state.get("intended_paper_id")

        for paragraph_id in decision.cited_paragraph_ids:
            if paragraph_id in evidence_by_id:
                if (
                    intended_paper_id is not None
                    and str(evidence_by_id[paragraph_id]["paper_id"])
                    != str(intended_paper_id)
                ):
                    if paragraph_id not in citation_scope_violations:
                        citation_scope_violations.append(paragraph_id)
                    continue
                if paragraph_id not in valid_citation_ids:
                    valid_citation_ids.append(
                        paragraph_id
                    )

        citations = [
            {
                "paragraph_id": paragraph_id,
                "paper_id": evidence_by_id[
                    paragraph_id
                ]["paper_id"],
                "title": evidence_by_id[
                    paragraph_id
                ]["title"],
                "section": evidence_by_id[
                    paragraph_id
                ]["section"],
                "passage": evidence_by_id[
                    paragraph_id
                ]["text"],
            }
            for paragraph_id in valid_citation_ids
        ]

        if decision.insufficient_evidence:
            return {
                "answer": (
                    "I could not find sufficient evidence "
                    "to answer this question."
                ),
                "citations": citations,
                "abstained": True,
                "abstention_reason": (
                    "The retrieved passages did not provide "
                    "enough information."
                ),
                "citation_scope_violations": citation_scope_violations,
            }

        if not citations:
            return {
                "answer": (
                    "I could not produce a properly cited "
                    "answer from the retrieved evidence."
                ),
                "citations": [],
                "abstained": True,
                "abstention_reason": (
                    "The model did not return valid citations."
                ),
                "citation_scope_violations": citation_scope_violations,
            }

        return {
            "answer": decision.answer,
            "citations": citations,
            "abstained": False,
            "abstention_reason": "",
            "citation_scope_violations": citation_scope_violations,
        }

    return generate_answer_node
