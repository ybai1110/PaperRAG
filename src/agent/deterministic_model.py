"""Deterministic structured model for tests and explicit offline smoke runs."""

import re


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
STOP_WORDS = {
    "a", "an", "and", "are", "did", "do", "does", "for", "from",
    "how", "in", "is", "of", "on", "paper", "the", "they", "to",
    "used", "was", "were", "what", "which", "with",
}


class _DeterministicStructuredModel:
    def __init__(self, schema):
        self.schema = schema

    @staticmethod
    def _user_content(messages: list[dict[str, str]]) -> str:
        return str(messages[-1]["content"])

    def _route(self, content: str):
        match = re.search(
            r"User query:\s*(.*?)\s*Surface-search results:",
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )
        query = (match.group(1) if match else content).strip()
        lowered = query.casefold()
        discovery_signals = (
            "papers about", "literature on", "research on", "methods for",
        )
        unresolved = re.search(
            r"\b(they|their|this paper|this approach|the model|it)\b",
            lowered,
        )
        has_anchor = '"' in query or " qasper " in f" {lowered} "

        if any(signal in lowered for signal in discovery_signals):
            route = "discovery"
            reason = "Broad literature-discovery wording."
        elif unresolved and not has_anchor:
            route = "needs_context"
            reason = "The query contains an unresolved referent."
        else:
            route = "targeted_qa"
            reason = "A specific, self-contained question was provided."
        return self.schema(route=route, confidence=1.0, reason=reason)

    def _answer(self, content: str):
        question_match = re.search(
            r"Question:\s*(.*?)\s*Evidence:",
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )
        question = question_match.group(1).strip() if question_match else ""
        passages = re.findall(
            r"Paragraph ID: (.*?)\nPaper: .*?\nSection: .*?\nText: "
            r"(.*?)(?=\n\nParagraph ID:|\Z)",
            content,
            flags=re.DOTALL,
        )
        if not passages:
            return self.schema(
                answer="", cited_paragraph_ids=[], insufficient_evidence=True
            )

        evidence_blob = " ".join(text for _, text in passages)
        years = re.findall(r"\b(?:19|20)\d{2}\b", question)
        if years and not all(year in evidence_blob for year in years):
            return self.schema(
                answer="", cited_paragraph_ids=[], insufficient_evidence=True
            )

        query_terms = {
            term.casefold()
            for term in TOKEN_RE.findall(question)
            if term.casefold() not in STOP_WORDS
        }
        paragraph_id, text = max(
            passages,
            key=lambda item: len(
                query_terms
                & {term.casefold() for term in TOKEN_RE.findall(item[1])}
            ),
        )
        return self.schema(
            answer=text.strip(),
            cited_paragraph_ids=[paragraph_id.strip()],
            insufficient_evidence=False,
        )

    def invoke(self, messages):
        content = self._user_content(messages)
        if self.schema.__name__ == "RouterDecision":
            return self._route(content)
        if self.schema.__name__ == "GeneratedAnswer":
            return self._answer(content)
        raise TypeError(f"Unsupported schema: {self.schema.__name__}")


class DeterministicEvaluationModel:
    """No-network model; never enabled unless explicitly configured."""

    def with_structured_output(self, schema):
        return _DeterministicStructuredModel(schema)
