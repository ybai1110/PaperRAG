import unittest
from unittest.mock import Mock

from src.agent.generator import make_answer_generation_node
from src.agent.nodes import make_evidence_retrieval_node


class EmptyPaperEvidenceTest(unittest.TestCase):
    def test_empty_and_blank_passages_abstain(self) -> None:
        state = {
            "user_query": "What dataset did they use?",
            "question_context_mode": "paper_anchored",
            "selected_paper_id": "empty",
            "paper_candidates": [{"paper_id": "empty", "title": "Empty paper"}],
        }

        for paragraphs in ([], [[]], [["", "  "]]):
            with self.subTest(paragraphs=paragraphs):
                paper = {
                    "id": "empty",
                    "title": "Empty paper",
                    "full_text": {
                        "section_name": ["Methods"],
                        "paragraphs": paragraphs,
                    },
                }
                retrieve = make_evidence_retrieval_node({"empty": paper})
                result = retrieve(state)
                self.assertEqual(result["evidence_candidates"], [])

                model = Mock()
                generate = make_answer_generation_node(model)
                answer = generate({**state, **result})
                self.assertTrue(answer["abstained"])
                self.assertEqual(answer["citations"], [])
                self.assertEqual(
                    answer["abstention_reason"],
                    "No evidence passages were retrieved.",
                )
                model.with_structured_output.return_value.invoke.assert_not_called()
