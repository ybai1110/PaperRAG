from collections import defaultdict, deque
from typing import Any


class _StructuredInvoker:
    def __init__(self, schema, responses):
        self.schema = schema
        self.responses = responses

    def invoke(self, messages):
        response = self.responses
        if isinstance(response, deque):
            response = response.popleft()
        if callable(response):
            response = response(messages)
        if isinstance(response, self.schema):
            return response
        return self.schema.model_validate(response)


class ScriptedModel:
    """Small structured-output model used by deterministic graph tests."""

    def __init__(self, **responses: Any):
        self.responses = defaultdict(deque)
        for schema_name, values in responses.items():
            if callable(values):
                self.responses[schema_name] = values
            elif isinstance(values, list):
                self.responses[schema_name] = deque(values)
            else:
                self.responses[schema_name] = deque([values])

    def with_structured_output(self, schema):
        return _StructuredInvoker(schema, self.responses[schema.__name__])


def sample_papers() -> list[dict]:
    return [
        {
            "id": "qasper",
            "title": "QASPER Evaluation Study",
            "abstract": "QASPER scientific question answering evaluation.",
            "full_text": {
                "section_name": ["Evaluation"],
                "paragraphs": [[
                    "The evaluation metrics are exact match and F1.",
                    "The dataset contains scientific papers and questions.",
                    "Experiments were conducted in 2020.",
                ]],
            },
        },
        {
            "id": "discourse",
            "title": "Discourse Relation Detection",
            "abstract": "Methods for discourse relation detection.",
            "full_text": {
                "section_name": ["Methods"],
                "paragraphs": [["A discourse parser is evaluated."]],
            },
        },
        {
            "id": "vision",
            "title": "Visual Recognition",
            "abstract": "Image classification with neural networks.",
            "full_text": {
                "section_name": ["Results"],
                "paragraphs": [["Accuracy improves on image benchmarks."]],
            },
        },
    ]
