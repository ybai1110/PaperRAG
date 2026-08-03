from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "enum": ["standalone", "repairable", "ambiguous"],
        },
        "reason": {
            "type": "string",
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
        "missing_information": {
            "type": "string",
        },
    },
    "required": [
        "label",
        "reason",
        "confidence",
        "missing_information",
    ],
    "additionalProperties": False,
}


SYSTEM_PROMPT = """
You are auditing questions for an open-domain scientific paper retrieval task.

The retrieval system sees only the question. It does not know the paper title,
paper ID, authors, gold answer, supporting evidence, or surrounding conversation.

Assign exactly one label:

standalone:
The question contains enough meaningful scientific information to support a
reasonable search across a collection of papers. It does not need to uniquely
identify one paper. A question can be standalone if it includes a recognizable
research topic, task, dataset, method, model type, experiment, comparison, or
technical concept that would help narrow the search.

repairable:
The question is understandable, but it is missing one important anchor that
would make retrieval more reliable. It could become searchable by adding a
small amount of information, such as a model name, dataset, task, method,
research topic, or paper title.

ambiguous:
The question depends strongly on missing context and provides little useful
information for retrieval. Many unrelated papers could reasonably match it,
and adding only one small detail may not be enough to clarify the intended
paper.

Decision rules:

1. Do not require the question to uniquely identify a single paper.
2. Prefer standalone when the question contains enough technical or topical
   detail to retrieve a relevant group of papers.
3. Use repairable when the question has a clear meaning but lacks one key
   identifying detail.
4. Use ambiguous only when the question is highly generic or depends mainly on
   references such as "the paper," "the authors," "this method," "their model,"
   or "the results" without enough additional context.
5. Do not guess the source paper or use outside knowledge.
6. Judge only whether the question is usable for retrieval, not whether the
   answer can be found easily.

Tie-breaking rule:
If you are unsure between standalone and repairable, choose standalone when the
question contains a specific scientific topic, method, dataset, task, or
technical concept. If you are unsure between repairable and ambiguous, choose
repairable when adding one short phrase would likely make the question usable.

Return:
- label
- brief reason
- confidence from 0 to 1
- missing_information, if any

Example 1:
Question: What datasets are used to evaluate multilingual question answering?
Label: standalone
Reason: The question includes a clear research task and topic.

Example 2:
Question: What dataset did the authors use?
Label: repairable
Reason: The question is understandable but needs the model, task, or paper topic.

Example 3:
Question: What are the main results?
Label: ambiguous
Reason: The question provides almost no searchable scientific context.
"""


def label_question(
    client: OpenAI,
    model: str,
    question_id: str,
    question: str,
) -> dict:
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"Question ID: {question_id}\n"
                    f"Question: {question}"
                ),
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "question_audit_label",
                "schema": LABEL_SCHEMA,
                "strict": True,
            }
        },
    )

    return json.loads(response.output_text)


def label_audit_file(
    input_path: Path,
    output_path: Path,
) -> None:
    client = OpenAI()
    model = os.environ["OPENAI_MODEL"]

    audit = pd.read_csv(input_path).fillna("")

    for index, row in audit.iterrows():
        # Skip questions that have already been processed.
        if row.get("ai_label_1", ""):
            continue

        result = label_question(
            client=client,
            model=model,
            question_id=str(row["question_id"]),
            question=str(row["question"]),
        )

        audit.loc[index, "ai_label_1"] = result["label"]
        audit.loc[index, "ai_reason_1"] = result["reason"]
        audit.loc[index, "ai_confidence_1"] = result["confidence"]
        audit.loc[index, "ai_missing_information_1"] = (
            result["missing_information"]
        )

        # Save after every row so progress is not lost.
        audit.to_csv(output_path, index=False)
        time.sleep(0.2)


if __name__ == "__main__":
    label_audit_file(
        input_path=Path("data/audits/question_audit.csv"),
        output_path=Path("data/audits/question_audit_ai.csv"),
    )