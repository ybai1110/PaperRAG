# PaperRAG

An agentic retrieval-augmented question answering system over NLP research papers, built on the [QASPER](https://huggingface.co/datasets/allenai/qasper) corpus.

Unlike a standard RAG demo, PaperRAG treats QASPER as an **open-domain** task: the paper is not given to the system. A query first has to find the right paper among all of them, then find the right paragraphs inside it, and only then produce a cited answer — or refuse to answer when the evidence does not support one.

<img width="1536" height="1024" alt="PaperRAG architecture" src="https://github.com/user-attachments/assets/463806e6-338e-44f4-97ae-8ad6745ed926" />

---

## What it does

| Capability | How |
| --- | --- |
| **Two-stage retrieval** | BM25 over a configurable title weight plus abstract, then BM25 over `paper title + section + paragraph` inside candidate papers |
| **Intent routing** | An LLM router classifies each query as `discovery`, `targeted_qa`, or `needs_context` and returns a confidence score and a reason |
| **Grounded generation** | Answers are produced only from retrieved passages, with paragraph-level citations |
| **Citation validation** | Citation IDs and passages are tied to retrieved evidence; duplicates and out-of-scope anchored-paper citations are rejected |
| **Explicit abstention** | The system returns `abstained: true` with a reason when evidence is missing, insufficient, or uncitable |
| **Explicit paper context** | Requests distinguish standalone queries, selected-paper questions, and unresolved/missing context without rewriting the user's words |
| **Evaluation harness** | Raw open-domain, paper-anchored, paper-oracle, and evidence-oracle conditions are logged and reported separately |

### The three routes

- **`discovery`** — a broad topic query (`"multilingual RAG evaluation"`). Returns ranked papers, no generated answer.
- **`targeted_qa`** — a specific, self-contained question (`"What evaluation metrics are used in the QASPER paper?"`) or any question with explicit selected-paper context. Runs evidence retrieval and answer generation.
- **`needs_context`** — a question with an unresolved referent (`"What dataset did they use?"`). Returns a clarifying question instead of guessing.

This third route exists because a large share of QASPER questions were written by annotators who could see the paper, so they are not answerable open-domain as written. Routing them out is treated as correct behaviour rather than a failure.

---

## Architecture

The agent is a [LangGraph](https://github.com/langchain-ai/langgraph) `StateGraph`:

```mermaid
flowchart TD
    A([User query]) --> B[surface_search<br/>BM25 · title + abstract · top 5]
    B --> C{query_router<br/>LLM structured output}
    C -->|discovery| D[discovery<br/>return ranked papers]
    C -->|needs_context| E[needs_context<br/>return clarifying question]
    C -->|targeted_qa| T[targeted_qa<br/>intentional pass-through]
    T --> F[evidence_retrieval<br/>BM25 · selected paper or top 3 papers · top 5]
    F --> G[generate_answer<br/>cited answer or abstention]
    D --> H([Response])
    E --> H
    G --> H
```

Shared state is a `TypedDict` (`src/agent/state.py`) carrying the query, paper candidates, route decision, evidence passages, answer, citations, and abstention flags. It also carries optional `selected_paper_id`, `selected_paper_title`, `question_context_mode`, and `paper_context_source`. The `targeted_qa` node is intentionally a visible pass-through: it preserves the route as a first-class graph stage while evidence retrieval remains independently testable.

Routing runs *after* surface search, so the router sees the retrieved titles and abstracts as extra signal — but its prompt explicitly instructs it not to let noisy retrieval override linguistic evidence in the query itself.

---

## Repository layout

```
src/
├── config.py                   # Paths, seeds, default top-k
├── api/
│   └── main.py                 # FastAPI service (/health, /search)
├── agent/
│   ├── graph.py                # LangGraph wiring
│   ├── deterministic_model.py  # Explicit no-network smoke/evaluation model
│   ├── state.py                # PaperRAGState schema
│   ├── nodes.py                # surface search, route nodes, scoped evidence retrieval
│   ├── router.py               # Intent classification with structured output
│   └── generator.py            # Cited answer generation + citation validation
├── retrieval/
│   ├── bm25_engine.py          # Reusable BM25Okapi retriever class
│   ├── paper_bm25.py           # Paper-level index (title + abstract)
│   ├── evidence_bm25.py        # Paragraph-level index
│   └── bm25.py                 # Offline BM25 baseline + metrics CLI
├── evaluation/
│   ├── evaluate_graph.py       # Direct-vs-two-stage controlled evaluator
│   ├── metrics.py              # Retrieval, answer, and abstention metrics
│   └── query_log.py            # QueryLogRecord + JSONL read/write
├── data/
│   ├── prepare_qasper.py       # Flatten QASPER into paragraph/question tables
│   ├── question_audit.py       # Legacy train-split validity audit
│   ├── question_context_audit.py # Validation context audit; proposals + manual gold
│   └── ai_label_audit.py       # LLM-assisted labelling of the same sample
└── pipelines/
    └── run_phases_1_to_5.py    # End-to-end runner for the offline phases

frontend/                       # Streamlit UI for the deployed API
test/                           # Retrieval and metric tests
```

---

## Quickstart

### Requirements

Python 3.12 (see `.python-version`) and an OpenAI API key.

```bash
git clone https://github.com/ybai1110/PaperRAG.git
cd PaperRAG

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

The same requirements file includes the offline evaluation and test dependencies.

### Environment

Create a `.env` file in the repository root:

```bash
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

`.env` is gitignored. Never place the API key in the Streamlit frontend — it belongs only in the backend environment.

### Run the API

```bash
uvicorn src.api.main:app --reload --port 8000
```

On startup the service loads the QASPER train split, builds the paper-level BM25 index, and compiles the agent graph. This takes a moment; `/health` reports `graph_ready` so you can tell when it is finished.

```bash
curl http://localhost:8000/health
# {"status":"ok","graph_ready":true}
```

For a no-API-key local smoke test only, explicitly set
`PAPERRAG_MODEL_MODE=deterministic`. This mode exercises the same graph, but its
extractive answers are not production-model quality. Retrieval controls are
available through `PAPER_TITLE_WEIGHT`, `PAPER_MIN_SCORE`,
`PAPER_MAX_SCORE_DROP`, `PAPER_TOP_K`, `EVIDENCE_MAX_PAPERS`,
`EVIDENCE_TOP_K`, and `EVIDENCE_PER_PAPER_LIMIT`.

### Run the UI

```bash
pip install streamlit requests
export PIPELINE_API_URL=http://localhost:8000
streamlit run frontend/app.py
```

Without `PIPELINE_API_URL` the app points at the hosted backend. See `frontend/README.md` for Streamlit Community Cloud deployment.

---

## API

### `GET /health`

```json
{ "status": "ok", "graph_ready": true }
```

### `POST /search`

```json
{ "query": "What evaluation metrics are used in the QASPER paper?" }
```

For a question about a paper selected in the UI/session, keep the question text
unchanged and send explicit context:

```json
{
  "query": "What dataset did they use?",
  "selected_paper_id": "1911.03072",
  "selected_paper_title": "A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers",
  "question_context_mode": "paper_anchored"
}
```

The three request modes are `standalone`, `paper_anchored`, and
`missing_context`. Supplying a selected paper automatically promotes the
default standalone mode to `paper_anchored`. A paper-anchored request always
uses the `targeted_qa` route, and evidence plus citations are restricted to the
selected paper.

Query length is validated at 2–1000 characters. Returns `503` while the graph is still loading.

```json
{
  "query": "What evaluation metrics are used in the QASPER paper?",
  "route": "targeted_qa",
  "route_confidence": 0.93,
  "route_reason": "Specific question with an identifiable paper name.",
  "question_context_mode": "standalone",
  "paper_context_source": "original_user_query",
  "selected_paper_id": null,
  "selected_paper_title": null,
  "results": [
    {
      "paper_id": "1911.03072",
      "rank": 1,
      "title": "A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers",
      "abstract": "...",
      "score": 21.47
    }
  ],
  "evidence": [
    {
      "paragraph_id": "1911.03072_s004_p002",
      "paper_id": "1911.03072",
      "title": "...",
      "section": "Evaluation",
      "text": "...",
      "rank": 1,
      "score": 14.02
    }
  ],
  "answer": "...",
  "citations": [
    {
      "paragraph_id": "1911.03072_s004_p002",
      "paper_id": "1911.03072",
      "title": "...",
      "section": "Evaluation",
      "passage": "..."
    }
  ],
  "clarification": null,
  "abstained": false,
  "abstention_reason": ""
}
```

Which fields are populated depends on the route: `discovery` returns papers with a null answer, `needs_context` returns a `clarification` question, and `targeted_qa` returns an answer with citations or an abstention.

---

## Offline pipeline

The offline side exists so that retrieval quality is measured rather than assumed.

**1 — Prepare the data.** Flattens all QASPER splits into a paragraph table and a question table, mapping each question's gold evidence spans back to stable paragraph IDs.

```bash
python -m src.data.prepare_qasper
# → data/processed/qasper_paragraphs.csv
# → data/processed/qasper_questions.csv
```

**2 — Audit question context.** Create the fixed sample from the validation
split (never the final test split):

```bash
python -m src.data.question_context_audit create --sample-size 100 --split validation
# manually review final_context_label and set manual_review_complete=true
python -m src.data.question_context_audit validate
```

The automatic `proposed_context_label` and `expected_route` are review aids,
not evaluation gold. Only a non-empty `final_context_label` on a row marked
`manual_review_complete=true` is eligible for normal scoring. Valid labels are
`standalone_discovery`, `standalone_targeted`,
`paper_dependent_repairable`, `ambiguous_needs_context`, and `unanswerable`.

**3 — Run the BM25 baseline.** Evaluates retrieval on the audited questions and writes a per-question JSONL log alongside aggregate metrics.

```bash
python -m src.retrieval.bm25 --top-k 10 --audit data/audits/question_audit.csv
# → results/logs/bm25_results.jsonl
# → results/metrics/bm25_metrics.json
```

Useful flags: `--question-validity` selects which audited category to score,
`--question-column` switches between original and rewritten question text,
`--index-mode contextual` indexes title and section context, and `--split` and
`--max-questions` restrict the evaluation set. Sampling is deterministic and
the selected question IDs are stored with the metrics.

**4 — Evaluate the deployed graph shape.** This writes aggregate JSON plus a
per-query JSONL while keeping `raw_open_domain`, `paper_anchored`,
`oracle_paper`, and `oracle_evidence` separate:

```bash
python -m src.evaluation.evaluate_graph \
  --split validation \
  --max-questions 50 \
  --model-mode deterministic
```

The raw graph input never receives the gold paper ID or title. Normal
paper/evidence correctness is reported only for manually reviewed standalone
questions; until review is complete those fields remain `null`, rather than
turning proposed labels into headline scores. Paper-anchored and oracle results
are explicitly diagnostic and are never presented as open-domain retrieval.

Use `--model-mode openai` only when `OPENAI_API_KEY` is available and a
production-model run is intended. Structural citation checks do not establish
semantic support; that still requires human review or a separate evaluator.

Or run the whole sequence, which pauses for the manual audit step:

```bash
python -m src.pipelines.run_phases_1_to_5
```

---

## Metrics

Defined in `src/evaluation/metrics.py`:

| Metric | Meaning |
| --- | --- |
| `paper_recall_at_k` | Did the gold paper appear in the top *k* results? |
| `paper_mrr_at_k` | Reciprocal rank of the gold paper within *k* results |
| `evidence_precision_recall_f1` | Set overlap between gold and retrieved paragraph IDs |
| `evidence_ranking_metrics` | Evidence Precision/Recall/F1/Success/MRR/nDCG at a requested cutoff; evaluation reports @5 and @10 |
| `max_answer_f1` | SQuAD-style token F1 against the best of several human gold answers |
| `abstention_precision_recall` | Were abstentions actually on unanswerable questions? |
| `grounded_answer_f1` | Answer F1 computed **only** when retrieval recovered at least one gold paragraph — separates generation quality from retrieval failure |

Every graph evaluation row records its exact `graph_input`, context mode, and
context source (`original_user_query`, `user_selected_session`,
`diagnostic_gold_paper_oracle`, or `query_rewrite`). Conditional evidence
metrics are computed on scored questions where surface search retrieved the
gold paper in its top five.

The answer-normalisation logic is a starter implementation; for reported research numbers it should be checked against the official QASPER evaluation script.

---

## Tests

```bash
pytest
```

The suite is deterministic and network-free. It covers metrics, retrieval
ranking and sampling, preprocessing/live paragraph-ID parity, all three graph
routes, abstention, citation grounding rules, title scoping, and API response
shape.

---

## Design notes

**Evaluation before retrieval.** The metric and logging layers were written first, so every subsequent retrieval change could be compared against a fixed baseline instead of judged by eye.

**BM25 as the baseline, not the endpoint.** A transparent lexical baseline makes it possible to attribute later gains to a dense or hybrid retriever rather than to incidental changes in preprocessing.

**Abstention as a first-class outcome.** A system that answers every question confidently is easy to build and hard to trust. Abstentions and clarification requests are recorded and scored like any other prediction.

## Roadmap

- Dense and hybrid retrieval compared against the BM25 baseline on identical logs
- Distractor-corpus scaling: measuring how retrieval degrades as non-QASPER papers are added to the index
- Near-duplicate confusion analysis for topically adjacent papers
- Closed-book contamination control — how much the generator can answer with no evidence at all

## License

Not yet specified.
