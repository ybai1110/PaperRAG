from pathlib import Path

import pytest
import requests
from fastapi.testclient import TestClient

from src.api import main as api
from test.helpers import sample_papers


def test_clarification_title_search_reruns_original_question(monkeypatch):
    app_test = pytest.importorskip("streamlit.testing.v1").AppTest
    monkeypatch.setenv("PAPERRAG_MODEL_MODE", "deterministic")
    monkeypatch.setenv("PIPELINE_API_URL", "http://paperrag.test")
    monkeypatch.setattr(api, "load_dataset", lambda *args, **kwargs: sample_papers())
    original = "What dataset did they use?"
    title = "QASPER Evaluation Study"
    posts, lookups = [], []

    with TestClient(api.app) as client:
        def post(url, *, json, timeout):
            assert url == "http://paperrag.test/search"
            posts.append(json)
            return client.post("/search", json=json)

        def get(url, *, params, timeout):
            assert url == "http://paperrag.test/papers"
            lookups.append(params)
            return client.get("/papers", params=params)

        monkeypatch.setattr(requests, "post", post)
        monkeypatch.setattr(requests, "get", get)
        app = app_test.from_file(str(Path(__file__).resolve().parents[1] / "frontend/app.py")).run()

        def click(label):
            next(button for button in app.button if button.label == label).click().run()
            assert not app.exception

        app.text_input(key="query").set_value(original)
        click("Ask PaperRAG")
        assert app.session_state.response["route"] == "needs_context"
        assert app.session_state.pending_question == original
        assert any(warning.value == "Which paper do you mean?" for warning in app.warning)

        for lookup in ("no such paper", "qasPer  evaluation"):
            next(widget for widget in app.text_input if widget.label == "Paper title").set_value(lookup)
            click("Search paper titles")
            assert len(posts) == 1
            assert app.session_state.pending_question == original
            assert app.session_state.response["query"] == original

        assert lookups == [{"title": "no such paper"}, {"title": "qasPer  evaluation"}]
        assert [paper["title"] for paper in app.session_state.paper_results] == [title]

        click("Ask this paper")
        assert posts == [
            {"query": original, "question_context_mode": "standalone"},
            {
                "query": original,
                "question_context_mode": "paper_anchored",
                "selected_paper_id": "qasper",
                "selected_paper_title": title,
            },
        ]
        assert app.text_input(key="query").value == original
        assert app.session_state.pending_question is None
        assert app.session_state.paper_results is None
        answer = app.session_state.response
        assert answer["route"] == "targeted_qa"
        assert {citation["paper_id"] for citation in answer["citations"]} == {"qasper"}
