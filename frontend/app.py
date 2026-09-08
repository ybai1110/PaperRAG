import os
from typing import Any

import requests
import streamlit as st


DEFAULT_API_URL = "https://paperrag-a48p.onrender.com"
ROUTE_LABELS = {
    "discovery": "Paper discovery",
    "targeted_qa": "Targeted question answering",
    "needs_context": "Needs more context",
}


def get_api_url() -> str:
    configured = os.getenv("PIPELINE_API_URL")
    try:
        configured = st.secrets.get("PIPELINE_API_URL", configured)
    except FileNotFoundError:
        pass
    return (configured or DEFAULT_API_URL).rstrip("/")


def choose_paper(paper_id: str, title: str) -> None:
    st.session_state.selected_paper = {
        "paper_id": paper_id,
        "title": title,
    }
    st.session_state.query = ""
    st.session_state.response = None
    st.session_state.error = None


def clear_paper() -> None:
    st.session_state.selected_paper = None
    st.session_state.query = ""
    st.session_state.response = None
    st.session_state.error = None


def search(query: str) -> None:
    selected = st.session_state.selected_paper
    backend_query = query
    if selected:
        backend_query = f'In the paper "{selected["title"]}", {query}'

    try:
        response = requests.post(
            f"{get_api_url()}/search",
            json={"query": backend_query},
            timeout=100,
        )
        response.raise_for_status()
        data = response.json()
        data["query"] = query
        st.session_state.response = data
        st.session_state.error = None
    except requests.Timeout:
        st.session_state.error = (
            "The request timed out. The backend may be waking up; try once more."
        )
        st.session_state.response = None
    except (requests.RequestException, ValueError) as exc:
        st.session_state.error = f"PaperRAG could not complete the request: {exc}"
        st.session_state.response = None


def render_citation(citation: dict[str, Any], number: int) -> None:
    title = citation.get("title") or "Untitled paper"
    section = citation.get("section") or "Untitled section"
    with st.expander(f"[{number}] {title} — {section}"):
        st.write(citation.get("passage") or "No passage returned.")
        st.caption(citation.get("paragraph_id") or "No paragraph ID")


def render_paper(
    paper: dict[str, Any],
    number: int,
    evidence: list[dict[str, Any]],
) -> None:
    title = paper.get("title") or "Untitled paper"
    paper_id = str(paper.get("paper_id") or "")
    score = paper.get("score")

    with st.expander(f"{number}. {title}"):
        if score is not None:
            st.caption(f"Paper ID: {paper_id} · BM25 score: {score:.4f}")
        else:
            st.caption(f"Paper ID: {paper_id}")

        abstract = paper.get("abstract")
        if abstract:
            st.write(abstract)

        st.button(
            "Ask this paper",
            key=f"select_{number}_{paper_id}",
            on_click=choose_paper,
            args=(paper_id, title),
        )

        paper_evidence = [
            passage
            for passage in evidence
            if str(passage.get("paper_id")) == paper_id
        ]
        if paper_evidence:
            st.markdown("**Retrieved passages**")
            for passage in paper_evidence:
                st.markdown(f"**{passage.get('section') or 'Untitled section'}**")
                st.write(passage.get("text") or "")
                st.caption(passage.get("paragraph_id") or "")


st.set_page_config(
    page_title="PaperRAG",
    page_icon="📄",
    layout="wide",
)

for key, default in {
    "query": "",
    "response": None,
    "error": None,
    "selected_paper": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

st.title("PaperRAG")
st.write("Search QASPER papers and ask questions supported by retrieved passages.")

selected_paper = st.session_state.selected_paper
if selected_paper:
    selected_col, clear_col = st.columns([5, 1])
    with selected_col:
        st.info(f"Selected paper: **{selected_paper['title']}**")
    with clear_col:
        st.button("Clear paper", on_click=clear_paper, use_container_width=True)

with st.form("search_form"):
    label = "Question about this paper" if selected_paper else "Research question"
    placeholder = (
        "Ask a specific question about the selected paper"
        if selected_paper
        else "Enter a research topic or a question about a named paper"
    )
    query = st.text_input(label, key="query", placeholder=placeholder)
    submitted = st.form_submit_button(
        "Ask PaperRAG",
        type="primary",
        use_container_width=True,
    )

if submitted:
    cleaned_query = query.strip()
    if len(cleaned_query) < 2:
        st.session_state.error = "Enter at least two characters."
        st.session_state.response = None
    else:
        with st.spinner("Searching papers and retrieving evidence…"):
            search(cleaned_query)

st.caption(
    "Try: papers about multilingual question answering · "
    "How does CamemBERT compare with multilingual BERT? · "
    "What dataset did they use?"
)

if st.session_state.error:
    st.error(st.session_state.error)

data = st.session_state.response
if data:
    route = data.get("route")
    st.divider()
    st.subheader(ROUTE_LABELS.get(route, route or "PaperRAG result"))
    if data.get("route_reason"):
        st.caption(data["route_reason"])

    if route == "needs_context":
        st.warning(data.get("clarification") or "Which paper or model do you mean?")

    if route == "targeted_qa":
        if data.get("abstained"):
            st.warning("The retrieved passages were not strong enough to answer confidently.")
            if data.get("abstention_reason"):
                st.write(data["abstention_reason"])
        else:
            st.success("Answer generated from retrieved evidence")
        st.markdown(data.get("answer") or "No answer was returned.")

        citations = data.get("citations") or []
        if citations:
            st.markdown("### Supporting citations")
            for index, citation in enumerate(citations, start=1):
                render_citation(citation, index)

    papers = data.get("results") or []
    evidence = data.get("evidence") or []
    st.markdown("### Retrieved papers")
    if papers:
        for index, paper in enumerate(papers, start=1):
            render_paper(paper, index, evidence)
    else:
        st.info("No papers were returned. Try a different topic or include a paper title.")

with st.sidebar:
    st.header("Demo status")
    st.write(f"Backend: `{get_api_url()}`")
    if st.button("Check backend", use_container_width=True):
        try:
            health = requests.get(f"{get_api_url()}/health", timeout=30)
            health.raise_for_status()
            if health.json().get("graph_ready"):
                st.success("Backend is ready")
            else:
                st.warning("Backend is starting")
        except (requests.RequestException, ValueError):
            st.error("Backend is unavailable")
    st.caption("Answers may contain errors. Check the displayed passages before relying on them.")
