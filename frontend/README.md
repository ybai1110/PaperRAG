# PaperRAG Streamlit interface

This is a Python-only interface for the deployed PaperRAG API.

## Run locally

From the repository root:

```bash
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py
```

The app uses `https://paperrag-a48p.onrender.com` by default. To use another backend:

```bash
export PIPELINE_API_URL=http://localhost:8000
streamlit run streamlit_app/app.py
```

## Deploy on Streamlit Community Cloud

1. Push this directory to the GitHub repository.
2. Open <https://share.streamlit.io> and select **Create app**.
3. Choose the `ybai1110/PaperRAG` repository and the `main` branch.
4. Set the main file path to `streamlit_app/app.py`.
5. In Advanced settings, choose Python 3.12.
6. Deploy the app.

The dependency file is in the same directory as the entrypoint, so Community Cloud uses the small frontend-only dependency set instead of installing the backend retrieval stack.

The default Render API URL is already present in the app. If the backend address changes, add this Streamlit secret:

```toml
PIPELINE_API_URL = "https://your-api.example.com"
```

Do not put `OPENAI_API_KEY` in the Streamlit app. The key belongs only in the FastAPI backend environment.
