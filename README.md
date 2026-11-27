# DevDocs Assistant

A small agentic RAG demo for asking questions about technical documentation. It supports PDF, DOCX, Markdown, and TXT uploads; semantic retrieval; citations; follow-up questions; term explanations; document comparisons; and a second evidence-checking pass.

## Run locally

1. Create and activate a Python 3.12 virtual environment.
2. Install dependencies: `pip install -r requirements.txt`.
3. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and add your Gemini API key. Never commit this secret file.
4. Run `streamlit run app.py`.
5. Upload the files in `sample_docs/` to try the demo. Ask about timeout defaults to see the intended conflicting-document example.

If you are using a model name different from the example, set the model identifier available to your API key in `GEMINI_MODEL`.

## Features

- Parses PDF pages, Word paragraphs, Markdown headings, and text files while keeping source metadata.
- Splits text into overlapping passages, embeds them with Sentence Transformers, and indexes them in ChromaDB.
- Retrieves passages semantically and displays each supporting source.
- Uses a single Python orchestrator to route the question as regular Q&A, term explanation, or comparison; retrieves evidence; generates a response; and in Evidence checked agent mode performs a separate support and citation review.
- Offers Baseline RAG mode so the group can compare generation without the evidence review.
- Keeps indexed documents in temporary session memory for the demo. Re-upload documents after the app restarts.

## Evaluation

Use `evaluation/questions.csv` as the starter benchmark. Run the questions in both answer modes and record correctness, retrieval relevance, citation support, and correct handling of missing or conflicting content. Have group members independently rate answers and discuss disagreements. Expand the set if time allows.

## Deployment

1. Push the project to a GitHub repository.
2. In Streamlit Community Cloud, create an app from that repository and select `app.py` as the entry point.
3. Add these values in the deployment's Secrets field:

```toml
GEMINI_API_KEY = "your-key"
GEMINI_MODEL = "gemini-3.8-flash"
```

4. Deploy and try the sample questions from the public app URL.

## Limitations to state in the report

This is a course prototype. It does not provide user accounts or durable document storage. PDF text extraction will not read image-only scans unless OCR is added. The evidence review can reduce unsupported answers but does not guarantee correctness; evaluation should report errors as well as successes.
