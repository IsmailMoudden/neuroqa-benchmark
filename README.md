# NeuroQA Testing Tool

A Streamlit app to benchmark RAG chunking strategies on your own documents. You upload a Word file, define evaluation questions, run the benchmark, and compare how different chunking approaches affect retrieval and answer quality.

## What it does

The core idea is simple: given a document and a set of questions, which chunking strategy helps the model find the right passages and generate better answers?

Four strategies are benchmarked:
- **C** — Sliding window, 512 tokens, 25% overlap
- **D** — Sliding window, 512 tokens, 50% overlap
- **E** — Structure-aware (splits on headings and section boundaries)
- **F** — Semantic (splits where topic similarity drops)

Each strategy is evaluated on five metrics: Recall@5, MRR, F1, Faithfulness, and Relevance.

## Project structure

```
app.py                  entry point — sidebar routing only
run_benchmark.py        benchmark engine (embed → retrieve → generate → score)
metrics.py              all metric functions (retrieval + LLM-as-judge)
active_questions.py     your current question set (written by the UI)
sample_questions.py     16 preset CLS questions you can enable from the UI
labels.py               display labels and metric descriptions used by the UI

parsing/                how documents are read and parsed
chunking/               how parsed text is split into chunks
ui/                     Streamlit pages and data helpers
```

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Add your Azure credentials to a `.env` file

## Deploying to Streamlit Cloud

Push to GitHub and connect the repo on share.streamlit.io. Add the two Azure variables under App Settings → Secrets in TOML format


## Adding a new chunking strategy

1. Create a file in `chunking/` (e.g. `chunking/sentence_window.py`)
2. Export the function from `chunking/__init__.py`
3. Add a new entry to `STRATEGIES` in `run_benchmark.py`

## Adding a new file parser (PDF, HTML, etc.)

1. Create a parser in `parsing/` that implements `BaseParser`
2. Register it in `parsing/__init__.py` under `PARSERS`

Everything else — the benchmark, the UI, the metrics — picks it up automatically.
