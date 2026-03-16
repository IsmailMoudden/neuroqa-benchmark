import os
import threading
import streamlit as st
from pathlib import Path
from ui.storage import RESULTS_DIR, list_source_docs
from ui.db import db_save_run, db_list_runs, db_load_run
from ui.storage import load_results

HERE = Path(__file__).parent.parent.parent

LOG_FILE  = RESULTS_DIR / "_benchmark_log.txt"
DONE_FILE = RESULTS_DIR / "_benchmark_done.txt"
ERR_FILE  = RESULTS_DIR / "_benchmark_error.txt"

STRATEGY_OPTIONS = {
    "C": "Sliding window — 512 tokens, overlap 128",
    "D": "Sliding window — 512 tokens, overlap 256",
    "E": "Structure-aware — 512 tokens",
    "F": "Semantic — threshold 0.75, max 512 tokens",
}


def _persist_to_db_if_new():
    raw, summary = load_results()
    if not raw:
        return
    runs = db_list_runs()
    if runs:
        last_raw, _ = db_load_run(runs[0]["id"])
        if last_raw == raw:
            return
    db_save_run(raw, summary)


@st.cache_resource(show_spinner="Loading embedding model (first run only)…")
def get_embed_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def _run_thread(active_strategies, embed_model, the_api_key, the_endpoint, the_questions, log_file, done_file, err_file):
    import io, contextlib, sys
    sys.path.insert(0, str(HERE))
    import run_benchmark as _rb

    # Make endpoint available to azure client in this thread
    os.environ["AZURE_API_ENDPOINT"] = the_endpoint

    buf = io.StringIO()
    strat_ids = ", ".join(s["id"] for s in active_strategies)
    buf.write(f"Strategies: {strat_ids}  |  Questions: {len(the_questions)}\n\n")
    log_file.write_text(buf.getvalue())

    try:
        with contextlib.redirect_stdout(buf):
            _rb.run_benchmark(
                strategies=active_strategies,
                embed_model=embed_model,
                api_key=the_api_key,
                questions=the_questions,
            )
        log_file.write_text(buf.getvalue())
        _persist_to_db_if_new()
        done_file.write_text("done")
    except Exception as exc:
        import traceback
        log_file.write_text(buf.getvalue() + f"\n\nERROR: {exc}\n{traceback.format_exc()}")
        err_file.write_text(str(exc))


def render():
    st.title("Run Benchmark")
    st.info("Select which chunking strategies to evaluate, then start the run.", icon=":material/info:")

    docs = list_source_docs()
    if not docs:
        st.error("No source documents found. Go to the Documents page and upload at least one .docx file.", icon=":material/error:")
        st.stop()

    if not st.session_state.questions:
        st.error("No active questions. Go to the Questions page and enable at least one question.", icon=":material/error:")
        st.stop()

    # Resolve Azure credentials: st.secrets (Cloud) > .env (local)
    api_key, api_key_source = None, "not found"
    try:
        if "AZURE_API_KEY" in st.secrets:
            api_key = str(st.secrets["AZURE_API_KEY"]).strip()
            api_key_source = "st.secrets"
        if "AZURE_API_ENDPOINT" in st.secrets:
            os.environ["AZURE_API_ENDPOINT"] = str(st.secrets["AZURE_API_ENDPOINT"]).strip()
    except Exception:
        pass
    if not api_key:
        env_path = HERE / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if "=" not in line or line.startswith("#"):
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip()
                if k == "AZURE_API_KEY":
                    api_key = v
                    api_key_source = ".env file"
                elif k == "AZURE_API_ENDPOINT":
                    os.environ["AZURE_API_ENDPOINT"] = v

    if not api_key:
        st.warning("No API key found. Add `AZURE_API_KEY` to Streamlit secrets (cloud) or a `.env` file (local).", icon=":material/warning:")
    else:
        st.caption(f":material/key: Azure API key loaded from **{api_key_source}** — `{api_key[:12]}...`")

    st.markdown("### Chunking strategies to run")
    st.caption("Select at least one. Running fewer strategies is faster.")
    sel_cols = st.columns(len(STRATEGY_OPTIONS))
    selected_strategies = []
    for col, (sid, label) in zip(sel_cols, STRATEGY_OPTIONS.items()):
        with col:
            if st.checkbox(f"**{sid}**  \n{label}", value=True, key=f"strat_{sid}"):
                selected_strategies.append(sid)

    if not selected_strategies:
        st.warning("Select at least one strategy to run.", icon=":material/warning:")

    st.divider()
    st.markdown(f"**{len(docs)} document(s) ready:** {', '.join(docs)}")
    st.markdown(f"**{len(st.session_state.questions)} active question(s)** · **{len(selected_strategies)} strategy/ies** selected")
    st.divider()

    _is_running = st.session_state.benchmark_pid is not None and not DONE_FILE.exists() and not ERR_FILE.exists()

    if st.button("Start benchmark", icon=":material/play_arrow:", type="primary", disabled=_is_running or not selected_strategies):
        os.environ["OPENROUTER_API_KEY"] = api_key or ""
        LOG_FILE.write_text("Benchmark started...\n")
        DONE_FILE.unlink(missing_ok=True)
        ERR_FILE.unlink(missing_ok=True)

        import sys as _sys
        _sys.path.insert(0, str(HERE))
        from run_benchmark import STRATEGIES as _ALL_STRATEGIES

        _active    = [s for s in _ALL_STRATEGIES if s["id"] in selected_strategies]
        _embed     = get_embed_model()
        _questions = list(st.session_state.questions)

        # Pass endpoint explicitly so the background thread can access it
        _endpoint = os.environ.get("AZURE_API_ENDPOINT", "")

        t = threading.Thread(
            target=_run_thread,
            args=(_active, _embed, api_key, _endpoint, _questions, LOG_FILE, DONE_FILE, ERR_FILE),
            daemon=True,
        )
        t.start()
        st.session_state.benchmark_pid = t.ident
        st.rerun()

    if _is_running:
        st.warning("Benchmark is running…")
        st.code(LOG_FILE.read_text() if LOG_FILE.exists() else "Starting…", language="")
        if st.button("Refresh log", icon=":material/refresh:"):
            st.rerun()
    elif DONE_FILE.exists():
        st.success("Benchmark complete. Go to the Results page to explore the output.")
        st.code(LOG_FILE.read_text() if LOG_FILE.exists() else "", language="")
    elif ERR_FILE.exists():
        st.error(f"Benchmark failed: {ERR_FILE.read_text()}")
        st.code(LOG_FILE.read_text() if LOG_FILE.exists() else "", language="")
