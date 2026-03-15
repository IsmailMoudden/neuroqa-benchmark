"""
NeuroQA Benchmark — Simple UI for non-technical users.
Run with:  streamlit run app.py
"""

import json
import copy
import subprocess
import sys
import threading
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ─── Paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
SOURCES_DIR = HERE / "sources "          # trailing space matches existing folder
RESULTS_DIR = HERE / "results"
QA_PATH = HERE / "qa_dataset.py"

SOURCES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroQA Benchmark",
    page_icon=":material/troubleshoot:",
    layout="wide",
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def list_source_docs() -> list[str]:
    return sorted(f.name for f in SOURCES_DIR.iterdir() if f.suffix.lower() == ".docx")


def load_questions() -> list[dict]:
    ns = {}
    exec(QA_PATH.read_text(encoding="utf-8"), ns)
    return copy.deepcopy(ns["QA_DATASET"])


def save_questions(questions: list[dict]):
    lines = ["QA_DATASET = [\n"]
    for q in questions:
        lines.append("    {\n")
        lines.append(f'        "id": {json.dumps(q["id"])},\n')
        lines.append(f'        "type": {json.dumps(q.get("type", "factual"))},\n')
        lines.append(f'        "difficulty": {json.dumps(q.get("difficulty", "medium"))},\n')
        lines.append(f'        "source_doc": {json.dumps(q.get("source_doc", ""))},\n')
        lines.append(f'        "question": {json.dumps(q["question"])},\n')
        lines.append(f'        "expected": {json.dumps(q.get("expected", ""))},\n')
        kws_str = json.dumps(q.get("keywords", []))
        lines.append(f'        "keywords": {kws_str},\n')
        lines.append("    },\n")
    lines.append("]\n")
    QA_PATH.write_text("".join(lines), encoding="utf-8")


def load_results() -> tuple[list[dict], list[dict]]:
    raw_path = RESULTS_DIR / "raw_results.json"
    summary_path = RESULTS_DIR / "summary.json"
    raw = json.loads(raw_path.read_text()) if raw_path.exists() else []
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else []
    return raw, summary


# ─── Metric definitions ───────────────────────────────────────────────────────
METRIC_INFO = {
    "recall_at_5": {
        "label": "Recall@5",
        "help": "Out of all relevant document passages, how many were found in the top 5 results? "
                "A score of 1.0 means all relevant passages were retrieved.",
    },
    "mrr": {
        "label": "MRR",
        "help": "Mean Reciprocal Rank — how high does the first relevant passage rank? "
                "1.0 = always first, 0.5 = always second. Higher is better.",
    },
    "f1": {
        "label": "F1 Score",
        "help": "Word overlap between the generated answer and the expected answer. "
                "Combines precision (no extra words) and recall (no missing words). "
                "Only meaningful when an expected answer is provided.",
    },
    "faithfulness": {
        "label": "Faithfulness",
        "help": "Does the answer stay within what the retrieved passages say? "
                "A low score means the model is adding information not in the source.",
    },
    "relevance": {
        "label": "Relevance",
        "help": "Is the answer actually addressing the question asked? "
                "Scored by an AI judge from 0 (off-topic) to 1 (perfectly on-topic).",
    },
    "composite_score": {
        "label": "Composite Score",
        "help": "Weighted combination of all metrics above into a single overall score. "
                "Use this to quickly compare strategies.",
    },
}

DIFFICULTY_BADGE = {"easy": "Low", "medium": "Medium", "hard": "High"}
TYPE_LABELS = {
    "factual": "Factual",
    "definition": "Definition",
    "procedural": "Procedural",
    "multi_hop": "Multi-hop",
    "causal": "Causal",
}

# ─── Session state ────────────────────────────────────────────────────────────
if "questions" not in st.session_state:
    st.session_state.questions = load_questions()
if "benchmark_running" not in st.session_state:
    st.session_state.benchmark_running = False
if "benchmark_log" not in st.session_state:
    st.session_state.benchmark_log = ""


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — navigation
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.title("NeuroQA Benchmark")
page = st.sidebar.radio(
    "Navigate",
    [
        ":material/folder: Documents",
        ":material/list_alt: Questions",
        ":material/play_circle: Run Benchmark",
        ":material/bar_chart: Results",
    ],
    label_visibility="collapsed",
)
st.sidebar.divider()
st.sidebar.caption("Chunking strategy evaluation interface.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 0 — Documents
# ══════════════════════════════════════════════════════════════════════════════
if page == ":material/folder: Documents":
    st.title("Source Documents")
    st.caption(
        "Upload the `.docx` files you want to evaluate. "
        "They will be stored in the `sources/` folder and made available for benchmarking."
    )

    # ── Upload ────────────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload one or more Word documents",
        type=["docx"],
        accept_multiple_files=True,
        help="Only .docx files are supported.",
    )
    if uploaded:
        saved, skipped = [], []
        for f in uploaded:
            dest = SOURCES_DIR / f.name
            if dest.exists():
                skipped.append(f.name)
            else:
                dest.write_bytes(f.read())
                saved.append(f.name)
        if saved:
            st.success(f"Uploaded: {', '.join(saved)}")
        if skipped:
            st.info(f"Already exists (skipped): {', '.join(skipped)}")

    st.divider()

    # ── Document list ─────────────────────────────────────────────────────────
    docs = list_source_docs()
    st.markdown(f"### Documents in sources/ &nbsp; `{len(docs)} file{'s' if len(docs) != 1 else ''}`")

    if not docs:
        st.warning("No documents yet. Upload at least one .docx file before running the benchmark.")
    else:
        to_delete_doc = None
        for doc in docs:
            col_name, col_size, col_del = st.columns([5, 2, 1])
            size_kb = (SOURCES_DIR / doc).stat().st_size / 1024
            with col_name:
                st.markdown(f":material/description: &nbsp; {doc}")
            with col_size:
                st.caption(f"{size_kb:.1f} KB")
            with col_del:
                if st.button("Remove", key=f"del_doc_{doc}", icon=":material/delete:"):
                    to_delete_doc = doc

        if to_delete_doc:
            (SOURCES_DIR / to_delete_doc).unlink()
            st.success(f"Removed {to_delete_doc}")
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Questions
# ══════════════════════════════════════════════════════════════════════════════
elif page == ":material/list_alt: Questions":
    st.title("Question Bank")
    st.caption(f"{len(st.session_state.questions)} questions · Expected answers are optional")

    available_docs = list_source_docs()

    col_add, col_reload, col_save, _ = st.columns([1, 1, 1, 4])
    with col_add:
        if st.button("Add question", icon=":material/add:", use_container_width=True):
            new_id = f"Q{len(st.session_state.questions)+1:02d}"
            st.session_state.questions.append({
                "id": new_id,
                "type": "factual",
                "difficulty": "medium",
                "source_doc": available_docs[0] if available_docs else "",
                "question": "",
                "expected": "",
                "keywords": [],
            })
    with col_reload:
        if st.button("Reload from file", icon=":material/refresh:", use_container_width=True):
            st.session_state.questions = load_questions()
            st.success("Reloaded from qa_dataset.py")
    with col_save:
        if st.button("Save to file", icon=":material/save:", use_container_width=True, type="primary"):
            save_questions(st.session_state.questions)
            st.success("Saved to qa_dataset.py")

    if not available_docs:
        st.warning(
            "No source documents found. Go to the Documents page to upload your .docx files first.",
            icon=":material/warning:",
        )

    st.divider()

    to_delete = None
    for i, q in enumerate(st.session_state.questions):
        diff_label = DIFFICULTY_BADGE.get(q.get("difficulty", "medium"), "Medium")
        label = f"**{q['id']}** — {q['question'][:80] or '(empty question)'}  `{diff_label}`"
        with st.expander(label, expanded=False):
            c1, c2, c3 = st.columns([2, 2, 2])
            with c1:
                q["id"] = st.text_input("ID", value=q["id"], key=f"id_{i}")
            with c2:
                q["type"] = st.selectbox(
                    "Question type",
                    options=list(TYPE_LABELS.keys()),
                    index=list(TYPE_LABELS.keys()).index(q.get("type", "factual")),
                    format_func=lambda x: TYPE_LABELS[x],
                    key=f"type_{i}",
                    help="Factual: simple fact lookup. Definition: explain a term. "
                         "Procedural: step-by-step process. Multi-hop: requires combining several passages. "
                         "Causal: why/how something happens.",
                )
            with c3:
                q["difficulty"] = st.selectbox(
                    "Difficulty",
                    options=["easy", "medium", "hard"],
                    index=["easy", "medium", "hard"].index(q.get("difficulty", "medium")),
                    format_func=lambda x: DIFFICULTY_BADGE[x],
                    key=f"diff_{i}",
                )

            # Source doc — dropdown from uploaded files
            current_doc = q.get("source_doc", "")
            doc_options = available_docs if available_docs else [current_doc] if current_doc else []
            if doc_options:
                idx = doc_options.index(current_doc) if current_doc in doc_options else 0
                q["source_doc"] = st.selectbox(
                    "Source document",
                    options=doc_options,
                    index=idx,
                    key=f"src_{i}",
                    help="The document this question refers to. "
                         "Only documents uploaded in the Documents page appear here.",
                )
            else:
                st.caption("No documents available — upload files in the Documents page.")
                q["source_doc"] = current_doc

            q["question"] = st.text_area(
                "Question", value=q["question"], height=80, key=f"q_{i}"
            )
            q["expected"] = st.text_area(
                "Expected answer",
                value=q.get("expected", ""),
                height=100,
                key=f"exp_{i}",
                placeholder="(optional) — leave blank if you do not have a reference answer. "
                            "F1 Score will not be computed for this question.",
            )
            kw_str = ", ".join(q.get("keywords", []))
            new_kw = st.text_input(
                "Keywords",
                value=kw_str,
                key=f"kw_{i}",
                placeholder="comma-separated, e.g. CLS, settlement, payment",
                help="Words that should appear in a relevant passage. "
                     "Used to evaluate whether the right chunks were retrieved.",
            )
            q["keywords"] = [k.strip() for k in new_kw.split(",") if k.strip()]

            if st.button("Delete this question", icon=":material/delete:", key=f"del_{i}"):
                to_delete = i

    if to_delete is not None:
        st.session_state.questions.pop(to_delete)
        st.rerun()

    st.divider()
    if st.button("Save all changes", icon=":material/save:", type="primary", use_container_width=True):
        save_questions(st.session_state.questions)
        st.success("All changes saved to qa_dataset.py")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Run Benchmark
# ══════════════════════════════════════════════════════════════════════════════
elif page == ":material/play_circle: Run Benchmark":
    st.title("Run Benchmark")

    st.info(
        "This will run all chunking strategies against your questions. "
        "It may take several minutes depending on the number of questions.",
        icon=":material/info:",
    )

    docs = list_source_docs()
    if not docs:
        st.error(
            "No source documents found. Go to the Documents page and upload at least one .docx file.",
            icon=":material/error:",
        )
        st.stop()

    env_path = HERE / ".env"
    api_key_present = env_path.exists() and "OPENROUTER_API_KEY" in env_path.read_text()
    if not api_key_present:
        st.warning(
            "No API key found in the `.env` file. Make sure `OPENROUTER_API_KEY` is set before running.",
            icon=":material/warning:",
        )

    st.markdown(f"**{len(docs)} document(s) ready:** {', '.join(docs)}")
    st.markdown(f"**{len(st.session_state.questions)} question(s)** loaded.")
    st.divider()

    run_btn = st.button(
        "Start benchmark",
        icon=":material/play_arrow:",
        type="primary",
        disabled=st.session_state.benchmark_running,
    )

    if run_btn:
        st.session_state.benchmark_running = True
        st.session_state.benchmark_log = ""

        def _run():
            proc = subprocess.Popen(
                [sys.executable, "run_benchmark.py"],
                cwd=str(HERE),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            log = ""
            for line in proc.stdout:
                log += line
                st.session_state.benchmark_log = log
            proc.wait()
            st.session_state.benchmark_running = False

        threading.Thread(target=_run, daemon=True).start()
        st.rerun()

    if st.session_state.benchmark_running:
        st.warning("Benchmark is running — click Refresh to see latest output.")
        st.code(st.session_state.benchmark_log or "Starting…", language="")
        if st.button("Refresh log", icon=":material/refresh:"):
            st.rerun()
    elif st.session_state.benchmark_log:
        st.success("Benchmark complete. Go to the Results page to explore the output.")
        st.code(st.session_state.benchmark_log, language="")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Results
# ══════════════════════════════════════════════════════════════════════════════
elif page == ":material/bar_chart: Results":
    st.title("Benchmark Results")

    raw, summary = load_results()

    if not summary:
        st.warning("No results yet. Run the benchmark first.")
        st.stop()

    df_summary = pd.DataFrame(summary)
    df_raw = pd.DataFrame(raw)

    # ── Winner banner ─────────────────────────────────────────────────────────
    best = df_summary.loc[df_summary["composite_score"].idxmax()]
    st.success(
        f"Best strategy: **{best['strategy_id']} — {best['strategy_name']}**  "
        f"(Composite score: {best['composite_score']:.3f})",
        icon=":material/trophy:",
    )

    # ── Metric legend ─────────────────────────────────────────────────────────
    with st.expander("What do the metrics mean?", icon=":material/help:"):
        cols = st.columns(3)
        for idx, (key, info) in enumerate(METRIC_INFO.items()):
            with cols[idx % 3]:
                st.markdown(f"**{info['label']}**")
                st.caption(info["help"])

    # ── Summary table ─────────────────────────────────────────────────────────
    st.markdown("### Strategy Summary")
    display_cols = {
        "strategy_id": "ID",
        "strategy_name": "Strategy",
        "n_chunks": "# Chunks",
        "avg_tokens": "Avg Tokens",
        "recall_at_5": METRIC_INFO["recall_at_5"]["label"],
        "mrr": METRIC_INFO["mrr"]["label"],
        "f1": METRIC_INFO["f1"]["label"],
        "faithfulness": METRIC_INFO["faithfulness"]["label"],
        "relevance": METRIC_INFO["relevance"]["label"],
        "cost_per_query": "Cost / Query ($)",
        "composite_score": "Composite Score",
    }
    df_disp = df_summary.rename(columns=display_cols)[list(display_cols.values())]

    numeric_cols = [
        METRIC_INFO["recall_at_5"]["label"],
        METRIC_INFO["mrr"]["label"],
        METRIC_INFO["f1"]["label"],
        METRIC_INFO["faithfulness"]["label"],
        METRIC_INFO["relevance"]["label"],
        "Composite Score",
    ]

    def highlight_best(s):
        is_best = s == s.max()
        return ["background-color: #d4edda; font-weight: bold" if v else "" for v in is_best]

    styled = df_disp.style.apply(highlight_best, subset=numeric_cols).format(
        {c: "{:.3f}" for c in numeric_cols} | {"Cost / Query ($)": "{:.5f}", "Avg Tokens": "{:.1f}"}
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── Radar chart ───────────────────────────────────────────────────────────
    st.markdown("### Performance Radar")
    st.caption("Each axis is one metric (0 = worst, 1 = best). A larger filled area means a stronger strategy overall.")
    metrics_radar = ["recall_at_5", "mrr", "f1", "faithfulness", "relevance"]
    radar_labels = [METRIC_INFO[m]["label"] for m in metrics_radar]

    fig_radar = go.Figure()
    colors = px.colors.qualitative.Set2
    for i, row in df_summary.iterrows():
        vals = [row[m] for m in metrics_radar]
        vals_closed = vals + [vals[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals_closed,
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            name=f"{row['strategy_id']} — {row['strategy_name']}",
            line_color=colors[i % len(colors)],
            opacity=0.7,
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=420,
        margin=dict(t=30, b=30),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Bar chart ─────────────────────────────────────────────────────────────
    st.markdown("### Metric Comparison")
    metric_choice = st.selectbox(
        "Select a metric to compare",
        options=list(METRIC_INFO.keys()),
        format_func=lambda x: METRIC_INFO[x]["label"],
    )
    st.caption(METRIC_INFO[metric_choice]["help"])

    fig_bar = px.bar(
        df_summary,
        x="strategy_id",
        y=metric_choice,
        color="strategy_id",
        text=df_summary[metric_choice].round(3),
        labels={"strategy_id": "Strategy", metric_choice: METRIC_INFO[metric_choice]["label"]},
        color_discrete_sequence=px.colors.qualitative.Set2,
        height=350,
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(showlegend=False, margin=dict(t=20, b=20))
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Per-question heatmap ──────────────────────────────────────────────────
    if not df_raw.empty:
        st.markdown("### Per-Question F1 Heatmap")
        st.caption(
            "Each cell shows the F1 Score for one question (row) and one strategy (column). "
            "Green = good match with expected answer, red = poor match or no expected answer provided."
        )
        pivot = df_raw.pivot_table(
            index="q_id", columns="strategy_id", values="f1", aggfunc="mean"
        )
        fig_heat = px.imshow(
            pivot,
            color_continuous_scale="RdYlGn",
            zmin=0,
            zmax=1,
            labels={"color": "F1 Score"},
            aspect="auto",
            height=max(400, len(pivot) * 28),
        )
        fig_heat.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig_heat, use_container_width=True)

        col_d, col_t = st.columns(2)
        with col_d:
            st.markdown("#### F1 by Difficulty")
            st.caption("Are harder questions systematically worse for certain strategies?")
            df_diff = df_raw.groupby(["strategy_id", "difficulty"])["f1"].mean().reset_index()
            fig_d = px.bar(
                df_diff, x="difficulty", y="f1", color="strategy_id",
                barmode="group",
                labels={"f1": "Avg F1", "difficulty": "Difficulty"},
                color_discrete_sequence=px.colors.qualitative.Set2,
                height=320,
                category_orders={"difficulty": ["easy", "medium", "hard"]},
            )
            st.plotly_chart(fig_d, use_container_width=True)

        with col_t:
            st.markdown("#### F1 by Question Type")
            st.caption("Which types of questions are handled best by each strategy?")
            df_type = df_raw.groupby(["strategy_id", "type"])["f1"].mean().reset_index()
            fig_t = px.bar(
                df_type, x="type", y="f1", color="strategy_id",
                barmode="group",
                labels={"f1": "Avg F1", "type": "Type"},
                color_discrete_sequence=px.colors.qualitative.Set2,
                height=320,
            )
            st.plotly_chart(fig_t, use_container_width=True)

        # ── Full detail table ──────────────────────────────────────────────────
        st.markdown("### Full Detail Table")
        with st.expander("Show all results"):
            detail_cols = [
                "strategy_id", "q_id", "question", "difficulty", "type",
                "source_doc", "recall_at_5", "mrr", "f1", "faithfulness",
                "relevance", "answer", "expected",
            ]
            existing = [c for c in detail_cols if c in df_raw.columns]
            st.dataframe(df_raw[existing], use_container_width=True, hide_index=True)

    # ── Download ──────────────────────────────────────────────────────────────
    st.divider()
    xlsx_path = RESULTS_DIR / "benchmark_results.xlsx"
    if xlsx_path.exists():
        with open(xlsx_path, "rb") as f:
            st.download_button(
                label="Download Excel Report",
                icon=":material/download:",
                data=f,
                file_name="benchmark_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
            )
