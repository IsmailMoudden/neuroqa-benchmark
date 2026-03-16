import sys
from pathlib import Path
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from ui.storage import load_questions
from ui.pages import documents, questions, run, results

st.set_page_config(
    page_title="NeuroQA Testing tool ",
    page_icon=":material/troubleshoot:",
    layout="wide",
)

if "questions" not in st.session_state:
    st.session_state.questions = load_questions()
if "benchmark_pid" not in st.session_state:
    st.session_state.benchmark_pid = None

st.sidebar.title("NeuroQA Testing tool ")
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
active_count = len(st.session_state.questions)
st.sidebar.caption(f"{active_count} active question{'s' if active_count != 1 else ''}")
st.sidebar.caption("Chunking strategy evaluation interface.")

if page == ":material/folder: Documents":
    documents.render()
elif page == ":material/list_alt: Questions":
    questions.render()
elif page == ":material/play_circle: Run Benchmark":
    run.render()
elif page == ":material/bar_chart: Results":
    results.render()
