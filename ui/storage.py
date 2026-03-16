import copy
import json
from pathlib import Path

HERE        = Path(__file__).parent.parent
SOURCES_DIR = HERE / "sources "   # trailing space matches existing folder
RESULTS_DIR = HERE / "results"
QA_PATH     = HERE / "active_questions.py"

SOURCES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def list_source_docs() -> list:
    return sorted(f.name for f in SOURCES_DIR.iterdir() if f.suffix.lower() == ".docx")


def load_questions() -> list:
    ns = {}
    exec(QA_PATH.read_text(encoding="utf-8"), ns)
    return copy.deepcopy(ns.get("QA_DATASET", []))


def save_questions(questions: list):
    lines = ["QA_DATASET = [\n"]
    for q in questions:
        lines.append("    {\n")
        for key in ["id", "type", "difficulty", "source_doc", "question", "expected"]:
            lines.append(f'        {json.dumps(key)}: {json.dumps(q.get(key, ""))},\n')
        lines.append(f'        "keywords": {json.dumps(q.get("keywords", []))},\n')
        lines.append("    },\n")
    lines.append("]\n")
    QA_PATH.write_text("".join(lines), encoding="utf-8")


def load_results() -> tuple:
    raw_path     = RESULTS_DIR / "raw_results.json"
    summary_path = RESULTS_DIR / "summary.json"
    raw     = json.loads(raw_path.read_text())     if raw_path.exists()     else []
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else []
    return raw, summary
