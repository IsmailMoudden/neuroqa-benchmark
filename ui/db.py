import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
DB_PATH     = RESULTS_DIR / "benchmark_history.db"

RESULTS_DIR.mkdir(exist_ok=True)

with sqlite3.connect(str(DB_PATH)) as _con:
    _con.executescript("""
        CREATE TABLE IF NOT EXISTS runs (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            run_at       TEXT NOT NULL,
            label        TEXT NOT NULL,
            raw_json     TEXT NOT NULL,
            summary_json TEXT NOT NULL
        );
    """)


def _db():
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    return con


def db_save_run(raw: list, summary: list, label: str = ""):
    if not label:
        strats = ", ".join(sorted({r["strategy_id"] for r in raw}))
        n_q    = len({r["q_id"] for r in raw})
        label  = f"{strats} · {n_q} question{'s' if n_q != 1 else ''}"
    run_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    with _db() as con:
        con.execute(
            "INSERT INTO runs (run_at, label, raw_json, summary_json) VALUES (?, ?, ?, ?)",
            (run_at, label, json.dumps(raw), json.dumps(summary)),
        )


def db_list_runs() -> list:
    with _db() as con:
        rows = con.execute("SELECT id, run_at, label FROM runs ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]


def db_load_run(run_id: int) -> tuple:
    with _db() as con:
        row = con.execute("SELECT raw_json, summary_json FROM runs WHERE id = ?", (run_id,)).fetchone()
    return (json.loads(row["raw_json"]), json.loads(row["summary_json"])) if row else ([], [])


def db_delete_run(run_id: int):
    with _db() as con:
        con.execute("DELETE FROM runs WHERE id = ?", (run_id,))
