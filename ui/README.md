# ui/

Everything Streamlit-related. The app is split into pages so each one can be worked on independently.

## Structure

**app.py** (root) — sets up the page config, session state, and sidebar. Just routes to the right page, nothing else.

**storage.py** — file I/O helpers: listing source documents, loading/saving questions, reading result JSON files.

**db.py** — SQLite-backed run history. Saves benchmark results after each run so they persist across sessions. Functions: `db_save_run`, `db_list_runs`, `db_load_run`, `db_delete_run`.

## Pages

Each page is a module with a single `render()` function called from `app.py`.

| File | Page | What it does |
|---|---|---|
| `pages/documents.py` | Documents | Upload and manage source `.docx` files |
| `pages/questions.py` | Questions | Edit active questions, browse sample library |
| `pages/run.py` | Run Benchmark | Pick strategies, start the benchmark, follow the log |
| `pages/results.py` | Results | View past runs, metrics table, charts, per-question drill-down |

## Adding a new page

1. Create `pages/your_page.py` with a `render()` function
2. Add it to the sidebar radio in `app.py`
3. Add the routing `elif` at the bottom of `app.py`
