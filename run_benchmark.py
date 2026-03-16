import os
import json
import time
import random
import hashlib
from pathlib import Path
from typing import List, Dict

from openai import AzureOpenAI
import numpy as np
from sentence_transformers import SentenceTransformer
import openpyxl
from openpyxl.styles import PatternFill, Font

from chunking import (
    load_docx_text,
    load_docx_structured,
    sliding_window_chunks,
    structure_aware_chunks,
    semantic_chunks,
)
from metrics import (
    is_relevant_chunk,
    recall_at_k,
    mrr,
    token_f1,
    faithfulness_score,
    relevance_score,
    compute_cost,
)


def _load_dotenv():
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

_load_dotenv()

random.seed(42)
np.random.seed(42)

DOCS_DIR    = Path(__file__).parent / "sources "
RESULTS_DIR = Path(__file__).parent / "results"
CACHE_DIR   = RESULTS_DIR / "cache"
RESULTS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

EMBED_MODEL_NAME   = "all-MiniLM-L6-v2"
_EMBED_MODEL_CACHE = None
AZURE_DEPLOYMENT   = "gpt-4o"
TOP_K = 5

STRATEGIES = [
    {
        "id": "C",
        "name": "sliding_window_512_128",
        "type": "sliding_window",
        "params": {"chunk_size": 512, "overlap": 128},
    },
    {
        "id": "D",
        "name": "sliding_window_512_256",
        "type": "sliding_window",
        "params": {"chunk_size": 512, "overlap": 256},
    },
    {
        "id": "E",
        "name": "structure_aware_512",
        "type": "structure_aware",
        "params": {"max_tokens": 512, "min_tokens": 50},
    },
    {
        "id": "F",
        "name": "semantic_075_512",
        "type": "semantic",
        "params": {"threshold": 0.75, "max_tokens": 512, "embed_model": EMBED_MODEL_NAME},
    },
]

ANSWER_PROMPT = (
    "You are a helpful assistant. Using ONLY the context below, answer the question concisely.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)


def load_all_docs() -> Dict[str, str]:
    docs = {}
    for f in DOCS_DIR.iterdir():
        if f.suffix.lower() == ".docx":
            docs[f.name] = load_docx_text(str(f))
    print(f"Loaded {len(docs)} documents: {list(docs.keys())}")
    return docs


def build_chunks(strategy: Dict, docs: Dict[str, str]) -> List[Dict]:
    chunks = []
    stype  = strategy["type"]
    params = strategy["params"]
    for doc_name, doc_text in docs.items():
        if stype == "sliding_window":
            chunks.extend(sliding_window_chunks(doc_text, chunk_size=params["chunk_size"], overlap=params["overlap"], doc_name=doc_name))
        elif stype == "structure_aware":
            elements = load_docx_structured(str(DOCS_DIR / doc_name))
            chunks.extend(structure_aware_chunks(elements, max_tokens=params["max_tokens"], min_tokens=params["min_tokens"], doc_name=doc_name))
        elif stype == "semantic":
            chunks.extend(semantic_chunks(doc_text, threshold=params["threshold"], max_tokens=params["max_tokens"], embed_model_name=params["embed_model"], doc_name=doc_name))
        else:
            raise ValueError(f"Unknown strategy type: {stype}")
    return chunks


def embed_chunks(chunks: List[Dict], model: SentenceTransformer) -> np.ndarray:
    return model.encode([c["text"] for c in chunks], show_progress_bar=False, normalize_embeddings=True, batch_size=64)


def _cache_key(strategy: Dict, docs: Dict[str, str]) -> str:
    mtimes = {name: os.path.getmtime(str(DOCS_DIR / name)) for name in docs}
    payload = json.dumps({"strategy_id": strategy["id"], "params": strategy["params"], "mtimes": mtimes}, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()


def build_chunks_cached(strategy: Dict, docs: Dict[str, str], embed_model: SentenceTransformer):
    key      = _cache_key(strategy, docs)
    c_chunks = CACHE_DIR / f"{key}_chunks.json"
    c_embs   = CACHE_DIR / f"{key}_embs.npz"

    if c_chunks.exists() and c_embs.exists():
        print("  (cache hit — skipping parse + embed)")
        chunks     = json.loads(c_chunks.read_text(encoding="utf-8"))
        embeddings = np.load(str(c_embs))["embeddings"]
        return chunks, embeddings

    chunks     = build_chunks(strategy, docs)
    embeddings = embed_chunks(chunks, embed_model)
    c_chunks.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(str(c_embs), embeddings=embeddings)
    return chunks, embeddings


def retrieve_top_k(query: str, chunks: List[Dict], embeddings: np.ndarray, model: SentenceTransformer, k: int = 5) -> List[Dict]:
    q_emb = model.encode([query], normalize_embeddings=True)[0]
    scores = embeddings @ q_emb
    return [chunks[i] for i in np.argsort(scores)[::-1][:k]]


def _azure_client(api_key: str) -> AzureOpenAI:
    endpoint = os.getenv("AZURE_API_ENDPOINT", "")
    # Strip the path — AzureOpenAI wants just the base URL
    from urllib.parse import urlparse
    base_url = "{uri.scheme}://{uri.netloc}/".format(uri=urlparse(endpoint))
    api_version = "2025-01-01-preview"
    return AzureOpenAI(azure_endpoint=base_url, api_key=api_key, api_version=api_version)


def generate_answer(api_key: str, question: str, context_chunks: List[Dict]) -> tuple:
    context  = "\n\n---\n\n".join(c["text"] for c in context_chunks)
    messages = [{"role": "user", "content": ANSWER_PROMPT.format(context=context, question=question)}]
    client   = _azure_client(api_key)
    for attempt in range(3):
        try:
            resp   = client.chat.completions.create(model=AZURE_DEPLOYMENT, messages=messages, max_tokens=512, temperature=0)
            answer = resp.choices[0].message.content.strip()
            usage  = resp.usage
            return answer, usage.prompt_tokens, usage.completion_tokens
        except Exception as e:
            wait = 15 if "429" in str(e) else 2 ** attempt
            if attempt < 2:
                print(f"  [generate] retry {attempt+1} (wait {wait}s): {type(e).__name__}: {e}")
                time.sleep(wait)
            else:
                print(f"  [generate] FAILED after 3 attempts: {type(e).__name__}: {e}")
                raise


def run_benchmark(strategies=None, embed_model=None, api_key=None, questions=None):
    if strategies is None:
        strategies = STRATEGIES
    if api_key is None:
        api_key = os.getenv("AZURE_API_KEY")
    if not api_key:
        raise ValueError("AZURE_API_KEY not set. Add it to .env")
    if not questions:
        raise ValueError("No questions to evaluate. Add questions in the Questions page.")

    global _EMBED_MODEL_CACHE
    if embed_model is None:
        if _EMBED_MODEL_CACHE is None:
            print(f"Loading embedding model {EMBED_MODEL_NAME}...")
            _EMBED_MODEL_CACHE = SentenceTransformer(EMBED_MODEL_NAME)
        embed_model = _EMBED_MODEL_CACHE

    docs = load_all_docs()
    all_results = []
    chunk_stats = {}

    for strategy in strategies:
        print(f"\n[{strategy['id']}] {strategy['name']}")
        chunks, embeddings = build_chunks_cached(strategy, docs, embed_model)
        tok_counts = [c["n_tokens"] for c in chunks]
        print(f"  {len(chunks)} chunks, avg {np.mean(tok_counts):.0f} tokens")

        chunk_stats[strategy["id"]] = {
            "n_chunks":   len(chunks),
            "avg_tokens": float(np.mean(tok_counts)),
            "min_tokens": int(min(tok_counts)),
            "max_tokens": int(max(tok_counts)),
            "std_tokens": float(np.std(tok_counts)),
        }

        def _eval_question(q):
            retrieved    = retrieve_top_k(q["question"], chunks, embeddings, embed_model, k=TOP_K)
            context_text = "\n\n".join(c["text"] for c in retrieved)

            answer, tok_in, tok_out = generate_answer(api_key, q["question"], retrieved)
            faith = faithfulness_score(api_key, context_text, answer)
            rel   = relevance_score(api_key, q["question"], answer)
            rec5  = recall_at_k(chunks, retrieved, q, k=TOP_K)
            mrr_v = mrr(retrieved, q)
            f1    = token_f1(answer, q.get("expected", ""))
            cost  = compute_cost(tok_in, tok_out)

            print(f"  {q['id']}  R@5={rec5:.2f} MRR={mrr_v:.2f} F1={f1:.2f} Faith={faith:.2f} Rel={rel:.2f} ${cost:.4f}")
            return {
                "strategy_id":   strategy["id"],
                "strategy_name": strategy["name"],
                "q_id":          q["id"],
                "question":      q["question"],
                "type":          q["type"],
                "difficulty":    q["difficulty"],
                "source_doc":    q["source_doc"],
                "answer":        answer,
                "expected":      q.get("expected", ""),
                "retrieved_chunks": [{"id": c["chunk_id"], "text": c["text"]} for c in retrieved],
                "recall_at_5":   rec5,
                "mrr":           mrr_v,
                "f1":            f1,
                "faithfulness":  faith,
                "relevance":     rel,
                "tokens_input":  tok_in,
                "tokens_output": tok_out,
                "cost_usd":      cost,
            }

        for q in questions:
            all_results.append(_eval_question(q))

    # Save raw results
    raw_path = RESULTS_DIR / "raw_results.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Aggregate per strategy
    summary = []
    costs   = []
    for strategy in strategies:
        sid  = strategy["id"]
        rows = [r for r in all_results if r["strategy_id"] == sid]
        if not rows:
            continue
        entry = {
            "strategy_id":   sid,
            "strategy_name": strategy["name"],
            **chunk_stats[sid],
            "recall_at_5":    float(np.mean([r["recall_at_5"]  for r in rows])),
            "mrr":            float(np.mean([r["mrr"]          for r in rows])),
            "f1":             float(np.mean([r["f1"]           for r in rows])),
            "faithfulness":   float(np.mean([r["faithfulness"] for r in rows])),
            "relevance":      float(np.mean([r["relevance"]    for r in rows])),
            "cost_per_query": float(np.mean([r["cost_usd"]     for r in rows])),
        }
        summary.append(entry)
        costs.append(entry["cost_per_query"])

    max_cost = max(costs) if costs else 1.0
    for s in summary:
        norm_cost = s["cost_per_query"] / max_cost if max_cost > 0 else 0.0
        s["composite_score"] = (
            s["recall_at_5"]  * 0.20
            + s["mrr"]        * 0.15
            + s["f1"]         * 0.15
            + s["faithfulness"] * 0.15
            + s["relevance"]  * 0.10
            + (1 - norm_cost) * 0.10
        )

    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    xlsx_path = RESULTS_DIR / "benchmark_results.xlsx"
    export_excel(all_results, summary, xlsx_path)

    (RESULTS_DIR / "_benchmark_done.txt").write_text("done")

    best = max(summary, key=lambda x: x["composite_score"])
    print(f"\nDone. Best: {best['strategy_id']} ({best['composite_score']:.3f})")


def export_excel(all_results, summary, path):
    wb    = openpyxl.Workbook()
    GREEN = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    BOLD  = Font(bold=True)
    best_sid = max(summary, key=lambda x: x["composite_score"])["strategy_id"]

    ws1 = wb.active
    ws1.title = "Summary"
    headers = ["Strategy", "N_chunks", "Avg_tokens", "Recall@5", "MRR", "F1",
               "Faithfulness", "Relevance", "Cost/query", "Composite_score"]
    ws1.append(headers)
    for c in ws1[1]:
        c.font = BOLD
    for s in summary:
        ws1.append([
            s["strategy_id"], s["n_chunks"], round(s["avg_tokens"], 1),
            round(s["recall_at_5"], 4), round(s["mrr"], 4), round(s["f1"], 4),
            round(s["faithfulness"], 4), round(s["relevance"], 4),
            round(s["cost_per_query"], 6), round(s["composite_score"], 4),
        ])
        if s["strategy_id"] == best_sid:
            for cell in ws1[ws1.max_row]:
                cell.fill = GREEN

    ws2 = wb.create_sheet("Per Question")
    ws2.append(["Strategy", "Q_ID", "Question", "Type", "Difficulty",
                "Recall@5", "MRR", "F1", "Faithfulness", "Relevance", "Cost"])
    for c in ws2[1]:
        c.font = BOLD
    for r in all_results:
        ws2.append([
            r["strategy_id"], r["q_id"], r["question"], r["type"], r["difficulty"],
            round(r["recall_at_5"], 4), round(r["mrr"], 4), round(r["f1"], 4),
            round(r["faithfulness"], 4), round(r["relevance"], 4), round(r["cost_usd"], 6),
        ])

    ws3 = wb.create_sheet("Chunk Stats")
    ws3.append(["Strategy", "n_chunks", "avg_tokens", "min_tokens", "max_tokens", "std_tokens"])
    for c in ws3[1]:
        c.font = BOLD
    for s in summary:
        ws3.append([
            s["strategy_id"], s["n_chunks"], round(s["avg_tokens"], 1),
            s["min_tokens"], s["max_tokens"], round(s["std_tokens"], 1),
        ])

    wb.save(str(path))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategies", nargs="+", default=None,
                        help="Strategy IDs to run, e.g. --strategies C E")
    args = parser.parse_args()
    if args.strategies:
        valid = {s["id"] for s in STRATEGIES}
        unknown = set(args.strategies) - valid
        if unknown:
            raise ValueError(f"Unknown strategy IDs: {unknown}. Valid: {valid}")
        run_benchmark(strategies=[s for s in STRATEGIES if s["id"] in args.strategies])
    else:
        run_benchmark()
