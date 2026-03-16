"""
Metric computation for the RAG benchmark.
"""

import re
import json
import time
from collections import Counter
from typing import List, Dict, Optional

from openai import OpenAI


# ─── Source doc normalisation ─────────────────────────────────────────────────

def _norm_doc(name: str) -> str:
    """Strip .docx extension and all non-alphanumeric chars, lowercase."""
    name = re.sub(r"\.docx$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"[^a-z0-9]", "", name.lower())
    return name


def is_relevant_chunk(chunk: Dict, question: Dict) -> bool:
    """A chunk is relevant if source matches AND ≥2 keywords appear in text."""
    if _norm_doc(chunk["source_doc"]) != _norm_doc(question["source_doc"]):
        return False
    text_lower = chunk["text"].lower()
    matches = sum(1 for kw in question["keywords"] if kw.lower() in text_lower)
    return matches >= 2


# ─── Retrieval metrics ────────────────────────────────────────────────────────

def recall_at_k(
    all_chunks: List[Dict],
    retrieved: List[Dict],
    question: Dict,
    k: int = 5,
) -> float:
    total_relevant = sum(1 for c in all_chunks if is_relevant_chunk(c, question))
    if total_relevant == 0:
        return 0.0
    relevant_retrieved = sum(1 for c in retrieved[:k] if is_relevant_chunk(c, question))
    return relevant_retrieved / total_relevant


def mrr(retrieved: List[Dict], question: Dict) -> float:
    for rank, chunk in enumerate(retrieved, start=1):
        if is_relevant_chunk(chunk, question):
            return 1.0 / rank
    return 0.0


# ─── Token F1 ────────────────────────────────────────────────────────────────

def _tokenize_f1(text: str) -> List[str]:
    """Lowercase + strip punctuation → word tokens."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _tokenize_f1(prediction)
    ref_tokens = _tokenize_f1(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_count = Counter(pred_tokens)
    ref_count = Counter(ref_tokens)
    common = sum((pred_count & ref_count).values())
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ─── LLM-as-judge ─────────────────────────────────────────────────────────────

FAITHFULNESS_PROMPT = (
    'Score 0-5: does the answer rely only on the context? '
    'Respond JSON {{"score": <int>, "reason": "<str>"}}\n\n'
    'Context:\n{context}\n\nAnswer:\n{answer}'
)

RELEVANCE_PROMPT = (
    'Score 0-5: is the answer relevant to the question? '
    'Respond JSON {{"score": <int>, "reason": "<str>"}}\n\n'
    'Question:\n{question}\n\nAnswer:\n{answer}'
)


def _call_judge(client: OpenAI, prompt: str, retries: int = 3) -> Optional[Dict]:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model="meta-llama/llama-3.1-8b-instruct:free",
                max_tokens=256,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.choices[0].message.content.strip()
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                return json.loads(m.group())
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  [judge] FAILED after {retries} attempts: {type(e).__name__}: {e}")
    return None


def faithfulness_score(client: OpenAI, context: str, answer: str) -> float:
    prompt = FAITHFULNESS_PROMPT.format(context=context[:3000], answer=answer)
    result = _call_judge(client, prompt)
    if result and "score" in result:
        return min(max(int(result["score"]), 0), 5) / 5.0
    return 0.0


def relevance_score(client: OpenAI, question: str, answer: str) -> float:
    prompt = RELEVANCE_PROMPT.format(question=question, answer=answer)
    result = _call_judge(client, prompt)
    if result and "score" in result:
        return min(max(int(result["score"]), 0), 5) / 5.0
    return 0.0


# ─── Cost formula ─────────────────────────────────────────────────────────────

def compute_cost(tokens_input: int, tokens_output: int) -> float:
    return (tokens_input * 0.003 + tokens_output * 0.015) / 1000
