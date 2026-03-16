import os
import re
import time
from collections import Counter
from typing import List, Dict, Optional
from urllib.parse import urlparse

from openai import AzureOpenAI

AZURE_DEPLOYMENT = "gpt-4o"

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


def _norm_doc(name: str) -> str:
    name = re.sub(r"\.docx$", "", name, flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]", "", name.lower())


def is_relevant_chunk(chunk: Dict, question: Dict) -> bool:
    if _norm_doc(chunk["source_doc"]) != _norm_doc(question["source_doc"]):
        return False
    text_lower = chunk["text"].lower()
    return sum(1 for kw in question["keywords"] if kw.lower() in text_lower) >= 2


def recall_at_k(all_chunks: List[Dict], retrieved: List[Dict], question: Dict, k: int = 5) -> float:
    total = sum(1 for c in all_chunks if is_relevant_chunk(c, question))
    if total == 0:
        return 0.0
    hits = sum(1 for c in retrieved[:k] if is_relevant_chunk(c, question))
    return hits / total


def mrr(retrieved: List[Dict], question: Dict) -> float:
    for rank, chunk in enumerate(retrieved, start=1):
        if is_relevant_chunk(chunk, question):
            return 1.0 / rank
    return 0.0


def token_f1(prediction: str, reference: str) -> float:
    def tokenize(text):
        return re.sub(r"[^\w\s]", " ", text.lower()).split()

    pred = tokenize(prediction)
    ref  = tokenize(reference)
    if not pred or not ref:
        return 0.0
    common = sum((Counter(pred) & Counter(ref)).values())
    p = common / len(pred)
    r = common / len(ref)
    return 2 * p * r / (p + r) if p + r else 0.0


def _azure_client(api_key: str) -> AzureOpenAI:
    endpoint = os.getenv("AZURE_API_ENDPOINT", "")
    base_url = "{uri.scheme}://{uri.netloc}/".format(uri=urlparse(endpoint))
    return AzureOpenAI(azure_endpoint=base_url, api_key=api_key, api_version="2025-01-01-preview")


def _call_judge(api_key: str, prompt: str, retries: int = 3) -> Optional[Dict]:
    client = _azure_client(api_key)
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=AZURE_DEPLOYMENT,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0,
            )
            raw = resp.choices[0].message.content.strip()
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                import json
                return json.loads(m.group())
        except Exception as e:
            wait = 15 if "429" in str(e) else 2 ** attempt
            if attempt < retries - 1:
                time.sleep(wait)
            else:
                print(f"  [judge] FAILED after {retries} attempts: {type(e).__name__}: {e}")
    return None


def faithfulness_score(api_key: str, context: str, answer: str) -> float:
    result = _call_judge(api_key, FAITHFULNESS_PROMPT.format(context=context[:3000], answer=answer))
    if result and "score" in result:
        return min(max(int(result["score"]), 0), 5) / 5.0
    return 0.0


def relevance_score(api_key: str, question: str, answer: str) -> float:
    result = _call_judge(api_key, RELEVANCE_PROMPT.format(question=question, answer=answer))
    if result and "score" in result:
        return min(max(int(result["score"]), 0), 5) / 5.0
    return 0.0


def compute_cost(tokens_input: int, tokens_output: int) -> float:
    # gpt-4o pricing: $2.50/1M input, $10/1M output
    return (tokens_input * 2.50 + tokens_output * 10.0) / 1_000_000
