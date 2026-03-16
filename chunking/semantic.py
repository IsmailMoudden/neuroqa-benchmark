import re
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer

from chunking.utils import count_tokens

_model_cache: Dict[str, SentenceTransformer] = {}


def _get_model(name: str) -> SentenceTransformer:
    if name not in _model_cache:
        _model_cache[name] = SentenceTransformer(name)
    return _model_cache[name]


def semantic_chunks(text: str, threshold: float, max_tokens: int, embed_model_name: str, doc_name: str) -> List[Dict]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return []

    model = _get_model(embed_model_name)
    embeddings = model.encode(sentences, show_progress_bar=False, normalize_embeddings=True)

    chunks = []
    current_sents = [sentences[0]]
    current_tokens = count_tokens(sentences[0])
    idx = 0

    for i in range(1, len(sentences)):
        sim = float(np.dot(embeddings[i], embeddings[i - 1]))
        sent_tokens = count_tokens(sentences[i])

        if sim < threshold or current_tokens + sent_tokens > max_tokens:
            chunks.append({
                "chunk_id": f"{doc_name}::sem_{idx}",
                "text": " ".join(current_sents),
                "source_doc": doc_name,
                "n_tokens": current_tokens,
            })
            idx += 1
            current_sents = [sentences[i]]
            current_tokens = sent_tokens
        else:
            current_sents.append(sentences[i])
            current_tokens += sent_tokens

    if current_sents:
        chunks.append({
            "chunk_id": f"{doc_name}::sem_{idx}",
            "text": " ".join(current_sents),
            "source_doc": doc_name,
            "n_tokens": current_tokens,
        })

    return chunks
