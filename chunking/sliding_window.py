from typing import List, Dict
from chunking.utils import TOKENIZER


def sliding_window_chunks(text: str, chunk_size: int, overlap: int, doc_name: str) -> List[Dict]:
    tokens = TOKENIZER.encode(text)
    chunks = []
    start, idx = 0, 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append({
            "chunk_id": f"{doc_name}::sw_{idx}",
            "text": TOKENIZER.decode(chunk_tokens),
            "source_doc": doc_name,
            "n_tokens": len(chunk_tokens),
        })
        if end >= len(tokens):
            break
        start = end - overlap
        idx += 1

    return chunks
