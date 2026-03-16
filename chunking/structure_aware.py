from typing import List, Dict
from chunking.utils import count_tokens


def structure_aware_chunks(elements: List[Dict], max_tokens: int, min_tokens: int, doc_name: str) -> List[Dict]:
    chunks = []
    current_texts: List[str] = []
    current_tokens = 0
    idx = 0

    def flush():
        nonlocal idx, current_texts, current_tokens
        if not current_texts:
            return
        chunks.append({
            "chunk_id": f"{doc_name}::sa_{idx}",
            "text": "\n\n".join(current_texts),
            "source_doc": doc_name,
            "n_tokens": current_tokens,
        })
        idx += 1
        current_texts = []
        current_tokens = 0

    for elem in elements:
        elem_tokens = count_tokens(elem["text"])
        if elem["is_heading"] and current_tokens >= min_tokens:
            flush()
        if current_tokens + elem_tokens > max_tokens and current_tokens >= min_tokens:
            flush()
        current_texts.append(elem["text"])
        current_tokens += elem_tokens

    flush()
    return chunks
