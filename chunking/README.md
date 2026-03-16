# chunking/

Takes parsed text (from `parsing/`) and splits it into chunks ready for embedding and retrieval.

## Files

**utils.py** — shared tokenizer (tiktoken cl100k_base) and `count_tokens(text)`

**loaders.py** — thin wrappers that call the right parser and return text/elements. This is the interface `run_benchmark.py` calls — you shouldn't need to touch it.

**sliding_window.py** — strategies C and D. Token-based fixed-size windows with configurable overlap.

**structure_aware.py** — strategy E. Splits on document headings and merges small paragraphs up to a token limit. No embeddings needed at ingestion time.

**semantic.py** — strategy F. Embeds each sentence and cuts where cosine similarity drops below a threshold. Produces thematically coherent chunks.

## Adding a new strategy

1. Create a new file here (e.g. `sentence_window.py`)
2. Add a function that returns a list of `{"chunk_id", "text", "source_doc", "n_tokens"}` dicts
3. Export it from `__init__.py`
4. Add an entry to `STRATEGIES` in `run_benchmark.py` with type and params

The benchmark engine will handle everything else.

## Planned integrations

- LlamaIndex `SentenceWindowNodeParser`
- LlamaIndex `HierarchicalNodeParser`
- Custom recursive splitters
