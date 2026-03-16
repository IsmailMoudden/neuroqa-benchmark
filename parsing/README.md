# parsing/

Responsible for reading source files and turning them into text the chunking layer can work with.

Every parser exposes two methods:
- `parse_flat` — returns a single plain text string (used by sliding window and semantic chunking)
- `parse_structured` — returns a list of `{text, style, is_heading}` dicts (used by structure-aware chunking)

## Current parsers

**DocxParser** (`docx_parser.py`) — reads `.docx` files using python-docx. Extracts paragraphs and tables.

## Adding a new parser

Create a class that extends `BaseParser` from `base.py` and implement both methods. Then register it in `__init__.py`:

```python
PARSERS = {
    ".docx": DocxParser,
    ".pdf":  YourPdfParser,   # add here
}
```

`get_parser(filepath)` will automatically pick the right one based on file extension. Nothing else needs to change.

## Planned integrations

- LlamaIndex document readers (PDF, HTML, Notion, etc.)
- Unstructured.io for mixed-format documents
