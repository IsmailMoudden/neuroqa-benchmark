from parsing.base import BaseParser
from parsing.docx_parser import DocxParser

# Registry: file extension → parser class
# To add a new parser: implement BaseParser, register it here.
PARSERS = {
    ".docx": DocxParser,
}


def get_parser(filepath: str) -> BaseParser:
    from pathlib import Path
    ext = Path(filepath).suffix.lower()
    cls = PARSERS.get(ext)
    if cls is None:
        raise ValueError(f"No parser registered for '{ext}'. Add one to parsing/__init__.py.")
    return cls()
