from typing import List, Dict
from parsing import get_parser


def load_docx_text(filepath: str) -> str:
    return get_parser(filepath).parse_flat(filepath)


def load_docx_structured(filepath: str) -> List[Dict]:
    return get_parser(filepath).parse_structured(filepath)
