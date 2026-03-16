from abc import ABC, abstractmethod
from typing import List, Dict


class BaseParser(ABC):
    """
    Contract every parser must satisfy.

    parse_flat(filepath)      → plain text string (used by sliding_window + semantic)
    parse_structured(filepath) → list of {text, style, is_heading} dicts (used by structure_aware)
    """

    @abstractmethod
    def parse_flat(self, filepath: str) -> str:
        ...

    @abstractmethod
    def parse_structured(self, filepath: str) -> List[Dict]:
        ...
