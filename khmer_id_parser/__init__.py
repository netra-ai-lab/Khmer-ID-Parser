"""
khmer_id_parser — extract structured fields from OCR text of Cambodian ID cards.

Public surface:
    from khmer_id_parser import KhmerIDParser, parse
    parser = KhmerIDParser()          # bundled weights, auto device
    parser.parse(ocr_text)            # → structured dict
    parse(ocr_text)                   # one-shot, caches a default parser
"""

from .config import ENTITY_LABELS, ModelConfig
from .model import KhmerIDParserModel
from .parser import KhmerIDParser, parse

__version__ = "0.1.0"
__all__ = [
    "KhmerIDParser",
    "parse",
    "KhmerIDParserModel",
    "ModelConfig",
    "ENTITY_LABELS",
    "__version__",
]
