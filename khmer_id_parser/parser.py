"""
parser.py — ★ the inference core.

``KhmerIDParser`` loads the model + vocabularies once and exposes *the* inference
methods. Every surface (Python API, CLI) is a thin adapter over this class. A
module-level ``parse()`` one-shot helper caches a default instance.
"""

import os
from typing import Optional

import torch

from .config import ENV_DEVICE
from .postprocess import _extract_spans, spans_to_structured
from .weights import load_checkpoint, resolve


class KhmerIDParser:
    """
    Programmatic interface for parsing Khmer ID card OCR text.

    Usage:
        parser = KhmerIDParser()                      # bundled weights, auto device
        parser = KhmerIDParser("custom.pth", "cpu")   # explicit checkpoint + device
        result = parser.parse(text)                   # → structured dict
        tags   = parser.tag(text)                     # → list of BIO tag strings
        spans  = parser.extract_spans(text)           # → {entity: [values]}
    """

    def __init__(self, checkpoint: Optional[str] = None, device: Optional[str] = None):
        if device is None:
            device = os.environ.get(ENV_DEVICE) or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

        checkpoint_path = resolve(checkpoint)
        self.model, self.char2idx, self.tag2idx = load_checkpoint(checkpoint_path, self.device)

        self.idx2tag = {v: k for k, v in self.tag2idx.items()}
        self.pad_idx = self.char2idx.get("<PAD>", 0)
        self.unk_idx = self.char2idx.get("<UNK>", 1)

    def _encode(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode text → (input_ids, mask) tensors of shape (1, seq_len)."""
        ids  = [self.char2idx.get(c, self.unk_idx) for c in text]
        t    = torch.tensor([ids], dtype=torch.long, device=self.device)
        mask = (t != self.pad_idx)
        return t, mask

    def tag(self, text: str) -> list[str]:
        """
        Run the model and return a BIO tag for every character in *text*.

        Returns:
            List[str] — same length as text, e.g. ['B-ID_NUM', 'I-ID_NUM', ...]
        """
        with torch.no_grad():
            input_ids, mask = self._encode(text)
            pred_ids = self.model.decode(input_ids, mask)[0]  # first (only) batch item
        return [self.idx2tag.get(i, "O") for i in pred_ids]

    def extract_spans(self, text: str) -> dict[str, list[str]]:
        """
        Run the model and return raw entity spans.

        Returns:
            {"ID_NUM": ["443322563"], "NAME_KH": ["រ៉ា ចរិយា"], "MRZ": [...], ...}
        """
        tags = self.tag(text)
        return _extract_spans(text, tags)

    def parse(self, text: str) -> dict:
        """
        Full pipeline: raw OCR text → structured dict.

        Returns:
            {
              "id_number":   str,
              "name_khmer":  str,
              "name_latin":  str,
              "date_of_birth": str,       # ASCII digits, DD.MM.YYYY
              "gender":      str,
              "height_cm":   int,
              "place_of_birth": {"commune", "district", "province"},
              "address":        {"village", "commune", "district", "province"},
              "validity":       {"issue_date", "expiry_date"},
              "distinguishing_marks": str,
              "mrz": [line1, line2, line3],
            }
        """
        spans = self.extract_spans(text)
        return spans_to_structured(spans)

    def parse_batch(self, texts: list[str]) -> list[dict]:
        """
        Parse multiple OCR strings in one call.
        Each string is processed independently (no padding between items).
        """
        return [self.parse(t) for t in texts]


# ── Module-level one-shot helper (caches a default instance) ────────────────────

_DEFAULT: Optional[KhmerIDParser] = None


def parse(text: str, checkpoint: Optional[str] = None, device: Optional[str] = None) -> dict:
    """
    One-shot parse using a cached default ``KhmerIDParser``.

    The first call constructs the parser (loading the bundled weights unless
    *checkpoint* is given); later calls reuse it.
    """
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = KhmerIDParser(checkpoint=checkpoint, device=device)
    return _DEFAULT.parse(text)
