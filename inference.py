"""
inference.py — Khmer ID Card Parser

Supports two usage modes:

  1. Command-line:
       python inference.py --text "101105287\nគោន្តនាម..."
       python inference.py --file card.txt
       echo "101105287..." | python inference.py --stdin
       python inference.py --text "..." --format tags   # raw BIO output

  2. Programmatic (import as a module):
       from inference import KhmerIDParser
       parser = KhmerIDParser("khmer_id_parser_v2.pth")
       result = parser.parse("101105287\nគោន្តនាម...")
       # result is a dict with all extracted fields
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
from model import KhmerIDParserModel


# ── Checkpoint loader ──────────────────────────────────────────────────────────

def _load_checkpoint(checkpoint_path: str, device: torch.device):
    """
    Loads the checkpoint saved by train.py.

    Expected format (saved at end of train.py):
        torch.save({
            "model_state": model.state_dict(),
            "char2idx":    char_vocab.char2idx,
            "tag2idx":     tag_vocab.tag2idx,
            "config": { vocab_size, num_tags, embed_dim, ... }
        }, path)
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Support both the full bundle (recommended) and the legacy state-dict-only format
    if "config" in ckpt:
        config    = ckpt["config"]
        char2idx  = ckpt["char2idx"]
        tag2idx   = ckpt["tag2idx"]
        # Normalise shorthand key names from train.py to match
        # KhmerIDParserModel.__init__ parameter names exactly.
        _aliases = {
            "cnn_kernels": "cnn_kernel_sizes",
            "lstm_hidden": "lstm_hidden_dim",
        }
        for saved_key, model_key in _aliases.items():
            if saved_key in config and model_key not in config:
                config[model_key] = config.pop(saved_key)
    else:
        # Legacy: user saved separate vocab JSONs alongside the .pth
        pth_dir   = Path(checkpoint_path).parent
        char2idx  = json.loads((pth_dir / "char_to_idx.json").read_text(encoding="utf-8"))
        tag2idx   = json.loads((pth_dir / "tag_to_idx.json").read_text(encoding="utf-8"))
        config = {
            "vocab_size":  len(char2idx),
            "num_tags":    len(tag2idx),
            "embed_dim":   64,
            "cnn_filters": 64,
            "cnn_kernel_sizes": [2, 3, 4],
            "lstm_hidden_dim":  256,
            "lstm_layers": 2,
            "dropout":     0.0,          # no dropout at inference
            "pad_idx":     char2idx.get("<PAD>", 0),
        }

    config["dropout"] = 0.0             # always disable dropout at inference
    config["pad_idx"] = char2idx.get("<PAD>", 0)

    model = KhmerIDParserModel(**config)
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
    model.to(device)
    model.eval()

    return model, char2idx, tag2idx


# ── Tag-sequence → entity spans ───────────────────────────────────────────────

def _extract_spans(text: str, tags: list[str]) -> dict[str, list[str]]:
    """
    Walk BIO tags and collect the text value for each entity.
    Returns {entity_type: [value, ...]}  (most entities have one value; MRZ has three).
    """
    spans: dict[str, list[str]] = {}
    chars = list(text)
    i = 0
    while i < len(tags):
        tag = tags[i]
        if tag.startswith("B-"):
            entity = tag[2:]
            j = i + 1
            while j < len(tags) and tags[j] == f"I-{entity}":
                j += 1
            value = "".join(chars[i:j]).strip()
            spans.setdefault(entity, []).append(value)
            i = j
        else:
            i += 1
    return spans


def _first(spans: dict, key: str) -> Optional[str]:
    vals = spans.get(key, [])
    return vals[0] if vals else None


def _normalise_date(raw: Optional[str]) -> Optional[str]:
    """Convert Khmer-digit date string to ASCII digits."""
    if not raw:
        return None
    return raw.translate(str.maketrans("០១២៣៤៥៦៧៨៩", "0123456789"))


def _parse_height(raw: Optional[str]) -> Optional[int]:
    if not raw:
        return None
    digits = "".join(
        str("០១២៣៤៥៦៧៨៩".index(c)) if c in "០១២៣៤៥៦៧៨៩" else c
        for c in raw
        if c in "0123456789០១២៣៤៥៦៧៨៩"
    )
    return int(digits) if digits else None


def _spans_to_structured(spans: dict[str, list[str]]) -> dict:
    """Map extracted spans to the final structured output schema."""
    return {
        "id_number":   _first(spans, "ID_NUM"),
        "name_khmer":  _first(spans, "NAME_KH"),
        "name_latin":  _first(spans, "NAME_EN"),
        "date_of_birth": _normalise_date(_first(spans, "DOB")),
        "gender":      _first(spans, "GENDER"),
        "height_cm":   _parse_height(_first(spans, "HEIGHT")),
        "place_of_birth": {
            "commune":  _first(spans, "POB_COMM"),
            "district": _first(spans, "POB_DIST"),
            "province": _first(spans, "POB_PROV"),
        },
        "address": {
            "village":  _first(spans, "ADDR_VILL"),
            "commune":  _first(spans, "ADDR_COMM"),
            "district": _first(spans, "ADDR_DIST"),
            "province": _first(spans, "ADDR_PROV"),
        },
        "validity": {
            "issue_date":  _first(spans, "ISSUE_DATE"),
            "expiry_date": _first(spans, "EXP_DATE"),
        },
        "distinguishing_marks": _first(spans, "MARKS"),
        "mrz": spans.get("MRZ", []),
    }


# ── Public API class ───────────────────────────────────────────────────────────

class KhmerIDParser:
    """
    Programmatic interface for parsing Khmer ID card OCR text.

    Usage:
        parser = KhmerIDParser("khmer_id_parser_v2.pth")
        result = parser.parse(text)             # → structured dict
        tags   = parser.tag(text)               # → list of BIO tag strings
        spans  = parser.extract_spans(text)     # → {entity: [values]}
    """

    def __init__(self, checkpoint: str, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device    = torch.device(device)
        self.model, self.char2idx, self.tag2idx = _load_checkpoint(checkpoint, self.device)
        self.idx2tag   = {v: k for k, v in self.tag2idx.items()}
        self.pad_idx   = self.char2idx.get("<PAD>", 0)
        self.unk_idx   = self.char2idx.get("<UNK>", 1)

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
        return _spans_to_structured(spans)

    def parse_batch(self, texts: list[str]) -> list[dict]:
        """
        Parse multiple OCR strings in one call.
        Each string is processed independently (no padding between items).
        """
        return [self.parse(t) for t in texts]


# ── CLI entry point ────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Khmer ID card OCR parser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parse a string directly (use \\n for newlines in shell)
  python inference.py --text $'101105287\\nគោន្តនាមនិងនាម...'

  # Parse from a plain-text file
  python inference.py --file card.txt

  # Pipe OCR output from another tool
  tesseract card.png stdout | python inference.py --stdin

  # Pretty-print structured JSON (default)
  python inference.py --file card.txt --format json

  # Raw BIO tags alongside each character (useful for debugging)
  python inference.py --file card.txt --format tags

  # Compact single-line JSON (useful for piping into jq)
  python inference.py --file card.txt --format compact
        """,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text",  type=str, help="Raw OCR text as a string")
    src.add_argument("--file",  type=str, help="Path to a UTF-8 text file")
    src.add_argument("--stdin", action="store_true", help="Read from stdin")

    p.add_argument(
        "--checkpoint", "-c",
        default="./checkpoints/khmer_id_parser_v2.pth",
        help="Path to the .pth checkpoint (default: khmer_id_parser_v2.pth)",
    )
    p.add_argument(
        "--format", "-f",
        choices=["json", "compact", "tags"],
        default="json",
        help="Output format: json (pretty), compact (one line), tags (BIO alignment)",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Force device: 'cpu' or 'cuda' (default: auto-detect)",
    )
    p.add_argument(
        "--output", "-o",
        default=None,
        help="Save output to this file instead of printing to stdout. "
             "Extension sets format: .json (pretty), .jsonl (compact), .txt (tags).",
    )
    return p


def _format_tags(text: str, tags: list[str]) -> list[str]:
    """Return character-level BIO alignment as a list of lines."""
    lines = [f"{'CHAR':>6}  TAG", f"{'─'*6}  {'─'*24}"]
    for ch, tag in zip(text, tags):
        display = repr(ch)[1:-1]
        lines.append(f"{display:>6}  {tag}")
    return lines


def _print_tags(text: str, tags: list[str]) -> None:
    """Print character-level BIO alignment to stdout."""
    print("\n".join(_format_tags(text, tags)))


def main() -> None:
    args = _build_arg_parser().parse_args()

    # ── Load input text ────────────────────────────────────────────────────────
    if args.text:
        # Allow \n in --text argument (shell won't expand it inside double quotes)
        text = args.text.replace("\\n", "\n")
    elif args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()

    if not text.strip():
        sys.exit("Error: input text is empty.")

    # ── Load model ─────────────────────────────────────────────────────────────
    if not Path(args.checkpoint).exists():
        sys.exit(f"Error: checkpoint not found at '{args.checkpoint}'.\n"
                 f"       Use --checkpoint to specify the correct path.")

    parser = KhmerIDParser(args.checkpoint, device=args.device)

    # ── Run ───────────────────────────────────────────────────────────────────
    # Infer format from output file extension when --output is given and
    # --format was not explicitly set (i.e. still at its default "json").
    out_path = Path(args.output) if args.output else None
    if out_path and args.format == "json":
        ext = out_path.suffix.lower()
        if ext == ".jsonl":
            args.format = "compact"
        elif ext == ".txt":
            args.format = "tags"

    if args.format == "tags":
        tags = parser.tag(text)
        lines = _format_tags(text, tags)
        output = "\n".join(lines)
    else:
        result = parser.parse(text)
        if args.format == "compact":
            output = json.dumps(result, ensure_ascii=False)
        else:
            output = json.dumps(result, ensure_ascii=False, indent=2)

    # ── Print or save ──────────────────────────────────────────────────────────
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
        print(f"Output saved to {out_path}")
    else:
        print(output)


if __name__ == "__main__":
    main()