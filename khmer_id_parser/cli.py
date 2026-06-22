"""
cli.py — Surface 2: command-line interface.

A thin adapter over ``KhmerIDParser``. Wired as the ``khmer-id-parse`` console script.

  khmer-id-parse --text "101105287\\nគោន្តនាម..."
  khmer-id-parse --file card.txt
  echo "101105287..." | khmer-id-parse --stdin
  khmer-id-parse --file card.txt --format tags     # raw BIO output
"""

import argparse
import json
import sys
from pathlib import Path

from .parser import KhmerIDParser
from .weights import resolve


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="khmer-id-parse",
        description="Khmer ID card OCR parser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parse a string directly (use \\n for newlines in shell)
  khmer-id-parse --text $'101105287\\nគោន្តនាមនិងនាម...'

  # Parse from a plain-text file
  khmer-id-parse --file card.txt

  # Pipe OCR output from another tool
  tesseract card.png stdout | khmer-id-parse --stdin

  # Pretty-print structured JSON (default)
  khmer-id-parse --file card.txt --format json

  # Raw BIO tags alongside each character (useful for debugging)
  khmer-id-parse --file card.txt --format tags

  # Compact single-line JSON (useful for piping into jq)
  khmer-id-parse --file card.txt --format compact
        """,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text",  type=str, help="Raw OCR text as a string")
    src.add_argument("--file",  type=str, help="Path to a UTF-8 text file")
    src.add_argument("--stdin", action="store_true", help="Read from stdin")

    p.add_argument(
        "--checkpoint", "-c",
        default=None,
        help="Path to the .pth checkpoint "
             "(default: $KHMER_ID_PARSER_CHECKPOINT or the bundled weights)",
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


def _force_utf8_streams() -> None:
    """
    Make stdin/stdout/stderr UTF-8 regardless of the OS locale.

    Without this, Khmer output fails to print on Windows (cp1252) and piped UTF-8
    input is misdecoded. Reconfiguring before any read/write fixes both.
    """
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except (ValueError, OSError):
                pass


def main() -> None:
    _force_utf8_streams()
    args = _build_arg_parser().parse_args()

    # ── Load input text ────────────────────────────────────────────────────────
    if args.text:
        # Allow \n in --text argument (shell won't expand it inside double quotes)
        text = args.text.replace("\\n", "\n")
    elif args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()

    text = text.lstrip("﻿")   # drop a leading BOM if the source had one

    if not text.strip():
        sys.exit("Error: input text is empty.")

    # ── Load model (resolve raises a clear error if no checkpoint is found) ──────
    try:
        resolve(args.checkpoint)
    except FileNotFoundError as e:
        sys.exit(f"Error: {e}")

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
        output = "\n".join(_format_tags(text, tags))
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
