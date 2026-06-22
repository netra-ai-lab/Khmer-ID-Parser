"""
postprocess.py — turn a BIO tag sequence into structured ID-card fields.

Shared by every surface: the parser produces per-character tags, these helpers
collect entity spans and map them to the final output schema.
"""

from typing import Optional


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


def spans_to_structured(spans: dict[str, list[str]]) -> dict:
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
