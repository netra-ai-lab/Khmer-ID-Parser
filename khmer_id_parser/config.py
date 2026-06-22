"""
config.py — constants and the ModelConfig dataclass.

Config is data, not code: every architecture/asset constant lives here so nothing
hard-codes a hyperparameter or a filename elsewhere. ``ModelConfig`` mirrors the
``config`` dict that ``train.py`` writes into each checkpoint.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path

# ── Paths & asset filenames ─────────────────────────────────────────────────────
PACKAGE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = PACKAGE_DIR / "assets"

DEFAULT_CHECKPOINT_NAME = "khmer_id_parser_v2.pth"
CHAR_VOCAB_FILENAME = "char_to_idx.json"
TAG_VOCAB_FILENAME = "tag_to_idx.json"

# ── Deploy-time overrides (read by weights.resolve / parser) ─────────────────────
ENV_CHECKPOINT = "KHMER_ID_PARSER_CHECKPOINT"
ENV_DEVICE = "KHMER_ID_PARSER_DEVICE"

# ── Inference constants ─────────────────────────────────────────────────────────
MAX_LEN = 1024  # sequence-length cap, mirrors scripts/dataset.py

# The 17 entity types the model tags (BIO scheme adds B-/I- prefixes per type).
ENTITY_LABELS = [
    "ID_NUM", "NAME_KH", "NAME_EN", "DOB", "GENDER", "HEIGHT",
    "POB_COMM", "POB_DIST", "POB_PROV",
    "ADDR_VILL", "ADDR_COMM", "ADDR_DIST", "ADDR_PROV",
    "ISSUE_DATE", "EXP_DATE", "MARKS", "MRZ",
]

# Shorthand keys some older checkpoints used → KhmerIDParserModel parameter names.
_CONFIG_ALIASES = {
    "cnn_kernels": "cnn_kernel_sizes",
    "lstm_hidden": "lstm_hidden_dim",
}


@dataclass
class ModelConfig:
    """Architecture config — mirrors the ``config`` dict saved inside the checkpoint."""

    vocab_size: int
    num_tags: int
    embed_dim: int = 64
    cnn_filters: int = 64
    cnn_kernel_sizes: list = field(default_factory=lambda: [3, 5, 7])
    lstm_hidden_dim: int = 256
    lstm_layers: int = 2
    dropout: float = 0.3
    pad_idx: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        """Build from a checkpoint config dict, tolerant of aliases and unknown keys."""
        d = dict(d)
        for saved_key, model_key in _CONFIG_ALIASES.items():
            if saved_key in d and model_key not in d:
                d[model_key] = d.pop(saved_key)
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_kwargs(self) -> dict:
        """Keyword args for ``KhmerIDParserModel(**...)``."""
        return asdict(self)
