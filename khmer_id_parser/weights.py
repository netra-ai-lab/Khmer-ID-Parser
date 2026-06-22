"""
weights.py — one resolver, one loader.

``resolve()`` decides *which* checkpoint to load (explicit path → env override →
bundled default), and ``load_checkpoint()`` turns that path into a ready-to-use model
plus its vocabularies. Every surface gets the same behaviour for free.
"""

import json
import os
from pathlib import Path
from typing import Optional, Union

import torch

from .config import (
    ASSETS_DIR,
    CHAR_VOCAB_FILENAME,
    DEFAULT_CHECKPOINT_NAME,
    ENV_CHECKPOINT,
    TAG_VOCAB_FILENAME,
    ModelConfig,
)


def resolve(checkpoint: Optional[Union[str, Path]] = None) -> Path:
    """
    Resolve the checkpoint path from three sources, in priority order:

      1. an explicit ``checkpoint`` argument,
      2. the ``$KHMER_ID_PARSER_CHECKPOINT`` environment variable,
      3. the weights bundled inside the package (``assets/khmer_id_parser_v2.pth``).

    Raises ``FileNotFoundError`` with a clear message if the chosen path is missing.
    """
    candidate = checkpoint or os.environ.get(ENV_CHECKPOINT) or (ASSETS_DIR / DEFAULT_CHECKPOINT_NAME)
    path = Path(candidate)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at '{path}'.\n"
            f"       Pass a path explicitly, set ${ENV_CHECKPOINT}, "
            f"or reinstall the package to restore the bundled weights."
        )
    return path


def _load_vocab(filename: str) -> dict:
    return json.loads((ASSETS_DIR / filename).read_text(encoding="utf-8"))


def load_checkpoint(checkpoint_path: Union[str, Path], device: torch.device):
    """
    Load a checkpoint into a ready-for-inference model.

    Returns ``(model, char2idx, tag2idx)``. Supports both the full bundle saved by
    ``scripts/train.py`` (weights + vocab + config) and a legacy state-dict-only
    checkpoint, in which case the vocabularies bundled in the wheel are used.
    """
    # Avoid a top-level torch.nn import cycle; model only needed here.
    from .model import KhmerIDParserModel

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "config" in ckpt:
        config   = ModelConfig.from_dict(ckpt["config"])
        char2idx = ckpt["char2idx"]
        tag2idx  = ckpt["tag2idx"]
        state    = ckpt["model_state"]
    else:
        # Legacy: bare state dict — fall back to the vocab bundled in the package.
        char2idx = _load_vocab(CHAR_VOCAB_FILENAME)
        tag2idx  = _load_vocab(TAG_VOCAB_FILENAME)
        config   = ModelConfig(vocab_size=len(char2idx), num_tags=len(tag2idx))
        state    = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt

    # Inference always runs without dropout, with pad_idx tied to the vocab.
    config.dropout = 0.0
    config.pad_idx = char2idx.get("<PAD>", 0)

    model = KhmerIDParserModel(**config.to_kwargs())
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return model, char2idx, tag2idx
