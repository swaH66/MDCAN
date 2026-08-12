"""Checkpoint loading helpers for MDCAN."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn


def extract_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    """Extract a state dict from common checkpoint layouts."""
    if isinstance(checkpoint, Mapping):
        for key in ("model", "state_dict", "model_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return value
        if checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
            return checkpoint
    raise ValueError("Checkpoint does not contain a recognizable model state dict.")


def normalize_state_dict_keys(state_dict: Mapping[str, torch.Tensor]) -> OrderedDict:
    """Remove wrappers commonly added by DataParallel or torch.compile."""
    normalized = OrderedDict()
    for key, value in state_dict.items():
        while key.startswith("module.") or key.startswith("_orig_mod."):
            key = key.split(".", 1)[1]
        normalized[key] = value
    return normalized


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load MDCAN weights and return checkpoint metadata."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = normalize_state_dict_keys(extract_state_dict(checkpoint))
    incompatible = model.load_state_dict(state_dict, strict=strict)
    metadata = dict(checkpoint) if isinstance(checkpoint, Mapping) else {}
    metadata["missing_keys"] = list(incompatible.missing_keys)
    metadata["unexpected_keys"] = list(incompatible.unexpected_keys)
    return metadata
