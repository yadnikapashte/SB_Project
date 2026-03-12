"""
utils/save_utils.py
-------------------
Utility functions for persisting artefacts produced by the pipeline:
  - PyTorch model weights  (.pth)
  - Class mapping          (.json)
  - Training history log   (.json)
"""

import json
import logging
import os
from typing import Dict, List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def save_model(model: nn.Module, save_path: str) -> None:
    """
    Saves model state dict to disk.

    Args:
        model     : The nn.Module whose weights to persist.
        save_path : Full file path ending in .pth.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    size_mb = os.path.getsize(save_path) / (1024 ** 2)
    logger.info(f"Model saved → '{save_path}'  ({size_mb:.1f} MB)")


def load_model_weights(model: nn.Module, load_path: str,
                       device: torch.device) -> nn.Module:
    """
    Loads saved state dict into an existing model instance.

    Args:
        model     : nn.Module with matching architecture.
        load_path : Path to the .pth file.
        device    : Target device for loading.

    Returns:
        The model with weights loaded (in-place, also returned for chaining).
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"[save_utils] Model file not found: {load_path}")
    state_dict = torch.load(load_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    logger.info(f"Model weights loaded ← '{load_path}'")
    return model


def save_class_mapping(class_names: List[str], save_path: str) -> None:
    """
    Saves a JSON file mapping integer indices to class name strings.

    Format:
        {"0": "class_a", "1": "class_b", ...}

    Args:
        class_names : Ordered list of class name strings.
        save_path   : Destination file path (e.g. outputs/class_mapping.json).
    """
    mapping = {str(idx): name for idx, name in enumerate(class_names)}
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(mapping, f, indent=2)
    logger.info(f"Class mapping saved → '{save_path}'  "
                f"({len(class_names)} classes)")


def save_training_log(history: Dict, save_path: str) -> None:
    """
    Persists the training history (loss / accuracy per epoch) to JSON.

    Args:
        history   : Dict with list values per metric key.
        save_path : Destination file path.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training log saved → '{save_path}'")
