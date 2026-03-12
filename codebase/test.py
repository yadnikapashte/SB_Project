"""
test.py
-------
Evaluates the trained model on the test DataLoader and returns
raw predictions and ground-truth labels for downstream evaluation.
"""

import logging
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config

logger = logging.getLogger(__name__)


def run_inference(
    model:       nn.Module,
    test_loader: DataLoader,
    device:      torch.device,
) -> Tuple[List[int], List[int], List[float]]:
    """
    Run the model over the entire test set.

    Args:
        model       : Trained nn.Module (already on device or will be moved).
        test_loader : DataLoader wrapping the test ImageClassificationDataset.
        device      : torch.device to run inference on.

    Returns:
        all_preds   : List of predicted class indices.
        all_labels  : List of ground-truth class indices.
        all_confs   : List of max-softmax confidence scores [0, 1].
    """
    model.to(device)
    model.eval()

    all_preds:  List[int]   = []
    all_labels: List[int]   = []
    all_confs:  List[float] = []

    logger.info("=" * 55)
    logger.info("Running inference on test set …")

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="[Test] Inference", leave=True)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            probs  = torch.softmax(logits, dim=1)        # (B, C)
            confs, preds = probs.max(dim=1)              # (B,)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_confs.extend(confs.cpu().tolist())

    n_correct = sum(p == l for p, l in zip(all_preds, all_labels))
    test_acc  = 100.0 * n_correct / len(all_labels)
    mean_conf = sum(all_confs) / len(all_confs)

    logger.info(
        f"Test inference complete — "
        f"samples: {len(all_labels)} | "
        f"accuracy: {test_acc:.2f}% | "
        f"mean confidence: {mean_conf:.4f}"
    )
    return all_preds, all_labels, all_confs
