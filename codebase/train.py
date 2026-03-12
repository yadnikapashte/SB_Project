"""
train.py
--------
Full training loop with:
  - CrossEntropy loss
  - Adam optimiser
  - Per-epoch train + validation metrics
  - Learning rate scheduling (ReduceLROnPlateau)
  - Early stopping
  - Saves best_model.pth and final_model.pth
"""

import argparse
import logging
import os
import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from utils.metrics import compute_accuracy
from utils.save_utils import save_model, save_training_log

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Single-epoch helpers
# ─────────────────────────────────────────────

def _train_one_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    criterion:  nn.Module,
    optimiser:  torch.optim.Optimizer,
    device:     torch.device,
    epoch:      int,
) -> Tuple[float, float]:
    """
    Run one training epoch.

    Returns:
        (avg_loss, accuracy)  both as Python floats.
    """
    model.train()
    running_loss = 0.0
    all_preds:  List[int] = []
    all_labels: List[int] = []

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [Train]", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimiser.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimiser.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = running_loss / len(loader.dataset)
    acc      = compute_accuracy(all_labels, all_preds)
    return avg_loss, acc


def _validate_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    epoch:     int,
) -> Tuple[float, float]:
    """
    Run one validation epoch (no gradient computation).

    Returns:
        (avg_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    all_preds:  List[int] = []
    all_labels: List[int] = []

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [Val  ]", leave=False)
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss   = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = running_loss / len(loader.dataset)
    acc      = compute_accuracy(all_labels, all_preds)
    return avg_loss, acc


# ─────────────────────────────────────────────
# Main Training Function
# ─────────────────────────────────────────────

def train_model(
    model:        nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    device:       torch.device,
) -> Dict:
    """
    Complete training pipeline: epochs, early stopping, LR scheduling,
    and model checkpointing.

    Args:
        model        : Initialised nn.Module (from model_factory).
        train_loader : Training DataLoader.
        val_loader   : Validation DataLoader.
        device       : torch.device.

    Returns:
        history dict with train/val loss and accuracy lists.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")
    patience_counter = 0
    history: Dict[str, List] = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
    }

    best_model_path  = os.path.join(config.MODEL_SAVE_PATH, "best_model.pth")
    final_model_path = os.path.join(config.MODEL_SAVE_PATH, "final_model.pth")
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)

    logger.info("=" * 55)
    logger.info(f"Training started — device: {device} | "
                f"epochs: {config.EPOCHS} | lr: {config.LEARNING_RATE}")
    logger.info("=" * 55)

    for epoch in range(1, config.EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = _train_one_epoch(
            model, train_loader, criterion, optimiser, device, epoch
        )
        val_loss, val_acc = _validate_one_epoch(
            model, val_loader, criterion, device, epoch
        )

        scheduler.step(val_loss)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        logger.info(
            f"Epoch {epoch:03d}/{config.EPOCHS} | "
            f"Train loss: {train_loss:.4f}  acc: {train_acc:.2f}% | "
            f"Val loss: {val_loss:.4f}  acc: {val_acc:.2f}% | "
            f"{elapsed:.1f}s"
        )

        # ── checkpoint best model ──────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_model(model, best_model_path)
            logger.info(f"  ✓ Best model saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            logger.debug(
                f"  No improvement. Patience: {patience_counter}/"
                f"{config.EARLY_STOPPING_PATIENCE}"
            )

        # ── early stopping ─────────────────────────────────────
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            logger.info(
                f"Early stopping triggered at epoch {epoch} "
                f"(patience={config.EARLY_STOPPING_PATIENCE})."
            )
            break

    # Always save the final state regardless of whether it is the best
    save_model(model, final_model_path)
    logger.info(f"Final model saved to '{final_model_path}'.")

    # Save training history as JSON log
    save_training_log(history, os.path.join(config.METRICS_SAVE_PATH, "training_log.json"))

    logger.info("Training complete.")
    return history


if __name__ == "__main__":
    # ─── standalone imports ──────────────────────────────────────────
    import preprocess
    from models.model_factory import get_model
    from dataset            import ImageClassificationDataset, build_dataloaders

    parser = argparse.ArgumentParser(description="Train the image classification model.")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to the data directory (containing train/val/test).")
    args = parser.parse_args()

    # Override config dirs if --data_dir is provided
    if args.data_dir:
        config.TRAIN_DIR = os.path.join(args.data_dir, "train")
        config.VAL_DIR   = os.path.join(args.data_dir, "val")
        config.TEST_DIR  = os.path.join(args.data_dir, "test")

    # Setup simple console logging for standalone run
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")

    device = torch.device(config.DEVICE)

    # Step 1: Preprocess
    pp = preprocess.prepare_datasets()
    class_names = pp["class_names"]
    num_classes = pp["num_classes"]

    # Step 2: Build Datasets/Loaders
    train_ds = ImageClassificationDataset(pp["train_classes"], class_names, transform=pp["train_transform"])
    val_ds   = ImageClassificationDataset(pp["val_classes"],   class_names, transform=pp["val_transform"])
    test_ds  = ImageClassificationDataset(pp["test_classes"],  class_names, transform=pp["test_transform"])
    train_l, val_l, _ = build_dataloaders(train_ds, val_ds, test_ds)

    # Step 3: Model
    model = get_model(config.MODEL_NAME, num_classes)

    # Step 4: Run Training
    train_model(model, train_l, val_l, device)
