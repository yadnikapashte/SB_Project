"""
evaluate.py
-----------
Generates full evaluation metrics and visualisations:
  - Confusion matrix (seaborn heatmap)
  - Per-class Precision, Recall, F1
  - Overall Accuracy
  - Classification report saved as CSV + PNG

All outputs go to outputs/reports/.
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List

import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import config
from utils.metrics import compute_accuracy, compute_precision_recall_f1

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Confusion Matrix
# ─────────────────────────────────────────────

def plot_confusion_matrix(
    y_true:      List[int],
    y_pred:      List[int],
    class_names: List[str],
    save_path:   str,
) -> None:
    """
    Plots a normalised confusion matrix and saves it as a PNG.

    Args:
        y_true      : Ground-truth labels.
        y_pred      : Predicted labels.
        class_names : Ordered list of class name strings.
        save_path   : File path for the saved figure.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)   # row-normalise

    fig, axes = plt.subplots(1, 2, figsize=(max(14, len(class_names) * 1.5),
                                            max(10, len(class_names) * 1.2)))

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Confusion Matrix (counts)", "Confusion Matrix (normalised)"],
        ["d", ".2f"],
    ):
        sns.heatmap(
            data,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            linewidths=0.5,
        )
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual",    fontsize=11)
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved → '{save_path}'")


# ─────────────────────────────────────────────
# Training Curve Plot
# ─────────────────────────────────────────────

def plot_training_curves(history: Dict, save_path: str) -> None:
    """
    Saves a 2-panel figure of training & validation loss/accuracy.

    Args:
        history   : Dict with keys train_loss, val_loss, train_acc, val_acc.
        save_path : Output PNG file path.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss panel
    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    ax1.plot(epochs, history["val_loss"],   "r-o", label="Val Loss")
    ax1.set_title("Loss per Epoch", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Accuracy panel
    ax2.plot(epochs, history["train_acc"], "b-o", label="Train Acc")
    ax2.plot(epochs, history["val_acc"],   "r-o", label="Val Acc")
    ax2.set_title("Accuracy per Epoch", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Training curves saved → '{save_path}'")


# ─────────────────────────────────────────────
# Per-class Metrics Report
# ─────────────────────────────────────────────

def generate_classification_report(
    y_true:      List[int],
    y_pred:      List[int],
    class_names: List[str],
    save_dir:    str,
) -> Dict:
    """
    Computes per-class and macro-average metrics, saves them as CSV + JSON,
    and prints a formatted table to the log.

    Args:
        y_true      : Ground-truth labels.
        y_pred      : Predicted labels.
        class_names : Ordered class name list.
        save_dir    : Directory to write report files.

    Returns:
        Dict of summary metrics (accuracy, macro precision/recall/F1).
    """
    os.makedirs(save_dir, exist_ok=True)

    # sklearn full report
    report_str = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
    )
    logger.info("\nClassification Report:\n" + report_str)

    # Save as plain text
    with open(os.path.join(save_dir, "classification_report.txt"), "w") as f:
        f.write(report_str)

    # Per-class metrics → DataFrame → CSV
    precision_per_class = precision_score(y_true, y_pred, average=None,
                                          zero_division=0)
    recall_per_class    = recall_score(y_true, y_pred, average=None,
                                       zero_division=0)
    f1_per_class        = f1_score(y_true, y_pred, average=None,
                                   zero_division=0)
    support_per_class   = np.bincount(y_true, minlength=len(class_names))

    df = pd.DataFrame({
        "class":     class_names,
        "precision": precision_per_class,
        "recall":    recall_per_class,
        "f1_score":  f1_per_class,
        "support":   support_per_class,
    })
    df.to_csv(os.path.join(save_dir, "per_class_metrics.csv"), index=False)
    logger.info("Per-class metrics CSV saved.")

    # Summary
    summary = {
        "accuracy":          accuracy_score(y_true, y_pred),
        "macro_precision":   float(precision_score(y_true, y_pred,
                                                   average="macro",
                                                   zero_division=0)),
        "macro_recall":      float(recall_score(y_true, y_pred,
                                               average="macro",
                                               zero_division=0)),
        "macro_f1":          float(f1_score(y_true, y_pred,
                                            average="macro",
                                            zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred,
                                                    average="weighted",
                                                    zero_division=0)),
        "weighted_recall":   float(recall_score(y_true, y_pred,
                                               average="weighted",
                                               zero_division=0)),
        "weighted_f1":       float(f1_score(y_true, y_pred,
                                           average="weighted",
                                           zero_division=0)),
    }
    with open(os.path.join(save_dir, "summary_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        f"Summary — Acc: {summary['accuracy']:.4f} | "
        f"Macro-F1: {summary['macro_f1']:.4f} | "
        f"Weighted-F1: {summary['weighted_f1']:.4f}"
    )
    return summary


# ─────────────────────────────────────────────
# Public Entry-Point
# ─────────────────────────────────────────────

def evaluate(
    y_true:      List[int],
    y_pred:      List[int],
    class_names: List[str],
    history:     Dict,
) -> Dict:
    """
    Full evaluation pipeline called from main.py.

    1. Plots confusion matrix.
    2. Plots training curves.
    3. Generates classification report.

    Args:
        y_true      : Ground-truth labels from test set.
        y_pred      : Predicted labels from test set.
        class_names : Ordered class name list.
        history     : Training history dict (from train.py).

    Returns:
        summary metrics dict.
    """
    report_dir = config.METRICS_SAVE_PATH

    plot_confusion_matrix(
        y_true, y_pred, class_names,
        save_path=os.path.join(report_dir, "confusion_matrix.png"),
    )

    if history:
        plot_training_curves(
            history,
            save_path=os.path.join(report_dir, "training_curves.png"),
        )

    summary = generate_classification_report(
        y_true, y_pred, class_names, save_dir=report_dir
    )
    return summary


if __name__ == "__main__":
    # ─── standalone imports ──────────────────────────────────────────
    import preprocess
    import test as tester
    from models.model_factory import get_model
    from dataset            import ImageClassificationDataset, build_dataloaders
    from utils.save_utils   import load_model_weights

    parser = argparse.ArgumentParser(description="Evaluate the image classification model.")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to the data root directory (containing train/val/test).")
    parser.add_argument("--test", type=str, default=None,
                        help="Direct path to the processed folder to evaluate (e.g. data/processed/val).")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to the directory where evaluation results will be saved.")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to the model checkpoint to evaluate.")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Architecture of the model (e.g., resnet50, resnet34). Overrides config.MODEL_NAME.")
    args = parser.parse_args()

    # Override config dirs if --data_dir is provided
    if args.data_dir:
        config.TRAIN_DIR = os.path.join(args.data_dir, "train")
        config.VAL_DIR   = os.path.join(args.data_dir, "val")
        config.TEST_DIR  = os.path.join(args.data_dir, "test")

    # Override output dir if --output is provided
    if args.output:
        config.METRICS_SAVE_PATH = args.output
        os.makedirs(config.METRICS_SAVE_PATH, exist_ok=True)

    if args.model_name:
        config.MODEL_NAME = args.model_name
    elif args.model:
        if 'resnet34' in args.model.lower():
            config.MODEL_NAME = 'resnet34'
        elif 'resnet50' in args.model.lower():
            config.MODEL_NAME = 'resnet50'
        elif 'mobilenet' in args.model.lower():
            config.MODEL_NAME = 'mobilenet'

    # Setup simple console + file logging for standalone run
    os.makedirs(config.LOG_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(config.LOG_FILE, mode="a"),
        ],
    )

    device = torch.device(config.DEVICE)

    # Step 1: Preprocess (ensures classes and standard transforms)
    pp = preprocess.prepare_datasets()
    class_names = pp["class_names"]

    # Step 2: Build DataLoader
    # If a specific --test folder is provided, use it. Otherwise default to config.TEST_DIR
    target_test_dir = args.test if args.test else config.TEST_DIR
    logger.info(f"Evaluating on data from: {target_test_dir}")
    
    test_classes = preprocess.discover_classes(target_test_dir)
    test_ds = ImageClassificationDataset(test_classes, class_names, transform=pp["test_transform"])
    _, _, test_loader = build_dataloaders(test_ds, test_ds, test_ds)

    # Step 3: Load Model
    model = get_model(config.MODEL_NAME, len(class_names))
    # Use --model if provided, otherwise default to config.MODEL_SAVE_PATH/best_model.pth
    best_path = args.model if args.model else os.path.join(config.MODEL_SAVE_PATH, "best_model.pth")
    
    if os.path.exists(best_path):
        model = load_model_weights(model, best_path, device)
    else:
        logger.warning(f"Best model checkpoint not found at {best_path}. Evaluation will use initialised weights.")
    
    # Step 4: Run Inference
    y_pred, y_true, _ = tester.run_inference(model, test_loader, device)

    # Step 5: Evaluate (history=None for standalone)
    evaluate(y_true, y_pred, class_names, history=None)
