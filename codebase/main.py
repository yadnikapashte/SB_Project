"""
main.py
-------
Orchestrates the entire image classification pipeline:

  1. Load configuration
  2. Preprocess dataset (validate dirs, build transforms, discover classes)
  3. Build PyTorch Datasets and DataLoaders
  4. Initialise model from model_factory
  5. Train model (with early stopping + LR scheduling)
  6. Run inference on test set
  7. Evaluate and save metrics / plots
  8. Save model weights
  9. Save class mapping JSON
 10. Run Grad-CAM explainability

Usage:
    python main.py
"""

import argparse
import logging
import os
import sys
import time

import torch

# ─── project imports ────────────────────────────────────────────
import config
import preprocess
import train as trainer
import test   as tester
import evaluate
import xai

from dataset            import ImageClassificationDataset, build_dataloaders
from models.model_factory import get_model
from utils.save_utils   import save_class_mapping, load_model_weights


# ─────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────

def setup_logging() -> None:
    """Configure root logger to write to console and a log file."""
    os.makedirs(config.LOG_DIR, exist_ok=True)
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.LOG_FILE, mode="a"),
    ]
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL, logging.INFO),
        format=fmt,
        handlers=handlers,
    )


# ─────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────

def main() -> None:
    t_start = time.time()

    parser = argparse.ArgumentParser(description="Image Classification Pipeline")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to the data directory (containing train/val/test).")
    parser.add_argument("--force", action="store_true", help="Force re-split of dataset")
    args = parser.parse_args()

    # Override config dirs if --data_dir is provided
    if args.data_dir:
        config.TRAIN_DIR = os.path.join(args.data_dir, "train")
        config.VAL_DIR   = os.path.join(args.data_dir, "val")
        config.TEST_DIR  = os.path.join(args.data_dir, "test")

    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("╔══════════════════════════════════════════════════════╗")
    logger.info("║        Image Classification Pipeline — START         ║")
    logger.info("╚══════════════════════════════════════════════════════╝")
    logger.info(f"Model      : {config.MODEL_NAME}")
    logger.info(f"Device     : {config.DEVICE}")
    logger.info(f"Epochs     : {config.EPOCHS}")
    logger.info(f"Batch size : {config.BATCH_SIZE}")
    logger.info(f"Image size : {config.IMAGE_SIZE}×{config.IMAGE_SIZE}")

    device = torch.device(config.DEVICE)

    # ── Step 1: Preprocess ────────────────────────────────────────
    logger.info("\n─── Step 1 / 9 : Preprocessing ───")
    try:
        pp = preprocess.prepare_datasets(force=args.force)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        logger.error(
            "Please make sure your data folders exist and follow the "
            "expected structure:\n"
            "  data/{train,val,test}/<class_name>/<image_files>"
        )
        sys.exit(1)

    class_names = pp["class_names"]
    num_classes = pp["num_classes"]

    # ── Step 2: Build Datasets ────────────────────────────────────
    logger.info("\n─── Step 2 / 9 : Building Datasets ───")
    train_ds = ImageClassificationDataset(
        pp["train_classes"], class_names, transform=pp["train_transform"]
    )
    val_ds = ImageClassificationDataset(
        pp["val_classes"], class_names, transform=pp["val_transform"]
    )
    test_ds = ImageClassificationDataset(
        pp["test_classes"], class_names, transform=pp["test_transform"]
    )
    train_loader, val_loader, test_loader = build_dataloaders(
        train_ds, val_ds, test_ds
    )

    # ── Step 3: Initialise Model ──────────────────────────────────
    logger.info("\n─── Step 3 / 9 : Initialising Model ───")
    model = get_model(config.MODEL_NAME, num_classes)
    model.to(device)

    # ── Step 4: Train ─────────────────────────────────────────────
    logger.info("\n─── Step 4 / 9 : Training ───")
    history = trainer.train_model(model, train_loader, val_loader, device)

    # ── Step 5: Load best checkpoint for evaluation ───────────────
    best_path = os.path.join(config.MODEL_SAVE_PATH, "best_model.pth")
    logger.info(f"\n─── Step 5 / 9 : Loading best checkpoint ← '{best_path}' ───")
    model = load_model_weights(model, best_path, device)

    # ── Step 6: Test Inference ────────────────────────────────────
    logger.info("\n─── Step 6 / 9 : Test Inference ───")
    y_pred, y_true, confidences = tester.run_inference(
        model, test_loader, device
    )

    # ── Step 7: Evaluate ─────────────────────────────────────────
    logger.info("\n─── Step 7 / 9 : Evaluating ───")
    summary = evaluate.evaluate(y_true, y_pred, class_names, history)
    logger.info(f"Final Test Accuracy : {summary['accuracy']:.4f}")
    logger.info(f"Macro F1-Score      : {summary['macro_f1']:.4f}")

    # ── Step 8: Save Class Mapping ────────────────────────────────
    logger.info("\n─── Step 8 / 9 : Saving Class Mapping ───")
    save_class_mapping(class_names, config.CLASS_MAPPING_PATH)

    # ── Step 9: Grad-CAM ─────────────────────────────────────────
    logger.info("\n─── Step 9 / 9 : Grad-CAM Explainability ───")
    xai.run_gradcam(
        model=model,
        model_name=config.MODEL_NAME,
        test_loader=test_loader,
        class_names=class_names,
        device=device,
        num_images=config.GRADCAM_NUM_IMAGES,
    )

    # ── Done ──────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    logger.info("\n╔══════════════════════════════════════════════════════╗")
    logger.info(f"║  Pipeline complete in {elapsed / 60:.1f} min.                    ║")
    logger.info(f"║  Outputs in: {config.OUTPUT_DIR:<39}║")
    logger.info("╚══════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
