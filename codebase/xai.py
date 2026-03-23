"""
xai.py
------
Grad-CAM (Gradient-weighted Class Activation Mapping) explainability.

For each selected test image:
  1. Computes the Grad-CAM heatmap at the last convolutional layer.
  2. Overlays the heatmap on the original image.
  3. Annotates with predicted class and confidence score.
  4. Saves the composite figure to outputs/gradcam/.

Works with resnet50, resnet34, and mobilenet architectures.
"""

import argparse
import logging
import os
import sys
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from models.model_factory import get_gradcam_target_layer

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Grad-CAM Core
# ─────────────────────────────────────────────

class GradCAM:
    """
    Hooks into the target layer to capture activations and gradients,
    then computes the Grad-CAM heatmap.

    Args:
        model        : The trained nn.Module.
        target_layer : The convolutional layer to hook (e.g. model.layer4[-1]).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model        = model
        self.target_layer = target_layer
        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None
        self._register_hooks()

    # ── hooks ───────────────────────────────────────────────────

    def _register_hooks(self) -> None:
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output) -> None:
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output) -> None:
        self._gradients = grad_output[0].detach()

    # ── public method ───────────────────────────────────────────

    def generate(
        self,
        image_tensor: torch.Tensor,   # (1, C, H, W) on device
        class_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Compute Grad-CAM heatmap for a single image.

        Args:
            image_tensor : Preprocessed image tensor with batch dim.
            class_idx    : Class to explain; if None uses argmax prediction.

        Returns:
            heatmap   : (H, W) float32 array in [0, 1].
            class_idx : Integer index of the explained class.
            confidence: Softmax probability of the predicted/explained class.
        """
        self.model.eval()
        image_tensor = image_tensor.requires_grad_(True)

        # Forward pass
        logits = self.model(image_tensor)           # (1, num_classes)
        probs  = torch.softmax(logits, dim=1)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        confidence = float(probs[0, class_idx].item())

        # Backward pass w.r.t. the target class score
        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Pool gradients over spatial dims → channel weights
        gradients  = self._gradients   # (1, C, H, W)
        activations = self._activations  # (1, C, H, W)

        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam     = (weights * activations).sum(dim=1).squeeze(0)  # (H, W)
        cam     = torch.relu(cam).cpu().numpy()

        # Normalise to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam, class_idx, confidence


# ─────────────────────────────────────────────
# Overlay & Save
# ─────────────────────────────────────────────

def _tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a normalised image tensor (C, H, W) back to a uint8 RGB ndarray.
    """
    mean = np.array(config.NORMALIZE_MEAN)
    std  = np.array(config.NORMALIZE_STD)
    img  = tensor.cpu().numpy().transpose(1, 2, 0)   # (H, W, C)
    img  = img * std + mean                           # de-normalise
    img  = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def _overlay_heatmap(
    original_img: np.ndarray,      # (H, W, 3) uint8 RGB
    heatmap:      np.ndarray,      # (h, w) float32 [0, 1]
    alpha:        float = config.GRADCAM_ALPHA,
) -> np.ndarray:
    """
    Resizes the heatmap to match the original image and alpha-blends it.

    Returns:
        Blended uint8 RGB image.
    """
    h, w = original_img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_color   = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    blended = (
        (1 - alpha) * original_img.astype(float) +
        alpha       * heatmap_rgb.astype(float)
    ).clip(0, 255).astype(np.uint8)
    return blended


def _save_gradcam_figure(
    original_img:  np.ndarray,
    overlay_img:   np.ndarray,
    heatmap:       np.ndarray,
    class_name:    str,
    confidence:    float,
    save_path:     str,
) -> None:
    """
    Saves a 3-panel figure: original | heatmap | overlay.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_img)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap", fontsize=12)
    axes[1].axis("off")
    plt.colorbar(
        plt.cm.ScalarMappable(cmap="jet"),
        ax=axes[1], fraction=0.046, pad=0.04
    )

    axes[2].imshow(overlay_img)
    axes[2].set_title(
        f"Overlay\nPredicted: {class_name}  ({confidence:.1%})",
        fontsize=12,
    )
    axes[2].axis("off")

    plt.suptitle(
        f"Grad-CAM Explanation — {class_name}",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.debug(f"Saved Grad-CAM figure → '{save_path}'")


# ─────────────────────────────────────────────
# Public Entry-Point
# ─────────────────────────────────────────────

def run_gradcam(
    model:       nn.Module,
    model_name:  str,
    test_loader: DataLoader,
    class_names: List[str],
    device:      torch.device,
    num_images:  int = config.GRADCAM_NUM_IMAGES,
) -> None:
    """
    Runs Grad-CAM on the first `num_images` samples from the test loader
    and saves visualisations to outputs/gradcam/.

    Args:
        model       : Trained model (nn.Module).
        model_name  : Architecture string from config (for layer selection).
        test_loader : Test DataLoader.
        class_names : Ordered list of class name strings.
        device      : Compute device.
        num_images  : How many images to explain.
    """
    logger.info("=" * 55)
    logger.info(f"Grad-CAM: explaining {num_images} test images (balanced across classes) …")

    os.makedirs(config.GRADCAM_SAVE_PATH, exist_ok=True)
    model.to(device)

    target_layer = get_gradcam_target_layer(model, model_name)
    gradcam      = GradCAM(model, target_layer)

    # Dictionary to track how many images we've explained per class
    images_per_class = num_images // len(class_names)
    remainder = num_images % len(class_names)
    
    # Target counts for each class index
    class_targets = {i: images_per_class + (1 if i < remainder else 0) for i in range(len(class_names))}
    class_counts = {i: 0 for i in range(len(class_names))}

    total_saved = 0
    for batch_images, batch_labels in tqdm(test_loader, desc="[GradCAM]"):
        if total_saved >= num_images:
            break

        for i in range(batch_images.size(0)):
            if total_saved >= num_images:
                break

            true_label = int(batch_labels[i].item())
            
            # Skip if we've already met the target for this class
            if class_counts[true_label] >= class_targets[true_label]:
                continue

            img_tensor = batch_images[i].unsqueeze(0).to(device)
            heatmap, pred_idx, confidence = gradcam.generate(img_tensor)

            original_img = _tensor_to_numpy_image(batch_images[i])
            overlay_img  = _overlay_heatmap(original_img, heatmap)
            pred_name    = class_names[pred_idx]
            true_name    = class_names[true_label]

            fname = (
                f"img{total_saved:03d}_true-{true_name}_pred-{pred_name}"
                f"_conf{confidence:.2f}.png"
            )
            _save_gradcam_figure(
                original_img, overlay_img, heatmap,
                class_name=pred_name,
                confidence=confidence,
                save_path=os.path.join(config.GRADCAM_SAVE_PATH, fname),
            )
            class_counts[true_label] += 1
            total_saved += 1

    logger.info(
        f"Grad-CAM complete. {total_saved} figures saved to "
        f"'{config.GRADCAM_SAVE_PATH}'."
    )


if __name__ == "__main__":
    # ─── standalone imports ──────────────────────────────────────────
    import preprocess
    from models.model_factory import get_model
    from dataset            import ImageClassificationDataset, build_dataloaders
    from utils.save_utils   import load_model_weights

    parser = argparse.ArgumentParser(description="Run Grad-CAM on the image classification model.")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to the data root directory (containing train/val/test).")
    parser.add_argument("--test", type=str, default=None,
                        help="Direct path to the processed folder to explain (e.g. data/processed/val).")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to the directory where Grad-CAM results will be saved.")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to the model checkpoint to use for explanations.")
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
        config.GRADCAM_SAVE_PATH = args.output
        os.makedirs(config.GRADCAM_SAVE_PATH, exist_ok=True)

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
    target_test_dir = args.test if args.test else config.TEST_DIR
    logger.info(f"Explaining images from: {target_test_dir}")
    
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
        logger.warning(f"Best model checkpoint not found at {best_path}. Grad-CAM will use initialised weights.")
    
    # Step 4: Run Grad-CAM
    run_gradcam(
        model=model,
        model_name=config.MODEL_NAME,
        test_loader=test_loader,
        class_names=class_names,
        device=device,
        num_images=config.GRADCAM_NUM_IMAGES,
    )
