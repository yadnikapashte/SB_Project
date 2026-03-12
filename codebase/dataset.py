"""
dataset.py
----------
Custom PyTorch Dataset that reads images from class-wise folder structures,
applies transforms, and returns (tensor, label) pairs.
"""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image, UnidentifiedImageError

import config

logger = logging.getLogger(__name__)


class ImageClassificationDataset(Dataset):
    """
    A generic image classification dataset.

    Expects data organised as:
        root/
            class_a/img1.jpg
            class_a/img2.jpg
            class_b/img1.jpg
            ...

    Args:
        class_to_images : dict mapping class_name → list of image paths.
        class_names     : ordered list of class names (defines label indices).
        transform       : torchvision transform to apply to each image.
    """

    def __init__(
        self,
        class_to_images: Dict[str, List[str]],
        class_names: List[str],
        transform: Optional[Callable] = None,
    ) -> None:
        self.transform    = transform
        self.class_names  = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        # Flatten into a list of (path, label_int) tuples
        self.samples: List[Tuple[str, int]] = []
        for class_name, paths in class_to_images.items():
            if class_name not in self.class_to_idx:
                logger.warning(
                    f"Class '{class_name}' not in class_names — skipping."
                )
                continue
            label = self.class_to_idx[class_name]
            for p in paths:
                self.samples.append((p, label))

        logger.info(
            f"Dataset created: {len(self.samples)} samples, "
            f"{len(self.class_names)} classes."
        )

    # ── dunder methods ──────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
            logger.error(f"Cannot load image '{img_path}': {exc}")
            # Return a black image tensor of the correct size so training
            # doesn't crash on a single bad file.
            image = Image.new("RGB", (config.IMAGE_SIZE, config.IMAGE_SIZE))

        if self.transform:
            image = self.transform(image)

        return image, label

    # ── helpers ─────────────────────────────────────────────────

    def get_class_name(self, label: int) -> str:
        """Convert integer label back to class name string."""
        return self.class_names[label]


# ─────────────────────────────────────────────
# DataLoader Factory
# ─────────────────────────────────────────────

def build_dataloaders(
    train_dataset: ImageClassificationDataset,
    val_dataset:   ImageClassificationDataset,
    test_dataset:  ImageClassificationDataset,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Wraps three Dataset objects into DataLoader instances using
    BATCH_SIZE and NUM_WORKERS from config.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    pin = config.DEVICE == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin,
        drop_last=True,          # avoids tiny last batch issues
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin,
    )

    logger.info(
        f"DataLoaders ready — "
        f"train: {len(train_loader)} batches | "
        f"val: {len(val_loader)} batches | "
        f"test: {len(test_loader)} batches"
    )
    return train_loader, val_loader, test_loader
