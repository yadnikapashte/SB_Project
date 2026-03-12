"""
preprocess.py
-------------
Handles dataset loading, image resizing, normalization, optional augmentation,
per-class image limits, and train/val/test split validation.
"""

import os
import logging
import random
import shutil
from pathlib import Path
from collections import defaultdict
from PIL import Image

import torch
from torchvision import transforms

import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Transform Builders
# ─────────────────────────────────────────────

def get_train_transforms() -> transforms.Compose:
    """
    Returns transforms for the training set.
    Includes augmentation if USE_AUGMENTATION is enabled in config.
    """
    base = [
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    ]

    if config.USE_AUGMENTATION:
        augment = []
        if config.APPLY_HFLIP:
            augment.append(transforms.RandomHorizontalFlip(p=config.AUG_HFLIP_PROB))
        
        if config.APPLY_VFLIP:
            augment.append(transforms.RandomVerticalFlip(p=config.AUG_VFLIP_PROB))
        
        if config.APPLY_ROTATION:
            augment.append(transforms.RandomRotation(degrees=config.AUG_ROTATION_DEGREES))
        
        if config.APPLY_COLOR_JITTER:
            augment.append(transforms.ColorJitter(
                brightness=config.AUG_BRIGHTNESS,
                contrast=config.AUG_CONTRAST,
                saturation=config.AUG_SATURATION,
                hue=config.AUG_HUE
            ))
        
        if config.APPLY_TRANSLATE or config.APPLY_SHEAR:
            translate = config.AUG_TRANSLATE if config.APPLY_TRANSLATE else None
            shear = config.AUG_SHEAR_DEGREES if config.APPLY_SHEAR else None
            augment.append(transforms.RandomAffine(
                degrees=0, 
                translate=translate,
                shear=shear
            ))
            
        base.extend(augment)
        logger.info("Advanced augmentation enabled for training set.")

    base.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN,
                             std=config.NORMALIZE_STD),
    ])

    if config.USE_AUGMENTATION and config.APPLY_RANDOM_ERASING:
        base.append(transforms.RandomErasing(p=config.AUG_RANDOM_ERASING_PROB))

    return transforms.Compose(base)


def get_val_test_transforms() -> transforms.Compose:
    """
    Returns deterministic transforms for validation and test sets.
    No augmentation — only resize + normalize.
    """
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN,
                             std=config.NORMALIZE_STD),
    ])


# ─────────────────────────────────────────────
# Directory / Split Validation
# ─────────────────────────────────────────────

def validate_split_dirs() -> bool:
    """
    Checks that TRAIN_DIR, VAL_DIR, and TEST_DIR exist and each contain
    at least one class sub-folder with images.

    Returns True if valid, raises FileNotFoundError otherwise.
    """
    for split_name, split_path in [
        ("train", config.TRAIN_DIR),
        ("val",   config.VAL_DIR),
        ("test",  config.TEST_DIR),
    ]:
        path = Path(split_path)
        if not path.exists():
            raise FileNotFoundError(
                f"[preprocess] {split_name} directory not found: {split_path}"
            )

        class_dirs = [d for d in path.iterdir() if d.is_dir()]
        if not class_dirs:
            raise FileNotFoundError(
                f"[preprocess] No class sub-folders found in {split_path}"
            )

        logger.info(f"[preprocess] {split_name}: {len(class_dirs)} classes "
                    f"found in '{split_path}'.")
    return True


# ─────────────────────────────────────────────
# Class Discovery & Image Counting
# ─────────────────────────────────────────────

def get_offline_transforms() -> transforms.Compose:
    """
    Returns transforms for offline augmentation (returns PIL Image).
    Used to generate new physical files on disk.
    """
    augment = [
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    ]
    if config.APPLY_HFLIP:
        augment.append(transforms.RandomHorizontalFlip(p=config.AUG_HFLIP_PROB))
    if config.APPLY_VFLIP:
        augment.append(transforms.RandomVerticalFlip(p=config.AUG_VFLIP_PROB))
    if config.APPLY_ROTATION:
        augment.append(transforms.RandomRotation(degrees=config.AUG_ROTATION_DEGREES))
    if config.APPLY_COLOR_JITTER:
        augment.append(transforms.ColorJitter(
            brightness=config.AUG_BRIGHTNESS,
            contrast=config.AUG_CONTRAST,
            saturation=config.AUG_SATURATION,
            hue=config.AUG_HUE
        ))
    if config.APPLY_TRANSLATE or config.APPLY_SHEAR:
        translate = config.AUG_TRANSLATE if config.APPLY_TRANSLATE else None
        shear = config.AUG_SHEAR_DEGREES if config.APPLY_SHEAR else None
        augment.append(transforms.RandomAffine(
            degrees=0, translate=translate, shear=shear
        ))
    return transforms.Compose(augment)


def discover_classes(root_dir: str) -> dict:
    """
    Walks root_dir and returns a dict mapping class_name → list of image paths.
    Respects IMAGES_PER_CLASS limit defined in config (as a cap).

    Args:
        root_dir: Path to the split directory (e.g. data/train).

    Returns:
        {class_name: [image_path, ...]}
    """
    supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    class_to_images = defaultdict(list)

    root = Path(root_dir)
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        images = [
            str(p) for p in class_dir.iterdir()
            if p.suffix.lower() in supported_exts
        ]
        
        if not images:
            logger.warning(f"  Class '{class_dir.name}': no images found.")
            continue

        # Enforce per-class image cap (only capping now, as oversampling is physical)
        if config.IMAGES_PER_CLASS:
            if len(images) > config.IMAGES_PER_CLASS:
                random.shuffle(images)
                images = images[: config.IMAGES_PER_CLASS]
                logger.debug(f"  Class '{class_dir.name}': capped at "
                             f"{config.IMAGES_PER_CLASS} images.")

        class_to_images[class_dir.name] = images
        # Clear log of the total images this class has in the dataset
        logger.info(f"  Class '{class_dir.name}': {len(images)} images total in dataset split.")

    return dict(class_to_images)


# ─────────────────────────────────────────────
# Dataset Splitting
# ─────────────────────────────────────────────

def split_dataset(force: bool = False) -> None:
    """
    Splits the raw dataset in RAW_DATA_PATH into train/val/test sets 
    based on DATA_SPLIT_RATIO, and saves them to TRAIN_DIR, VAL_DIR, TEST_DIR.
    """
    raw_path = Path(config.RAW_DATA_PATH)
    if not raw_path.exists():
        logger.error(f"[preprocess] Raw data path not found: {config.RAW_DATA_PATH}")
        return

    # Check if data already exists in split directories
    if not force:
        data_exists = all(Path(d).exists() and any(Path(d).iterdir()) 
                         for d in [config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR])
        if data_exists:
            logger.info("[preprocess] Split data already exists. Skipping split.")
            return

    logger.info(f"[preprocess] Splitting raw data from {config.RAW_DATA_PATH} ...")
    
    # Ensure split directories exist and are empty
    for d in [config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR]:
        path = Path(d)
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    train_ratio, val_ratio, test_ratio = config.DATA_SPLIT_RATIO

    for class_dir in sorted(raw_path.iterdir()):
        if not class_dir.is_dir():
            continue
        
        images = [
            p for p in class_dir.iterdir()
            if p.suffix.lower() in supported_exts
        ]
        random.shuffle(images)

        n = len(images)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        
        train_imgs = images[:n_train]
        val_imgs   = images[n_train : n_train + n_val]
        test_imgs  = images[n_train + n_val :]

        for split_name, split_imgs, split_dir in [
            ("train", train_imgs, config.TRAIN_DIR),
            ("val",   val_imgs,   config.VAL_DIR),
            ("test",  test_imgs,  config.TEST_DIR),
        ]:
            dest_class_dir = Path(split_dir) / class_dir.name
            dest_class_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. Copy initial split images
            written_paths = []
            for img_path in split_imgs:
                dest_path = dest_class_dir / img_path.name
                shutil.copy2(img_path, dest_path)
                written_paths.append(dest_path)

            # 2. If training split, physically augment images to reach target
            if split_name == "train" and config.IMAGES_PER_CLASS and len(written_paths) < config.IMAGES_PER_CLASS:
                n_original = len(written_paths)
                offline_transform = get_offline_transforms()
                
                logger.info(f"  Class '{class_dir.name}': physically augmenting "
                            f"{n_original} images to reach {config.IMAGES_PER_CLASS}...")
                
                # Outer loop over original images, inner loop over mutations
                # until we hit the target or the per-image limit
                target_hit = False
                for img_path in split_imgs:
                    if target_hit: break
                    
                    try:
                        with Image.open(img_path).convert("RGB") as pil_img:
                            for i in range(1, config.AUG_PER_IMAGE + 1):
                                if len(written_paths) >= config.IMAGES_PER_CLASS:
                                    target_hit = True
                                    break
                                
                                aug_img = offline_transform(pil_img)
                                aug_name = f"{img_path.stem}_aug_{i}{img_path.suffix}"
                                aug_path = dest_class_dir / aug_name
                                aug_img.save(aug_path)
                                written_paths.append(aug_path)
                    except Exception as e:
                        logger.warning(f"  Could not augment '{img_path.name}': {e}")
                
                logger.info(f"  Class '{class_dir.name}': reached {len(written_paths)} images in '{split_name}'.")
        
        logger.info(f"  Class '{class_dir.name}': finished with {len(train_imgs)} original train images.")

    logger.info("[preprocess] Dataset split and physical offline augmentation complete.")


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def prepare_datasets() -> dict:
    """
    Main entry-point called by main.py.

    1. Validates split directories.
    2. Discovers classes and counts per-class images.
    3. Returns transform objects for each split.

    Returns:
        {
          "train_transform": ...,
          "val_transform":   ...,
          "test_transform":  ...,
          "train_classes":   {class_name: [paths]},
          "val_classes":     {class_name: [paths]},
          "test_classes":    {class_name: [paths]},
          "class_names":     [sorted list of class names],
          "num_classes":     int,
        }
    """
    logger.info("=" * 55)
    logger.info("Preprocessing: ensuring dataset split ...")
    split_dataset()

    logger.info("Preprocessing: validating dataset directories …")
    validate_split_dirs()

    logger.info("Discovering training classes …")
    train_classes = discover_classes(config.TRAIN_DIR)

    logger.info("Discovering validation classes …")
    val_classes = discover_classes(config.VAL_DIR)

    logger.info("Discovering test classes …")
    test_classes = discover_classes(config.TEST_DIR)
    # Use training classes as the canonical class list
    class_names = sorted(train_classes.keys())
    num_classes  = len(class_names)
    logger.info(f"Total classes: {num_classes}  →  {class_names}")

    return {
        "train_transform": get_train_transforms(),
        "val_transform":   get_val_test_transforms(),
        "test_transform":  get_val_test_transforms(),
        "train_classes":   train_classes,
        "val_classes":     val_classes,
        "test_classes":    test_classes,
        "class_names":     class_names,
        "num_classes":     num_classes,
    }


if __name__ == "__main__":
    import logging
    import sys
    
    # Try to setup logging using main's setup if possible, otherwise use basicConfig
    try:
        from main import setup_logging
        setup_logging()
    except ImportError:
        logging.basicConfig(level=logging.INFO)
    
    prepare_datasets()
