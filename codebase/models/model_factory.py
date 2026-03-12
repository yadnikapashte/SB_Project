"""
models/model_factory.py
-----------------------
Factory function that returns a pretrained torchvision model with its
final classification layer replaced to match num_classes.

Supported architectures (controlled via config.MODEL_NAME):
    - resnet50
    - resnet34
    - mobilenet  (MobileNetV2)
"""

import logging
from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)

# Maps config string → (constructor, last-layer attribute, in_features getter)
_MODEL_REGISTRY = {
    "resnet50":  "resnet50",
    "resnet34":  "resnet34",
    "mobilenet": "mobilenet_v2",
}


def get_model(model_name: str, num_classes: int) -> nn.Module:
    """
    Load a pretrained torchvision model and swap the final FC / classifier
    layer to output `num_classes` logits.

    Args:
        model_name  : One of "resnet50", "resnet34", "mobilenet".
        num_classes : Number of target output classes.

    Returns:
        A modified nn.Module ready for fine-tuning.

    Raises:
        ValueError: If model_name is not in the supported list.
    """
    name = model_name.lower().strip()
    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {list(_MODEL_REGISTRY.keys())}"
        )

    logger.info(f"Loading pretrained model: '{name}' …")

    if name in ("resnet50", "resnet34"):
        model = _build_resnet(name, num_classes)
    elif name == "mobilenet":
        model = _build_mobilenet(num_classes)
    else:
        raise ValueError(f"Unhandled model name: {name}")

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model '{name}' ready — "
        f"total params: {total_params:,} | "
        f"trainable: {trainable_params:,} | "
        f"output classes: {num_classes}"
    )
    return model


# ─────────────────────────────────────────────
# Architecture Builders
# ─────────────────────────────────────────────

def _build_resnet(name: str, num_classes: int) -> nn.Module:
    """ResNet-34 / ResNet-50 with replaced FC head."""
    weights_map = {
        "resnet50": models.ResNet50_Weights.IMAGENET1K_V1,
        "resnet34": models.ResNet34_Weights.IMAGENET1K_V1,
    }
    constructor = getattr(models, name)
    model = constructor(weights=weights_map[name])

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    logger.debug(f"ResNet FC replaced: {in_features} → {num_classes}")
    return model


def _build_mobilenet(num_classes: int) -> nn.Module:
    """MobileNetV2 with replaced classifier head."""
    model = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
    )
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes),
    )
    logger.debug(f"MobileNetV2 classifier replaced: {in_features} → {num_classes}")
    return model


# ─────────────────────────────────────────────
# Target Layer Helper (used by GradCAM)
# ─────────────────────────────────────────────

def get_gradcam_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """
    Returns the last convolutional layer of the model, which is the
    standard hook point for Grad-CAM visualisation.

    Args:
        model      : The nn.Module returned by get_model().
        model_name : Config string identifying the architecture.

    Returns:
        The target nn.Module layer.
    """
    name = model_name.lower().strip()
    if name in ("resnet50", "resnet34"):
        return model.layer4[-1]          # last BasicBlock / Bottleneck
    elif name == "mobilenet":
        return model.features[-1]        # last ConvBNActivation block
    else:
        raise ValueError(f"No GradCAM target defined for model '{name}'.")
