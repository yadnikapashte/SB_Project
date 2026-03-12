# Image Classification Pipeline

A modular, production-style PyTorch deep learning pipeline for multi-class image classification with GradCAM explainability.

---

## Project Structure

```
project/
├── config.py           # All hyperparameters and paths (edit here first)
├── main.py             # Single entry-point: runs the entire pipeline
├── preprocess.py       # Transform builders, dataset validation, class discovery
├── dataset.py          # PyTorch Dataset class + DataLoader factory
├── train.py            # Training loop with early stopping + LR scheduling
├── test.py             # Test-set inference
├── evaluate.py         # Metrics, confusion matrix, training curves
├── xai.py              # Grad-CAM explainability
├── requirements.txt
│
├── models/
│   ├── __init__.py
│   └── model_factory.py   # get_model() — supports ResNet50/34, MobileNetV2
│
├── utils/
│   ├── __init__.py
│   ├── metrics.py          # accuracy, precision, recall, F1 helpers
│   └── save_utils.py       # model / JSON / log persistence helpers
│
└── outputs/               # Auto-created by the pipeline
    ├── models/             # best_model.pth, final_model.pth
    ├── reports/            # confusion_matrix.png, training_curves.png,
    │                       # classification_report.txt, per_class_metrics.csv,
    │                       # summary_metrics.json, training_log.json
    ├── gradcam/            # GradCAM overlay figures
    ├── class_mapping.json  # {index: class_name} mapping
    └── pipeline.log        # Full run log
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare your data

Organise images into class-wise sub-folders for each split:

```
data/
  train/
    pneumonia/   *.jpg …
    normal/      *.jpg …
  val/
    pneumonia/
    normal/
  test/
    pneumonia/
    normal/
```

### 3. Configure the pipeline

Open `config.py` and set at minimum:

```python
TRAIN_DIR   = "data/train"
VAL_DIR     = "data/val"
TEST_DIR    = "data/test"
MODEL_NAME  = "resnet50"   # or "resnet34" | "mobilenet"
```

### 4. Run

```bash
# Run the entire pipeline with default 'data/' folder
python main.py

# Run with a custom data directory
python main.py --data_dir "path/to/custom_data"

# Run individual components with custom data directory
python train.py --data_dir "path/to/custom_data"
python evaluate.py --data_dir "path/to/custom_data"
python xai.py --data_dir "path/to/custom_data"
```

---

## Configuration Reference (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_NAME` | `"resnet50"` | Architecture — `"resnet50"`, `"resnet34"`, `"mobilenet"` |
| `EPOCHS` | `20` | Maximum training epochs |
| `BATCH_SIZE` | `32` | Images per batch |
| `LEARNING_RATE` | `0.001` | Initial Adam learning rate |
| `NUM_WORKERS` | `4` | DataLoader worker processes |
| `IMAGE_SIZE` | `224` | Resize target (height = width) |
| `IMAGES_PER_CLASS` | `800` | Per-class image cap (`None` = unlimited) |
| `USE_AUGMENTATION` | `True` | Random flips / rotation / colour jitter |
| `EARLY_STOPPING_PATIENCE` | `5` | Epochs without improvement before stopping |
| `GRADCAM_NUM_IMAGES` | `5` | Test images to explain with Grad-CAM |
| `DEVICE` | auto | `"cuda"` if available, else `"cpu"` |

> [!NOTE]
> You can override the base data directory at runtime using the `--data_dir`
> CLI argument in `main.py`, `train.py`, `evaluate.py`, or `xai.py`.

Changing `MODEL_NAME` automatically switches the architecture **and** the
correct Grad-CAM target layer — no other edits required.

---

## Outputs

| File | Description |
|------|-------------|
| `outputs/models/best_model.pth` | Weights at best validation loss |
| `outputs/models/final_model.pth` | Weights at last epoch |
| `outputs/class_mapping.json` | `{index: class_name}` mapping |
| `outputs/reports/confusion_matrix.png` | Count + normalised heatmaps |
| `outputs/reports/training_curves.png` | Loss + accuracy per epoch |
| `outputs/reports/per_class_metrics.csv` | Precision / recall / F1 per class |
| `outputs/reports/summary_metrics.json` | Macro + weighted aggregate scores |
| `outputs/reports/classification_report.txt` | Full sklearn report |
| `outputs/gradcam/` | Original / heatmap / overlay triptychs |
| `outputs/pipeline.log` | Timestamped full run log |

---

## Supported Models

| `MODEL_NAME` | Architecture | ImageNet Weights |
|---|---|---|
| `"resnet50"` | ResNet-50 | `ResNet50_Weights.IMAGENET1K_V1` |
| `"resnet34"` | ResNet-34 | `ResNet34_Weights.IMAGENET1K_V1` |
| `"mobilenet"` | MobileNetV2 | `MobileNet_V2_Weights.IMAGENET1K_V1` |

All models are loaded with pretrained ImageNet weights. The final
classification head is replaced and all parameters are fine-tuned.

---

## Design Principles

- **Single source of truth** — everything flows from `config.py`.
- **Fail fast** — dataset paths are validated before training starts.
- **Graceful degradation** — corrupt images are replaced with blank tensors so training continues.
- **Reproducibility** — class index ordering is deterministic (alphabetical sort).
- **Device agnostic** — runs on CPU or GPU with no code changes.
- **Logging over printing** — all output goes through Python's `logging` module and is mirrored to `pipeline.log`.
