# Changelog — Anemia Classification Project

All changes are logged here in reverse chronological order.
Format: `[Date] | File | What Changed | Why It Helps Accuracy`

---

## 2026-03-23 — Accuracy Improvement Pass (Target: >99%)

### `codebase/config.py`

| Parameter | Old Value | New Value | Reason |
|---|---|---|---|
| `EPOCHS` | 20 | **40** | More epochs let cosine LR converge fully |
| `LEARNING_RATE` | 0.001 | **0.0005** | Lower base LR = more stable fine-tuning |
| `EARLY_STOPPING_PATIENCE` | 5 | **10** | Cosine schedule dips before recovering; needs more patience |
| `IMAGE_SIZE / RESIZE_SIZE` | 224 (only) | **224 crop / 256 resize** | Resize to 256 then crop to 224 avoids aspect-ratio distortion |
| `AUG_ROTATION_DEGREES` | 20 | **30** | Wider rotation = more diverse training views |

**New flags added:**

| Flag | Value | Reason |
|---|---|---|
| `LABEL_SMOOTHING` | 0.1 | Prevents model from being overconfident on mislabelled/ambiguous images |
| `USE_COSINE_LR` | True | Cosine decay is smoother than plateau-stepping; reaches better minima |
| `LR_BACKBONE_FACTOR` | 0.1 | Backbone trains 10× slower than head — protects pretrained ImageNet features |
| `GRAD_CLIP_MAX_NORM` | 1.0 | Clips exploding gradients; prevents unstable training steps |
| `USE_MIXUP` | True | Blends two images + labels to regularise; strong for binary classification |
| `MIXUP_ALPHA` | 0.2 | Beta-distribution parameter for MixUp blend ratio |
| `APPLY_GAUSSIAN_BLUR` | True | Simulates photo blur variation (out-of-focus conjunctiva/palm images) |
| `AUG_GAUSSIAN_BLUR_PROB` | 0.3 | Applied 30% of the time during training |
| `AUG_GAUSSIAN_BLUR_SIGMA` | (0.1, 2.0) | Random blur strength range |

---

### `codebase/preprocess.py`

#### Training transforms
- **Replaced** `Resize(224, 224)` → `RandomResizedCrop(224, scale=0.7–1.0, ratio=0.75–1.33)`
  - *Why:* Randomly crops different parts of the image each epoch — the model sees the subject at varying scales and positions, improving spatial generalization.
- **Added** `GaussianBlur(kernel=5, sigma=0.1–2.0)` with `prob=0.3`
  - *Why:* Anemia is detected from subtle colour differences in conjunctiva/palm; the model should not overfit to sharp edge features that may not appear in real deployment photos.

#### Validation & Test transforms
- **Replaced** `Resize(224, 224)` → `Resize(256, 256)` → `CenterCrop(224)`
  - *Why:* Standard best practice (used by ImageNet eval). Avoids squashing images to a fixed square which distorts colour histograms; centre-crop keeps the subject centred.

---

### `codebase/train.py`

#### Label Smoothing
- `nn.CrossEntropyLoss()` → `nn.CrossEntropyLoss(label_smoothing=0.1)`
- *Why:* Prevents the model from assigning probability 1.0 to a class. Acts as regularisation — model generalises better on test data.

#### Differential Learning Rates
- Backbone params: `lr = LEARNING_RATE × LR_BACKBONE_FACTOR` = **0.00005**
- Head (fc) params: `lr = LEARNING_RATE` = **0.0005**
- *Why:* The backbone (ResNet50 body) already has excellent features from ImageNet. Training it too fast destroys these. The head needs a fast LR to learn the new task. This balance consistently improves final accuracy.

#### Cosine Annealing LR Scheduler
- Replaced `ReduceLROnPlateau` with `CosineAnnealingLR(T_max=EPOCHS, eta_min=1e-6)`
- *Why:* Smoothly brings LR from start → ~0 over all epochs. Avoids hard LR drops; reaches flatter, better-generalising minima. Val accuracy typically improves 0.5–1.5% over plateau scheduling.

#### Gradient Clipping
- Added `nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` before each `optimizer.step()`
- *Why:* With lower LR and cosine decay, gradients can occasionally spike. Clipping to norm=1.0 prevents those spikes from corrupting weights.

#### MixUp Regularisation
- New `mixup_data()` and `mixup_criterion()` helper functions added.
- Activated when `config.USE_MIXUP = True`.
- *Why:* Instead of training on individual images, the model sees a weighted blend of two images and must predict a blend of their labels. This acts as a very strong regulariser, reducing overfitting and improving test accuracy on small medical image datasets by 0.5–2%.

---

## Previous History

*(Older changes will be appended below as work continues)*

---
