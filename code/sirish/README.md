# Radar-Based Human Activity Recognition

## Overview

This project implements a Deep Convolutional Neural Network (DCNN) for classifying human activities from radar data. Six activity classes are recognised: **Walking**, **Sitting**, **Standing**, **Drink water**, **Pick up**, and **Fall**.

---

## Pipeline Summary

### Data Split

- Train / Test split: **80% / 20%** (subject-independent)
- Test subjects: 14 out of 72

### Spectrogram Parameters

- Window type: Hamming
- Window length: 200 samples (200 ms)
- Overlap: 95% (190 samples)
- FFT points: 800 (4x zero-padded)
- Range bins: 10 to 30 (1-indexed)
- Observation window: 2 seconds (empirically selected)
- Tile overlap: 50%
- Output size: 128 x 128 pixels
- Normalisation: 1st/99th percentile clipping, scaled to [0, 1]

---

## Model Architecture

**DCNN (post-tuning)**

- Input: (1, 128, 128)
- Conv blocks: 5, with base filters doubling per block (64, 128, 256, 512, 512)
- Each block: Conv2D (3x3, padding=1) → BatchNorm → ReLU → MaxPool (2x2) → Dropout2D (0.25)
- Head: Flatten → FC(512) → ReLU → Dropout (0.297) → FC(6)
- Output activation: Softmax
- Total trainable parameters: ~13.1M

---

## Training Configuration

- Optimiser: Adam (lr=0.000172, weight decay=0.000133)
- Batch size: 32
- LR scheduler: StepLR (step=20, gamma=0.43)
- Max epochs: 100 with early stopping (patience=20)
- Loss function: Cross-entropy

---

## Results

### Cross-Validation (5-Fold, Stratified, Person-Independent)

- Mean accuracy: **87.89%** (±0.75%)
- Mean Macro F1: **0.8543** (±0.0078)

### Test Set

- Test subjects: 14 out of 72
- Test tiles: 1400
- Overall accuracy: **87.64%**
- Macro F1: **0.8512**
- Weighted F1: **0.88**

### Final Model (Trained on All Training Data)

- Best validation accuracy: **90.00%**
- Stopped at epoch 95 of 100

---

## Per-Class Metrics (Test Set)

- Walking: Precision 0.99 / Recall 0.97 / F1 0.98
- Fall: Precision 0.95 / Recall 0.91 / F1 0.93
- Sitting: Precision 0.86 / Recall 0.87 / F1 0.87
- Standing: Precision 0.89 / Recall 0.85 / F1 0.87
- Drink water: Precision 0.74 / Recall 0.76 / F1 0.75
- Pick up: Precision 0.69 / Recall 0.75 / F1 0.72

---

## Confusion Matrices

### CV Out-of-Fold Confusion Matrix

Values show normalised proportions with raw counts in parentheses.

| True \ Predicted | Walking | Sitting | Standing | Drink water | Pick up | Fall |
|---|---|---|---|---|---|---|
| **Walking** | **0.97** (2003) | 0.01 (18) | 0.00 (9) | 0.01 (11) | 0.01 (14) | 0.00 (9) |
| **Sitting** | 0.03 (20) | **0.87** (655) | 0.02 (18) | 0.02 (12) | 0.06 (46) | 0.00 (1) |
| **Standing** | 0.02 (13) | 0.01 (9) | **0.87** (648) | 0.03 (26) | 0.05 (40) | 0.01 (8) |
| **Drink water** | 0.01 (5) | 0.03 (24) | 0.04 (29) | **0.74** (552) | 0.17 (127) | 0.01 (10) |
| **Pick up** | 0.01 (4) | 0.05 (39) | 0.02 (17) | 0.14 (106) | **0.77** (576) | 0.00 (2) |
| **Fall** | 0.03 (13) | 0.00 (1) | 0.03 (13) | 0.03 (12) | 0.02 (11) | **0.89** (406) |

### Test Set Confusion Matrix

Values show normalised proportions with raw counts in parentheses.

| True \ Predicted | Walking | Sitting | Standing | Drink water | Pick up | Fall |
|---|---|---|---|---|---|---|
| **Walking** | **0.97** (496) | 0.01 (6) | 0.01 (5) | 0.00 (0) | 0.00 (2) | 0.01 (3) |
| **Sitting** | 0.02 (3) | **0.87** (165) | 0.00 (0) | 0.02 (4) | 0.09 (17) | 0.00 (0) |
| **Standing** | 0.01 (1) | 0.03 (5) | **0.85** (160) | 0.04 (7) | 0.07 (13) | 0.02 (3) |
| **Drink water** | 0.00 (0) | 0.04 (8) | 0.03 (5) | **0.76** (141) | 0.17 (32) | 0.00 (0) |
| **Pick up** | 0.00 (0) | 0.03 (6) | 0.04 (7) | 0.17 (33) | **0.75** (142) | 0.01 (1) |
| **Fall** | 0.00 (0) | 0.01 (2) | 0.02 (3) | 0.04 (6) | 0.01 (1) | **0.91** (123) |

---

## Key Observations

- **Walking** and **Fall** are the best-recognised classes (F1: 0.98 and 0.93), likely due to their distinctive radar signatures.
- **Drink water** and **Pick up** are the most confused pair, sharing similar arm-movement patterns that are harder to separate in the Doppler-range domain.
- CV and test metrics are closely aligned (accuracy: 87.89% vs 87.64%), indicating the model generalises well to unseen subjects.
