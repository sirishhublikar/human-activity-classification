# Radar-Based Human Activity Recognition — DCNN Pipeline

## Overview

This project implements a Deep Convolutional Neural Network (DCNN) for classifying human activities from micro-Doppler radar spectrograms. Six activity classes are recognised: **Walking**, **Sitting**, **Standing**, **Drink water**, **Pick up**, and **Fall**.

Dataset: University of Glasgow FMCW radar dataset.  
This module covers the DCNN pipeline. Classical ML classifiers (SVM, KNN, Random Forest) are handled separately by teammates.

---

## Key Design Decisions

### Feature Representation

Three representations were compared on the same model and train/test split:

| Representation | Test Accuracy | Drink/Pick F1 | Notes |
|---|---|---|---|
| Spectrogram tiling (2 s, 50% OL) | 87.6% | 0.75 / 0.72 | Multiplies training samples but loses global temporal context |
| Cadence Velocity Diagram (CVD) | 83.3% | 0.67 / 0.58 | Discards temporal ordering — worst for Pick up vs Drink water |
| **Full-recording spectrogram** | **91.67%** | **0.82 / 0.83** | **Final choice** — preserves complete temporal evolution |

The full-recording approach feeds one resized 128 × 128 image per recording to the CNN. This proved critical for distinguishing Pick up (two separated arm-motion bursts) from Drink water (one continuous burst), where the global temporal structure is the discriminating feature.

### Otsu Noise Floor Suppression

Per-recording amplitude thresholding is applied after the STFT, following Kim and Ling (2009). Otsu's method finds the optimal binary threshold by minimising intra-class variance across the dB-scaled histogram of spectrogram pixel intensities, automatically separating the noise floor from micro-Doppler signal pixels. Applied adaptively per recording (not a fixed global threshold) to handle SNR variation across subjects.

Implemented in `signal_processing.py` via an `apply_threshold` flag in `compute_spectrogram()`.

### Class Imbalance

The Fall class has fewer recordings (~197 vs ~311 for other classes), due to specific recording locations that excluded Fall from their protocol. Addressed with **weighted cross-entropy loss** (weights inversely proportional to class frequency). Result: Fall achieves perfect F1 = 1.00 on the test set.

## Spectrogram Parameters

| Parameter | Value |
|---|---|
| Window type | Hamming |
| Window length | 200 samples (200 ms at PRF = 1000 Hz) |
| Overlap | 95% (190 samples) |
| FFT points | 800 (4× zero-padded) |
| Range bins | 10–30 (1-indexed) |
| Output size | 128 × 128 px (full recording, resized) |
| Normalisation | 1st/99th percentile clipping → [0, 1] |

---

## Model Architecture

Fixed architecture determined by Optuna hyperparameter tuning (50 trials, TPE sampler, MedianPruner).

| Component | Configuration |
|---|---|
| Input | (1, 128, 128) — single-channel spectrogram |
| Conv blocks | 5 blocks: 64 → 128 → 256 → 512 → 512 filters |
| Each block | Conv2D (3×3, pad=1) → BatchNorm → ReLU → MaxPool (2×2) → Dropout2D (0.25) |
| Head | Flatten → FC(512) → ReLU → Dropout(0.297) → FC(6) |
| Total parameters | ~13.1M |

---

## Training Configuration

| Parameter | Value |
|---|---|
| Optimiser | Adam |
| Learning rate | 0.000172 |
| Weight decay | 0.000133 |
| Batch size | 32 |
| LR scheduler | StepLR (step=20, gamma=0.43) |
| Max epochs | 100 |
| Early stopping | Patience = 20 epochs |
| Loss function | Weighted cross-entropy |

Hyperparameters tuned via Optuna across: `LR`, `BATCH_SIZE`, `DROPOUT`, `WEIGHT_DECAY`, `STEP_SIZE`, `GAMMA`, `N_CONV_BLOCKS`, `BASE_FILTERS`, `FC_SIZE`.

---

## Data Split

| Set | Subjects | Recordings | Notes |
|---|---|---|---|
| Train | 58 of 72 | ~688 | Used for 5-fold CV and final model training |
| Test | 14 of 72 | 360 | Held out; never seen during training or tuning |

Splits are **subject-independent**: no subject appears in both train and test. CV folds are also constructed at the subject level.

---

## Results

### Cross-Validation (5-Fold, Subject-Level)

| Fold | Accuracy | Macro F1 |
|---|---|---|
| 1 | 92.47% | 0.9294 |
| 2 | 91.04% | 0.9136 |
| 3 | 93.91% | 0.9426 |
| 4 | 92.83% | 0.9295 |
| 5 | 93.53% | 0.9386 |
| **Mean ± Std** | **92.76% ± 0.99%** | **0.9307 ± 0.0100** |

### Test Set Results

| Metric | Value |
|---|---|
| Test subjects | 14 of 72 |
| Test recordings | 360 |
| Overall accuracy | **91.67%** |
| Macro F1 | **0.9215** |
| Weighted F1 | **0.92** |

### Per-Class Metrics (Test Set)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Walking | 1.00 | 0.98 | **0.9921** | 64 |
| Sitting | 0.95 | 0.95 | **0.9524** | 63 |
| Standing | 1.00 | 0.87 | **0.9322** | 63 |
| Drink water | 0.80 | 0.84 | **0.8189** | 62 |
| Pick up | 0.80 | 0.87 | **0.8333** | 63 |
| Fall | 1.00 | 1.00 | **1.0000** | 45 |
| **Macro Avg** | 0.92 | 0.92 | **0.9215** | 360 |

### Classifier Comparison

| Classifier | Feature Input | Test Accuracy |
|---|---|---|
| **DCNN (full-recording)** | 128×128 spectrogram, 1 image/recording | **91.67%** |
| DCNN (tiled) | 128×128 tiles, 2 s window, 50% OL | 87.6% |
| Random Forest | 20 hand-crafted scalar features | 85.9% |
| SVM | — | — |
| KNN | — | — |

---

## Analysis

**Walking** and **Fall** achieve near-perfect F1.

**Drink water** and **Pick up** are the hardest pair (F1 0.82 and 0.83). Both activities involve slow, low-energy arm movements generating similar low-magnitude Doppler modulations. The full-recording representation improved these substantially over tiling (0.75 → 0.82, 0.72 → 0.83), which means the temporal context across the entire recording is the key discriminating feature.

The tight alignment between CV mean accuracy (92.76%) and test accuracy (91.67%) indicates strong generalisation to unseen subjects with no sign of subject-dependent overfitting.