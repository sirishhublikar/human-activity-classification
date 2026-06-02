# Radar-Based Human Activity Recognition

## Overview

This project implements a Deep Convolutional Neural Network (DCNN) for classifying human activities from radar data. Six activity classes are recognised: **Walking**, **Sitting**, **Standing**, **Drink water**, **Pick up**, and **Fall**.

---

## Pipeline Summary

### Preprocessing

- Raw `.dat` files are processed by `preprocess_all.py` (shared between DCNN and classical ML pipelines)
- Signal processing pipeline per recording:
  1. IQ imbalance correction (Gram-Schmidt orthogonalisation)
  2. Mean removal on I and Q channels
  3. Range FFT (rectangular window, fftshift, positive half retained)
  4. MTI high-pass Butterworth filter (order 4, Wn = 0.0075) — removes static clutter
  5. DC range bin removal
  6. STFT summed over range bins 10-30 (1-indexed) — produces micro-Doppler spectrogram
- Output: `spectrograms.npy` (object array, ragged), `t_axes.npy`, `labels.npy`, `persons.npy`, `repetitions.npy`

### Data Split

- Train / Test split: **80% / 20%** (subject-independent)
- Split applied after tiling, using per-tile person IDs
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

| Fold | Accuracy | Macro F1 |
|------|----------|----------|
| 1    | 85.52%   | 0.8312   |
| 2    | 84.71%   | 0.8217   |
| 3    | 83.91%   | 0.8136   |
| 4    | 84.28%   | 0.8214   |
| 5    | 83.84%   | 0.8145   |

- **Mean accuracy: 84.45%** (±0.62%)
- **Mean Macro F1: 0.8205** (±0.0063)

### Test Set

- Test subjects: 14 out of 72
- Test tiles: 1760
- Overall accuracy: **84.32%**
- Macro F1: **0.8199**
- Weighted F1: **0.84**

### Final Model (Trained on All Training Data)

- Best validation accuracy: **84.35%**
- Early stopping at epoch 74 of 100

---

## Per-Class Metrics (Test Set)

| Class       | Precision | Recall | F1   | Support |
|-------------|-----------|--------|------|---------|
| Walking     | 0.98      | 0.98   | 0.98 | 576     |
| Sitting     | 0.83      | 0.81   | 0.82 | 252     |
| Standing    | 0.83      | 0.80   | 0.81 | 252     |
| Drink water | 0.69      | 0.71   | 0.70 | 248     |
| Pick up     | 0.64      | 0.69   | 0.67 | 252     |
| Fall        | 0.97      | 0.92   | 0.94 | 180     |
