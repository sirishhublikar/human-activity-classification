"""
feature_extraction.py  —  code/mathew/
=======================================
Extracts feature vectors from micro-Doppler spectrograms for SVM and KNN.

Two approaches are provided:
  1. Hand-crafted features  (~20 values, physically interpretable)
  2. PCA features           (data-driven, good baseline comparison)

Hand-crafted features are better for the report (shows radar understanding).
PCA features are faster to implement and often match or beat hand-crafted.
Running both and comparing is easy extra content for the 1-pager.

The Doppler axis is reconstructed from known radar parameters:
  fc = 5.8 GHz,  PRF = 1000 Hz,  nfft = 800
  velocity range ≈ ±12.9 m/s,  resolution ≈ 0.032 m/s

Usage:
  from feature_extraction import extract_all, build_pca_features, D_AXIS
  X_hand = extract_all(spectrograms)           # (N, 20)
  X_pca,  pca = build_pca_features(spectrograms, n_components=64, fit=True)
"""

import numpy as np
from sklearn.decomposition import PCA


# ── Fixed Doppler axis (same for every spectrogram) ───────────────────────

_FC    = 5.8e9          # carrier frequency (Hz)
_PRF   = 1000.0         # pulse repetition frequency = 1/Tsweep (Hz)
_NFFT  = 800            # STFT zero-padded length
_LAM   = 3e8 / _FC      # wavelength (m)

# Velocity axis in m/s — matches the flipud'd spectrogram row order
# Row 0 = highest positive velocity, Row 399 ≈ 0, Row 799 = most negative
_freq_axis = np.fft.fftshift(np.fft.fftfreq(_NFFT)) * _PRF   # Hz
D_AXIS     = (_freq_axis * _LAM / 2)[::-1]                    # m/s, flipped

# Zero-Doppler row index
_ZERO_IDX = np.argmin(np.abs(D_AXIS))   # ≈ 399


# ── Public API ────────────────────────────────────────────────────────────

def extract_all(spectrograms):
    """
    Extract hand-crafted features for every spectrogram.

    Parameters
    ----------
    spectrograms : ndarray, shape (N, 800, 500)

    Returns
    -------
    X : ndarray, shape (N, n_features)  — float32
    """
    return np.stack(
        [extract_one(spectrograms[i]) for i in range(len(spectrograms))],
        axis=0
    ).astype(np.float32)


def extract_one(spec):
    """
    Extract hand-crafted features from a single spectrogram.

    Parameters
    ----------
    spec : ndarray, shape (800, 500)

    Returns
    -------
    features : ndarray, shape (20,)

    Feature groups
    --------------
    Energy      [0-3]  : total, positive-Doppler, negative-Doppler, ratio
    Centroid    [4-8]  : mean, std, max, min, range of time-varying centroid
    Spread      [9-10] : mean and max Doppler spread over time
    Bandwidth   [11-13]: 5th / 95th percentile velocity, bandwidth
    Dynamics    [14-15]: motion duty cycle, energy variability
    Cadence     [16]   : dominant oscillation freq of centroid (normalised)
    Entropy     [17]   : spectral entropy of mean Doppler profile
    Magnitude   [18]   : mean absolute Doppler centroid
    Asymmetry   [19]   : (pos_energy - neg_energy) / total_energy
    """
    n_freq, n_time = spec.shape
    d = D_AXIS[:n_freq]          # trim if spec is shorter than 800 rows
    zero = np.argmin(np.abs(d))

    eps = 1e-10

    # Per-column (time-frame) normalised spectrum for centroid calc
    col_sum  = spec.sum(axis=0, keepdims=True) + eps
    spec_n   = spec / col_sum                                   # (800, 500)

    # Time-varying Doppler centroid (weighted mean velocity per frame)
    centroid = (spec_n * d[:, None]).sum(axis=0)                # (500,)

    # Time-varying Doppler spread (weighted std per frame)
    spread   = np.sqrt(
        (spec_n * (d[:, None] - centroid[None, :]) ** 2).sum(axis=0)
    )                                                            # (500,)

    # Mean spectral profile (averaged over time)
    mean_prof = spec.mean(axis=1)                               # (800,)

    # ── 1. Energy ─────────────────────────────────────────────────────
    total_e = spec.sum()
    pos_e   = spec[:zero, :].sum()    # positive velocity rows (top of spec)
    neg_e   = spec[zero:, :].sum()    # negative velocity rows

    feat_energy = [
        np.log1p(total_e),
        np.log1p(pos_e),
        np.log1p(neg_e),
        pos_e / (neg_e + eps),
    ]

    # ── 2. Centroid statistics ─────────────────────────────────────────
    feat_centroid = [
        centroid.mean(),
        centroid.std(),
        centroid.max(),
        centroid.min(),
        centroid.max() - centroid.min(),
    ]

    # ── 3. Spread statistics ───────────────────────────────────────────
    feat_spread = [
        spread.mean(),
        spread.max(),
    ]

    # ── 4. Bandwidth (percentile-based) ───────────────────────────────
    cum_e = np.cumsum(mean_prof)
    total = cum_e[-1] + eps
    i5    = np.searchsorted(cum_e, 0.05 * total)
    i95   = np.searchsorted(cum_e, 0.95 * total)
    i5    = min(i5,  len(d) - 1)
    i95   = min(i95, len(d) - 1)
    feat_bw = [
        float(d[i5]),
        float(d[i95]),
        float(d[i95] - d[i5]),
    ]

    # ── 5. Dynamics ───────────────────────────────────────────────────
    frame_e  = spec.sum(axis=0)                           # energy per frame
    threshold = np.percentile(frame_e, 40)
    duty      = (frame_e > threshold).mean()              # fraction with motion
    cv_e      = frame_e.std() / (frame_e.mean() + eps)    # coefficient of variation
    feat_dyn = [duty, cv_e]

    # ── 6. Cadence (periodicity of Doppler centroid) ──────────────────
    c_detrended = centroid - centroid.mean()
    c_fft       = np.abs(np.fft.rfft(c_detrended))
    # Skip DC (index 0); find dominant frequency above 0.01 * len
    search_start = max(1, int(0.01 * len(c_detrended)))
    peak_idx     = np.argmax(c_fft[search_start:]) + search_start
    # Normalise to [0, 0.5]
    cadence_norm = peak_idx / len(c_detrended)
    feat_cadence = [cadence_norm]

    # ── 7. Spectral entropy ───────────────────────────────────────────
    p       = mean_prof / (mean_prof.sum() + eps)
    entropy = -np.sum(p * np.log(p + eps))
    feat_entropy = [entropy]

    # ── 8. Mean absolute Doppler ───────────────────────────────────────
    feat_mag = [np.abs(centroid).mean()]

    # ── 9. Velocity asymmetry ─────────────────────────────────────────
    feat_asym = [(pos_e - neg_e) / (total_e + eps)]

    features = (feat_energy + feat_centroid + feat_spread +
                feat_bw + feat_dyn + feat_cadence +
                feat_entropy + feat_mag + feat_asym)

    return np.array(features, dtype=np.float32)


def build_pca_features(spectrograms, n_components=64, pca=None):
    """
    Flatten spectrograms and reduce with PCA.

    Parameters
    ----------
    spectrograms  : ndarray  (N, 800, 500)
    n_components  : int      number of PCA components to keep
    pca           : fitted sklearn PCA, or None to fit a new one

    Returns
    -------
    X   : ndarray  (N, n_components)
    pca : fitted PCA object  (pass back in for test-set transforms)
    """
    N = len(spectrograms)
    # Handle ragged object arrays by truncating/padding to fixed size
    if spectrograms.dtype == object:
        TARGET_ROWS, TARGET_COLS = 800, 500
        fixed = np.zeros((N, TARGET_ROWS, TARGET_COLS), dtype=np.float32)
        for i, s in enumerate(spectrograms):
            r = min(s.shape[0], TARGET_ROWS)
            c = min(s.shape[1], TARGET_COLS)
            fixed[i, :r, :c] = s[:r, :c]
        X_flat = fixed.reshape(N, -1)
    else:
        X_flat = spectrograms.reshape(N, -1).astype(np.float32)

    if pca is None:
        print(f"Fitting PCA ({n_components} components) on {N} samples …")
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X_flat)
        var_explained = pca.explained_variance_ratio_.sum()
        print(f"  Variance explained: {var_explained:.1%}")
    else:
        X_pca = pca.transform(X_flat)

    return X_pca.astype(np.float32), pca


def get_feature_names():
    """Return names for each hand-crafted feature (for inspection/plots)."""
    return [
        'log_total_energy', 'log_pos_energy', 'log_neg_energy', 'pos_neg_ratio',
        'centroid_mean', 'centroid_std', 'centroid_max', 'centroid_min', 'centroid_range',
        'spread_mean', 'spread_max',
        'vel_p5', 'vel_p95', 'bandwidth',
        'motion_duty', 'energy_cv',
        'cadence',
        'spectral_entropy',
        'mean_abs_doppler',
        'velocity_asymmetry',
    ]
