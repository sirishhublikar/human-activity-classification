"""
signal_processing.py  —  shared utility
=========================================
Converts a raw radar dict (from data_loader) into a
micro-Doppler spectrogram.

Mirrors DataProcessingExample.m step-by-step:
  1.  IQ imbalance correction (Gram-Schmidt orthogonalisation)
  2.  Mean removal on I and Q channels
  3.  Reshape IQ data into [NTS x nc] chirp matrix
  4.  Range FFT  (rectangular window, along fast-time axis)
  5.  High-pass Butterworth MTI filter  (order 4, Wn = 0.0075)
      — removes static clutter (walls, furniture)
  6.  STFT summed over range bins 10-30  -> micro-Doppler spectrogram
  7.  Otsu noise floor suppression

Default parameters match the MATLAB example exactly.

Usage:
  from signal_processing import compute_spectrogram
  spec, d_axis, t_axis = compute_spectrogram(radar_dict)
"""

import numpy as np
from scipy.signal import butter, lfilter, stft as _scipy_stft


# -- Default parameters ----------------

DEFAULT_PARAMS = dict(
    time_window    = 200,    # STFT window length in chirps
    overlap_factor = 0.95,   # STFT overlap fraction  -> noverlap = 190
    pad_factor     = 4,      # zero-padding  -> nfft = 800
    bin_indl       = 10,     # lower range bin (1-indexed, after DC removal)
    bin_indu       = 30,     # upper range bin (1-indexed, after DC removal)
)


# -- Public API ----------------------------------------------------------------

def iq_correction(i_raw, q_raw):
    """
    IQ imbalance correction via Gram-Schmidt orthogonalisation.

    Parameters
    ----------
    i_raw : ndarray  raw I (in-phase) samples, 1-D float
    q_raw : ndarray  raw Q (quadrature) samples, 1-D float

    Returns
    -------
    corrected : ndarray complex128, shape same as i_raw
    """
    i_norm     = i_raw / (np.std(i_raw) + 1e-12)
    q_norm     = q_raw / (np.std(q_raw) + 1e-12)
    cross_corr = np.mean(i_norm * q_norm)
    phi        = np.arcsin(np.clip(cross_corr, -1.0, 1.0))
    a_imb      = np.std(q_raw) / (np.std(i_raw) + 1e-12)
    q_corr     = (q_raw - i_raw * np.sin(phi)) / (a_imb * np.cos(phi) + 1e-12)
    return i_raw + 1j * q_corr


def preprocess_iq(data):
    """
    Parameters
    ----------
    data : ndarray complex128, shape (NTS * nc,)
           Raw complex IQ samples from data_loader.load_dat_file()

    Returns
    -------
    corrected : ndarray complex128, same shape as data
    """
    i_raw = np.real(data).copy()
    q_raw = np.imag(data).copy()

    # Mean removal (removes DC offset on each channel)
    i_raw -= np.mean(i_raw)
    q_raw -= np.mean(q_raw)

    return iq_correction(i_raw, q_raw)

def otsu_threshold(spec):
    """
    Suppress the noise floor using Otsu's method.
    Operates in dB scale to find the threshold, then zeros out
    all pixels below it in linear scale.
    Implemented with numpy only -- no skimage dependency.
    """
    spec_db = 20 * np.log10(spec + 1e-10)

    # Otsu's method: find threshold that minimises intra-class variance
    pixel_counts, bin_edges = np.histogram(spec_db.ravel(), bins=256)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    total       = pixel_counts.sum()
    probs       = pixel_counts / total

    best_thresh, best_var = 0, -1
    for t_idx in range(1, len(bin_centres)):
        w0 = probs[:t_idx].sum()
        w1 = probs[t_idx:].sum()
        if w0 == 0 or w1 == 0:
            continue
        mu0 = (probs[:t_idx] * bin_centres[:t_idx]).sum() / w0
        mu1 = (probs[t_idx:] * bin_centres[t_idx:]).sum() / w1
        var_between = w0 * w1 * (mu0 - mu1) ** 2
        if var_between > best_var:
            best_var    = var_between
            best_thresh = bin_centres[t_idx]

    mask = spec_db >= best_thresh
    return spec * mask

def compute_spectrogram(radar, **kwargs):
    """
    Parameters
    ----------
    radar  : dict  output of data_loader.load_dat_file()
    kwargs : override any key in DEFAULT_PARAMS

    Returns
    -------
    spec    : ndarray float64, shape (nfft, n_segments)
              Linear magnitude sum over range bins.
              Row 0 = highest positive Doppler (after flipud).
    d_axis  : ndarray, shape (nfft,)   Doppler velocity in m/s
    t_axis  : ndarray, shape (n_segs,) Time in seconds
    """
    p = {**DEFAULT_PARAMS, **kwargs}

    fc     = radar['fc']
    Tsweep = radar['Tsweep']
    NTS    = radar['NTS']
    Bw     = radar['Bw']
    data   = radar['data']

    # -- 1. IQ correction + mean removal ---------------------------------------
    data_corr = preprocess_iq(data)

    # -- 2. Reshape into chirp matrix ------------------------------------------
    nc        = int(len(data_corr) / NTS)
    Data_time = data_corr[:NTS * nc].reshape(NTS, nc, order='F')   # [NTS x nc]

    # -- 3. Range FFT ----------------------------------------------------------
    # Rectangular window (ones) applied implicitly; FFT along fast-time axis
    tmp        = np.fft.fftshift(np.fft.fft(Data_time, axis=0), axes=0)
    # Keep positive range bins only 
    Data_range = tmp[NTS // 2:, :]  # [NTS//2 x nc]

    # -- 4. MTI filter ---------------------------------------------------------
    # Process all chirps 
    b, a = butter(4, 0.0075, btype='high')

    Data_range_MTI = np.zeros_like(Data_range)
    for k in range(Data_range.shape[0]):
        Data_range_MTI[k] = lfilter(b, a, Data_range[k, :])

    # -- 5. Remove DC range bin ------------------------------------------------
    Data_range_MTI = Data_range_MTI[1:, :]  # [NTS//2-1 x nc]

    # -- 6. STFT -> micro-Doppler spectrogram ----------------------------------
    tw   = p['time_window']                          # 200
    ov   = int(round(tw * p['overlap_factor']))      # 190
    nfft = p['pad_factor'] * tw                      # 800
    bl = p['bin_indl'] - 1                           # 9
    bu = p['bin_indu'] - 1                           # 29

    spec = None
    for rb in range(bl, min(bu + 1, Data_range_MTI.shape[0])):
        sig = Data_range_MTI[rb, :]

        # scipy stft
        _, _, Zxx = _scipy_stft(
            sig,
            window          = 'hamming',
            nperseg         = tw,
            noverlap        = ov,
            nfft            = nfft,
            return_onesided = False,   # full two-sided spectrum (complex IQ)
        )
        # fftshift along frequency axis
        Zxx_shifted = np.fft.fftshift(Zxx, axes=0)

        mag  = np.abs(Zxx_shifted)
        spec = mag if spec is None else spec + mag

    # flipud
    spec = np.flipud(spec)

    # -- Build axes ------------------------------------------------------------
    PRF     = 1.0 / Tsweep                           # 1000 Hz
    lambda_ = 3e8 / fc                               # wavelength (m)

    freq_axis = np.fft.fftshift(np.fft.fftfreq(nfft)) * PRF
    d_axis    = freq_axis * lambda_ / 2              # Hz -> m/s
    d_axis    = d_axis[::-1]                         # compensate for flipud

    n_segs = spec.shape[1]
    t_axis = np.linspace(0, nc / PRF, n_segs)

    # 7. Otsu noise floor suppression
    spec = otsu_threshold(spec)

    return spec, d_axis, t_axis