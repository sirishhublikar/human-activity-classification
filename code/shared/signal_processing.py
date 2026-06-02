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

Default parameters match the MATLAB example exactly.

Usage:
  from signal_processing import compute_spectrogram
  spec, d_axis, t_axis = compute_spectrogram(radar_dict)

  # Quick visualisation
  import matplotlib.pyplot as plt
  plt.imshow(20*np.log10(spec + 1e-10),
             aspect='auto', origin='upper',
             extent=[t_axis[0], t_axis[-1], d_axis[-1], d_axis[0]])
  plt.xlabel('Time (s)')
  plt.ylabel('Velocity (m/s)')
  plt.colorbar(label='dB')
  plt.show()
"""

import numpy as np
from scipy.signal import butter, lfilter, stft as _scipy_stft


# -- Default parameters (match MATLAB DataProcessingExample.m) ----------------

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

    Corrects two hardware impairments present in the raw ADC output:
      - Phase imbalance: I and Q channels are not perfectly 90 deg apart
      - Amplitude imbalance: I and Q channels have different gains

    Steps:
      1. Estimate phase imbalance  phi = arcsin(corr(I_norm, Q_norm))
      2. Estimate amplitude imbalance  a = std(Q) / std(I)
      3. Correct Q: Q_corr = (Q - I*sin(phi)) / (a*cos(phi))

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
    Mean removal and IQ imbalance correction on the raw flat IQ array.

    Must be called BEFORE reshaping into the chirp matrix so that
    the correction statistics are computed over the full recording.

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


def compute_spectrogram(radar, **kwargs):
    """
    Full pipeline: raw IQ -> micro-Doppler spectrogram.

    Pipeline stages (mirrors DataProcessingExample.m):
      1. IQ correction + mean removal  (preprocess_iq)
      2. Reshape into chirp matrix  [NTS x nc]
      3. Range FFT  (rectangular window, fftshift, keep positive half)
      4. MTI high-pass Butterworth filter  (order 4, Wn=0.0075)
      5. Remove DC range bin
      6. STFT summed over range bins -> micro-Doppler spectrogram

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
    # MATLAB: reshape(Data, [NTS nc])  uses column-major order
    nc        = int(len(data_corr) / NTS)
    Data_time = data_corr[:NTS * nc].reshape(NTS, nc, order='F')   # [NTS x nc]

    # -- 3. Range FFT ----------------------------------------------------------
    # Rectangular window (ones) applied implicitly; FFT along fast-time axis
    # fftshift centres DC at row NTS//2
    tmp        = np.fft.fftshift(np.fft.fft(Data_time, axis=0), axes=0)
    # Keep positive range bins only  (matches MATLAB: tmp(NTS/2+1:NTS,:))
    Data_range = tmp[NTS // 2:, :]                                 # [NTS//2 x nc]

    # -- 4. MTI filter ---------------------------------------------------------
    # Process all chirps (no clamping needed -- scipy lfilter has no even-length
    # restriction unlike MATLAB's filter())
    b, a = butter(4, 0.0075, btype='high')

    Data_range_MTI = np.zeros_like(Data_range)
    for k in range(Data_range.shape[0]):
        Data_range_MTI[k] = lfilter(b, a, Data_range[k, :])

    # -- 5. Remove DC range bin ------------------------------------------------
    # MATLAB: Data_range_MTI(2:end,:)  -- row 0 is DC, discard it
    Data_range_MTI = Data_range_MTI[1:, :]                         # [NTS//2-1 x nc]

    # -- 6. STFT -> micro-Doppler spectrogram ----------------------------------
    tw   = p['time_window']                          # 200
    ov   = int(round(tw * p['overlap_factor']))      # 190
    nfft = p['pad_factor'] * tw                      # 800

    # Convert 1-indexed MATLAB bin numbers -> 0-indexed Python
    # DC bin was removed, so MATLAB bin 10 -> Python index 9
    bl = p['bin_indl'] - 1                           # 9
    bu = p['bin_indu'] - 1                           # 29

    spec = None
    for rb in range(bl, min(bu + 1, Data_range_MTI.shape[0])):
        sig = Data_range_MTI[rb, :]

        # scipy stft (window='hamming' matches MATLAB spectrogram default)
        _, _, Zxx = _scipy_stft(
            sig,
            window          = 'hamming',
            nperseg         = tw,
            noverlap        = ov,
            nfft            = nfft,
            return_onesided = False,   # full two-sided spectrum (complex IQ)
        )
        # fftshift along frequency axis (matches MATLAB fftshift(...,1))
        Zxx_shifted = np.fft.fftshift(Zxx, axes=0)

        mag  = np.abs(Zxx_shifted)
        spec = mag if spec is None else spec + mag

    # flipud matches MATLAB display convention (positive Doppler at top)
    spec = np.flipud(spec)

    # -- Build axes ------------------------------------------------------------
    PRF     = 1.0 / Tsweep                           # 1000 Hz
    lambda_ = 3e8 / fc                               # wavelength (m)

    freq_axis = np.fft.fftshift(np.fft.fftfreq(nfft)) * PRF
    d_axis    = freq_axis * lambda_ / 2              # Hz -> m/s
    d_axis    = d_axis[::-1]                         # compensate for flipud

    n_segs = spec.shape[1]
    t_axis = np.linspace(0, nc / PRF, n_segs)

    return spec, d_axis, t_axis