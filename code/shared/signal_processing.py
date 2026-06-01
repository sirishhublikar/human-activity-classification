"""
signal_processing.py  —  shared utility
=========================================
Converts a raw radar dict (from data_loader) into a
micro-Doppler spectrogram.

Mirrors DataProcessingExample.m step-by-step:
  1.  Reshape IQ data into [NTS × nc] chirp matrix
  2.  Range FFT  (rectangular window, along fast-time axis)
  3.  High-pass Butterworth MTI filter  (order 4, Wn = 0.0075)
      — removes static clutter (walls, furniture)
  4.  STFT summed over range bins 10–30  → micro-Doppler spectrogram

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


# ── Default parameters (match MATLAB DataProcessingExample.m) ─────────────

DEFAULT_PARAMS = dict(
    time_window    = 200,    # STFT window length in chirps
    overlap_factor = 0.95,   # STFT overlap fraction  → noverlap = 190
    pad_factor     = 4,      # zero-padding  → nfft = 800
    bin_indl       = 10,     # lower range bin (1-indexed, after DC removal)
    bin_indu       = 30,     # upper range bin (1-indexed, after DC removal)
)


# ── Public API ─────────────────────────────────────────────────────────────

def compute_spectrogram(radar, **kwargs):
    """
    Full pipeline: raw IQ → micro-Doppler spectrogram.

    Parameters
    ----------
    radar  : dict  — output of data_loader.load_dat_file()
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

    # ── 1. Reshape into chirp matrix ──────────────────────────────────
    # MATLAB: reshape(Data, [NTS nc])  uses column-major order
    nc        = int(len(data) / NTS)
    Data_time = data[:NTS * nc].reshape(NTS, nc, order='F')   # [NTS × nc]

    # ── 2. Range FFT ──────────────────────────────────────────────────
    # Rectangular window (ones) applied implicitly; FFT along fast-time axis
    # fftshift centres DC at row NTS//2
    tmp        = np.fft.fftshift(np.fft.fft(Data_time, axis=0), axes=0)
    # Keep DC + positive range bins only  (matches MATLAB: tmp(NTS/2+1:NTS,:))
    Data_range = tmp[NTS // 2:, :]                             # [NTS//2 × nc]

    # ── 3. MTI filter ─────────────────────────────────────────────────
    # Clamp chirp count to an even number (matches MATLAB oddnumber logic)
    ns   = _oddnumber(nc) - 1                                  # ≤ nc, even
    b, a = butter(4, 0.0075, btype='high')

    Data_range_MTI = np.zeros((Data_range.shape[0], ns), dtype=complex)
    for k in range(Data_range.shape[0]):
        # lfilter → causal (matches MATLAB filter(), not filtfilt)
        Data_range_MTI[k] = lfilter(b, a, Data_range[k, :ns])

    # Remove DC range bin row 0  (MATLAB: Data_range_MTI(2:end,:))
    Data_range_MTI = Data_range_MTI[1:, :]                     # [NTS//2-1 × ns]

    # ── 4. STFT → micro-Doppler spectrogram ───────────────────────────
    tw   = p['time_window']                          # 200
    ov   = int(round(tw * p['overlap_factor']))      # 190
    nfft = p['pad_factor'] * tw                      # 800

    # Convert 1-indexed MATLAB bin numbers → 0-indexed Python
    # (DC bin was removed, so MATLAB bin 10 → Python index 9)
    bl = p['bin_indl'] - 1                           # 9
    bu = p['bin_indu'] - 1                           # 29

    spec = None
    for rb in range(bl, min(bu + 1, Data_range_MTI.shape[0])):
        sig = Data_range_MTI[rb, :]

        # scipy stft  (window='hamming' matches MATLAB spectrogram default)
        _, _, Zxx = _scipy_stft(
            sig,
            window          = 'hamming',
            nperseg         = tw,
            noverlap        = ov,
            nfft            = nfft,
            return_onesided = False,   # full two-sided spectrum (complex IQ)
        )
        # fftshift along frequency axis  (matches MATLAB fftshift(...,1))
        Zxx_shifted = np.fft.fftshift(Zxx, axes=0)

        mag = np.abs(Zxx_shifted)
        spec = mag if spec is None else spec + mag

    # flipud matches MATLAB display convention (positive Doppler at top)
    spec = np.flipud(spec)

    # ── Build axes ────────────────────────────────────────────────────
    PRF     = 1.0 / Tsweep                           # 1000 Hz
    lambda_ = 3e8 / fc                               # wavelength (m)

    # Frequency axis: centred, from -PRF/2 to +PRF/2
    freq_axis = np.fft.fftshift(np.fft.fftfreq(nfft)) * PRF
    d_axis    = freq_axis * lambda_ / 2              # Hz → m/s
    d_axis    = d_axis[::-1]                         # compensate for flipud

    n_segs = spec.shape[1]
    t_axis = np.linspace(0, ns / PRF, n_segs)

    return spec, d_axis, t_axis


# ── Private helpers ────────────────────────────────────────────────────────

def _oddnumber(x):
    """
    Nearest odd integer to x.
    Direct Python port of oddnumber.m.
    """
    y = int(np.floor(x))
    if y % 2 == 0:
        y = int(np.ceil(x))
    if y % 2 == 0:
        y += 1
    return y
