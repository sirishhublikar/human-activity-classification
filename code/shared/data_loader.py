"""
data_loader.py  
Loads raw Ancortek .dat radar files and parses KPXXAYYRZ filenames.

File format (INSHEP project, University of Glasgow):
  Stored as ASCII text, one value per line.
  Line 1 : fc       carrier frequency  (5.8e9 Hz)    - plain float
  Line 2 : Tsweep   chirp duration     (1 ms)        - plain float
  Line 3 : NTS      samples per chirp  (128)         - plain float
  Line 4 : Bw       bandwidth          (400e6 Hz)    - plain float
  Line 5+: complex IQ samples as MATLAB format       - e.g. '1812+1897i'
            Real part  = I channel (in-phase)
            Imag part  = Q channel (quadrature)

Filename convention: KPXXAYYRZ.dat
  K  = activity digit  1=walk 2=sit 3=stand 4=pick 5=drink 6=fall
  XX = person ID (01-72)
  YY = activity code   (same as K, redundant)
  Z  = repetition (1-3)

Usage:
  from data_loader import load_dataset, parse_filename
  dataset = load_dataset('../../data')
"""

import re
import numpy as np
from pathlib import Path
from collections import Counter


# ── Label map ─────────────────────────────────────────────────────────────

ACTIVITY_MAP = {
    1: 'walking',
    2: 'sitting',
    3: 'standing',
    4: 'pick_up',
    5: 'drink_water',
    6: 'fall',
}


# ── Public API ────────────────────────────────────────────────────────────

def parse_filename(filename):
    stem = Path(filename).stem.upper()
    m = re.search(r'P(\d+)A(\d+)R(\d+)', stem)
    if not m:
        raise ValueError(f"Unrecognised filename: '{filename}'")

    p, a, r = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return dict(person=p, activity=a, repetition=r,
                activity_name=ACTIVITY_MAP.get(a, f'class_{a}'))


def load_dat_file(filepath):
    raw = _fast_load(Path(filepath))

    # Header: first 4 values are always real
    fc     = float(np.real(raw[0]))          # 5.8e9 Hz
    Tsweep = float(np.real(raw[1])) / 1000.0 # ms → s
    NTS    = int(np.real(raw[2]))             # 128
    Bw     = float(np.real(raw[3]))           # 400e6 Hz

    data = raw[4:]                            # complex IQ samples
    nc   = int(len(data) / NTS)
    fs   = NTS / Tsweep

    return dict(fc=fc, Tsweep=Tsweep, NTS=NTS, Bw=Bw,
                fs=fs, nc=nc, record_length=nc * Tsweep, data=data)


def load_dataset(data_dir, verbose=True):
    data_dir  = Path(data_dir)
    dat_files = sorted(data_dir.rglob('*.dat'))
    if not dat_files:
        raise FileNotFoundError(f"No .dat files found under '{data_dir}'")

    if verbose:
        print(f"Found {len(dat_files)} .dat files under {data_dir}")

    dataset, failures = [], []
    for i, fp in enumerate(dat_files):
        try:
            entry = {**parse_filename(fp.name), **load_dat_file(fp),
                     'filepath': str(fp)}
            dataset.append(entry)
        except Exception as exc:
            failures.append((fp.name, str(exc)))

        if verbose and (i + 1) % 200 == 0:
            print(f"  loaded {i+1}/{len(dat_files)} …")

    if verbose:
        _print_summary(dataset, failures, len(dat_files))
    return dataset


# ── Private helpers ────────────────────────────────────────────────────────

def _fast_load(filepath):
    with open(filepath, 'r', errors='replace') as f:
        content = f.read()

    # '1812+1897i' → '1812+1897j',   '5800000000.0' → unchanged (no 'i')
    content = content.replace('i', 'j')

    tokens = content.split()

    if len(tokens) < 5:
        raise ValueError(
            f"Only {len(tokens)} values in {filepath.name} - "
            f"file may be empty or corrupt."
        )

    try:
        # numpy parses '1812+1897j' strings directly with dtype=complex
        return np.array(tokens, dtype=complex)
    except (ValueError, TypeError):
        # fallback: Python-level conversion (slower but handles edge cases)
        vals = []
        for t in tokens:
            try:
                vals.append(complex(t) if 'j' in t else float(t))
            except ValueError:
                pass  # skip any malformed tokens
        return np.array(vals, dtype=complex)


def _print_summary(dataset, failures, total):
    print(f"\nLoaded {len(dataset)}/{total}  ({len(failures)} failed)")
    if failures:
        for name, err in failures[:5]:
            print(f"  FAIL: {name}: {err}")
    counts = Counter(d['activity_name'] for d in dataset)
    print("Class distribution:")
    for name, n in sorted(counts.items()):
        print(f"  {name:<15}  {n:4d}")
    print()
