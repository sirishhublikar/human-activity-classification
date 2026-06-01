"""
preprocess_all.py  —  shared preprocessing script
===================================================
Runs the full signal processing pipeline on every .dat file and saves
results to preprocessed/ so both team members load the same data.

Uses multiprocessing to run on all CPU cores — expect 5-20 min
depending on hardware (vs 1-2 hrs single-threaded).

Output files in --out_dir:
  spectrograms.npy   float32  (N, FREQ_BINS, TIME_BINS)
  labels.npy         int8     (N,)  activity class 1-6
  persons.npy        int16    (N,)  subject ID  (for subject-independent split)
  repetitions.npy    int8     (N,)  repetition number
  filepaths.txt                     one filepath per line

Usage:
  python code/shared/preprocess_all.py
  python code/shared/preprocess_all.py --data_dir data --out_dir preprocessed
  python code/shared/preprocess_all.py --workers 4  # limit CPU cores

Loading results (in classifier scripts):
  import numpy as np
  specs   = np.load('../../preprocessed/spectrograms.npy')  # (N, 800, 500)
  labels  = np.load('../../preprocessed/labels.npy')        # (N,)
  persons = np.load('../../preprocessed/persons.npy')       # (N,) for split
"""

import argparse
import sys
import time
import numpy as np
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent))
from data_loader       import load_dat_file, parse_filename, ACTIVITY_MAP
from signal_processing import compute_spectrogram


# ── Fixed output shape ────────────────────────────────────────────────────
FREQ_BINS = 800   # nfft (always 800 with default params)
TIME_BINS = 500   # truncate long / zero-pad short recordings


# ── Worker function (runs in separate process) ────────────────────────────

def _process_file(filepath):
    """
    Load one .dat file, compute spectrogram, resize to fixed shape.
    Returns (spec float32, label, person, repetition, filepath_str)
    or raises on error.

    This function is called in a worker process — no prints inside.
    """
    fp     = Path(filepath)
    labels = parse_filename(fp.name)
    radar  = load_dat_file(fp)
    spec, _, _ = compute_spectrogram(radar)
    spec_fixed = _resize(spec, FREQ_BINS, TIME_BINS).astype(np.float32)

    return (spec_fixed,
            labels['activity'],
            labels['person'],
            labels['repetition'],
            str(filepath))


def _resize(spec, target_f, target_t):
    """Pad or truncate spectrogram to (target_f, target_t)."""
    f, t = spec.shape

    # Frequency axis (should always be 800 — just a safety crop)
    if f > target_f:
        spec = spec[:target_f, :]
    elif f < target_f:
        spec = np.vstack([spec,
                          np.zeros((target_f - f, spec.shape[1]),
                                   dtype=spec.dtype)])
    # Time axis
    if t >= target_t:
        spec = spec[:, :target_t]
    else:
        spec = np.hstack([spec,
                          np.zeros((spec.shape[0], target_t - t),
                                   dtype=spec.dtype)])
    return spec


# ── Main ──────────────────────────────────────────────────────────────────

def main(data_dir='../../data', out_dir='../../preprocessed',
         n_workers=None, verbose=True):

    out_dir   = Path(out_dir)
    data_dir  = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dat_files = sorted(data_dir.rglob('*.dat'))
    if not dat_files:
        raise FileNotFoundError(f"No .dat files found under {data_dir}")

    N = len(dat_files)
    print(f"Found {N} .dat files")
    print(f"Output → {out_dir.resolve()}")
    print(f"Workers: {n_workers or 'all cores'}\n")

    specs_list, labels, persons, reps, paths = [], [], [], [], []
    n_done = n_failed = 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_process_file, str(fp)): fp
                   for fp in dat_files}

        for future in as_completed(futures):
            fp = futures[future]
            try:
                spec, lbl, person, rep, path = future.result()
                specs_list.append(spec)
                labels.append(lbl)
                persons.append(person)
                reps.append(rep)
                paths.append(path)
                n_done += 1
            except Exception as exc:
                n_failed += 1
                if verbose:
                    print(f"  WARN {fp.name}: {exc}")

            # Progress every 50 files
            done_total = n_done + n_failed
            if done_total % 50 == 0 or done_total == N:
                elapsed  = time.time() - t0
                rate     = done_total / elapsed if elapsed > 0 else 0
                eta      = (N - done_total) / rate if rate > 0 else 0
                print(f"  {done_total}/{N}  "
                      f"({n_done} ok, {n_failed} failed)  "
                      f"elapsed {elapsed/60:.1f} min  "
                      f"ETA {eta/60:.1f} min",
                      flush=True)

    # ── Stack and save ────────────────────────────────────────────────
    specs_arr = np.stack(specs_list, axis=0)

    np.save(out_dir / 'spectrograms.npy',  specs_arr)
    np.save(out_dir / 'labels.npy',        np.array(labels,  dtype=np.int8))
    np.save(out_dir / 'persons.npy',       np.array(persons, dtype=np.int16))
    np.save(out_dir / 'repetitions.npy',   np.array(reps,    dtype=np.int8))
    (out_dir / 'filepaths.txt').write_text('\n'.join(paths))

    total_time = time.time() - t0
    print(f"\n{'='*55}")
    print(f"Done in {total_time/60:.1f} min")
    print(f"Saved {n_done} spectrograms  ({n_failed} skipped)")
    print(f"Array shape : {specs_arr.shape}   dtype: {specs_arr.dtype}")

    counts = Counter(labels)
    print("\nClass distribution:")
    for cls in sorted(counts):
        name = ACTIVITY_MAP.get(cls, f'class_{cls}')
        print(f"  {cls}  {name:<15}  {counts[cls]:4d}")
    print('='*55)


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess all .dat radar files → .npy arrays'
    )
    parser.add_argument('--data_dir',  default='../../data')
    parser.add_argument('--out_dir',   default='../../preprocessed')
    parser.add_argument('--workers',   type=int, default=None,
                        help='Number of CPU cores (default: all)')
    args = parser.parse_args()
    main(args.data_dir, args.out_dir, args.workers)
