"""
preprocess_all.py  —  shared preprocessing script
===================================================
Runs the full signal processing pipeline on every .dat file and saves
results to preprocessed/ so both pipelines (classical ML and DCNN) load
the same data.

Uses multiprocessing to run on all CPU cores — expect 5-20 min
depending on hardware (vs 1-2 hrs single-threaded).

Output files in --out_dir:
  spectrograms.npy   object array  (N,)  each entry float32 (800, T_i)
                     T_i varies per recording — ragged array
  t_axes.npy         object array  (N,)  each entry float64 (T_i,)
  labels.npy         int8          (N,)  activity class 1-6
  persons.npy        int16         (N,)  subject ID  (for subject-independent split)
  repetitions.npy    int8          (N,)  repetition number
  filepaths.txt                          one filepath per line

Usage:
  python preprocess_all.py
  python preprocess_all.py --data_dir ../../data --out_dir ../../preprocessed
  python preprocess_all.py --workers 4   # limit CPU cores

Loading in DCNN notebook:
  import numpy as np
  specs   = np.load('../../preprocessed/spectrograms.npy', allow_pickle=True)
  t_axes  = np.load('../../preprocessed/t_axes.npy',       allow_pickle=True)
  labels  = np.load('../../preprocessed/labels.npy')
  persons = np.load('../../preprocessed/persons.npy')

Loading in classical ML notebook:
  import numpy as np
  specs   = np.load('../../preprocessed/spectrograms.npy', allow_pickle=True)
  labels  = np.load('../../preprocessed/labels.npy')
  persons = np.load('../../preprocessed/persons.npy')
  # t_axes not needed for classical ML
"""

import argparse
from multiprocessing import pool
import sys
import time
import numpy as np
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

sys.path.insert(0, str(Path(__file__).parent))
from data_loader       import load_dat_file, parse_filename, ACTIVITY_MAP
from signal_processing import compute_spectrogram


# ── Worker function (runs in separate process) ────────────────────────────

def _process_file(filepath):
    """
    Load one .dat file, run the full signal processing pipeline.
    Returns (spec float32, t_axis float64, label, person, repetition, filepath_str)
    or raises on error.

    spec shape is (800, T_i) where T_i depends on recording length.
    No resizing — raw shape is preserved so downstream code can use
    t_axis correctly (e.g. for spectrogram tiling in DCNN pipeline).
    """
    fp    = Path(filepath)
    meta  = parse_filename(fp.name)
    radar = load_dat_file(fp)

    spec, _, t_axis = compute_spectrogram(radar)

    return (spec.astype(np.float32),
            t_axis.astype(np.float64),
            meta['activity'],
            meta['person'],
            meta['repetition'],
            str(filepath))


# ── Main ──────────────────────────────────────────────────────────────────

def main(data_dir='../../data', out_dir='../../preprocessed',
         n_workers=None, verbose=True):

    out_dir  = Path(out_dir)
    data_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dat_files = sorted(data_dir.rglob('*.dat'))
    if not dat_files:
        raise FileNotFoundError(f"No .dat files found under {data_dir}")

    N = len(dat_files)
    print(f"Found {N} .dat files")
    print(f"Output  -> {out_dir.resolve()}")
    print(f"Workers : {n_workers or 'all cores'}\n")

    specs_list  = []
    t_axes_list = []
    labels      = []
    persons     = []
    reps        = []
    paths       = []
    n_done = n_failed = 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        worker = partial(_process_file)
        futures = {pool.submit(worker, str(fp)): fp for fp in dat_files}

        for future in as_completed(futures):
            fp = futures[future]
            try:
                spec, t_axis, lbl, person, rep, path = future.result()
                specs_list.append(spec)
                t_axes_list.append(t_axis)
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
                elapsed = time.time() - t0
                rate    = done_total / elapsed if elapsed > 0 else 0
                eta     = (N - done_total) / rate if rate > 0 else 0
                print(f"  {done_total}/{N}  "
                      f"({n_done} ok, {n_failed} failed)  "
                      f"elapsed {elapsed/60:.1f} min  "
                      f"ETA {eta/60:.1f} min",
                      flush=True)

    # ── Stack and save ────────────────────────────────────────────────────
    # Spectrograms are ragged (variable T_i per recording) so saved as
    # object arrays. Classical ML and DCNN pipelines both load these and
    # handle them per-recording in their own loops.
    specs_arr  = np.empty(len(specs_list),  dtype=object)
    t_axes_arr = np.empty(len(t_axes_list), dtype=object)
    for i, (s, t) in enumerate(zip(specs_list, t_axes_list)):
        specs_arr[i]  = s
        t_axes_arr[i] = t

    np.save(out_dir / 'spectrograms.npy',  specs_arr)
    np.save(out_dir / 't_axes.npy',        t_axes_arr)
    np.save(out_dir / 'labels.npy',        np.array(labels,  dtype=np.int8))
    np.save(out_dir / 'persons.npy',       np.array(persons, dtype=np.int16))
    np.save(out_dir / 'repetitions.npy',   np.array(reps,    dtype=np.int8))
    (out_dir / 'filepaths.txt').write_text('\n'.join(paths))

    total_time = time.time() - t0
    print(f"\n{'='*55}")
    print(f"Done in {total_time/60:.1f} min")
    print(f"Saved {n_done} spectrograms  ({n_failed} skipped)")
    print(f"spectrograms.npy : object array, shape ({n_done},)")
    print(f"t_axes.npy       : object array, shape ({n_done},)")

    # Summarise spectrogram shapes
    t_lengths = [s.shape[1] for s in specs_list]
    print(f"Time frames per recording: "
          f"min={min(t_lengths)}  max={max(t_lengths)}  "
          f"mean={int(np.mean(t_lengths))}")

    counts = Counter(labels)
    print("\nClass distribution:")
    for cls in sorted(counts):
        name = ACTIVITY_MAP.get(cls, f'class_{cls}')
        print(f"  {cls}  {name:<15}  {counts[cls]:4d}")
    print('='*55)


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess all .dat radar files -> .npy arrays'
    )
    parser.add_argument('--data_dir', default='../../data')
    parser.add_argument('--out_dir',  default='../../preprocessed')
    parser.add_argument('--workers',  type=int, default=None,
                        help='Number of CPU cores (default: all)')

    args = parser.parse_args()
    main(args.data_dir, args.out_dir, args.workers)