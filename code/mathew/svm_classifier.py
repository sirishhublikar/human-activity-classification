"""
svm_classifier.py  —  code/mathew/
=====================================
Full SVM classification pipeline:
  1. Load preprocessed spectrograms from  preprocessed/
  2. Extract hand-crafted features  (and optionally PCA features)
  3. Subject-independent train/test split
  4. Standardise features
  5. GridSearchCV to find best SVM hyperparameters
  6. Train final model, evaluate on held-out test set
  7. Save model to  trained_models/
  8. Save confusion matrix + results to  results/

Usage:
  python code/mathew/svm_classifier.py
  python code/mathew/svm_classifier.py --features pca   # use PCA instead
  python code/mathew/svm_classifier.py --features both  # run both, compare
"""

import argparse
import sys
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.svm            import SVC
from sklearn.preprocessing  import StandardScaler
from sklearn.pipeline       import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics        import (classification_report,
                                    confusion_matrix,
                                    ConfusionMatrixDisplay)

# ── paths ─────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[2]
PREPROCESSED = ROOT / 'preprocessed'
RESULTS      = ROOT / 'results' / 'confusion_matrices'
MODELS       = ROOT / 'trained_models'

sys.path.insert(0, str(ROOT / 'code' / 'mathew'))
from feature_extraction import extract_all, build_pca_features, get_feature_names

# ── class labels ──────────────────────────────────────────────────────────
CLASS_NAMES = {1: 'walking', 2: 'sitting', 3: 'standing',
               4: 'pick_up', 5: 'drink',   6: 'fall'}


# ── helpers ───────────────────────────────────────────────────────────────

def load_data():
    """Load spectrograms + labels + person IDs from preprocessed/."""
    print("Loading preprocessed data …")
    specs   = np.load(PREPROCESSED / 'spectrograms.npy')   # (N, 800, 500)
    labels  = np.load(PREPROCESSED / 'labels.npy')         # (N,)
    persons = np.load(PREPROCESSED / 'persons.npy')        # (N,)
    print(f"  Loaded {len(specs)} samples  "
          f"classes: {np.unique(labels)}  "
          f"subjects: {len(np.unique(persons))}")
    return specs, labels, persons


def subject_split(X, y, persons, test_frac=0.2, seed=42):
    """
    Split by subject — no subject appears in both train and test.
    This is the correct evaluation strategy to avoid data leakage.

    Holds out test_frac of all subjects; keeps remainder for training.
    Subjects are sorted then split deterministically (seed for shuffle).
    """
    rng           = np.random.default_rng(seed)
    unique_subs   = np.unique(persons)
    shuffled      = rng.permutation(unique_subs)
    n_test        = max(1, int(len(shuffled) * test_frac))
    test_subs     = set(shuffled[:n_test])

    test_mask  = np.array([p in test_subs  for p in persons])
    train_mask = ~test_mask

    print(f"\nSplit:  train={train_mask.sum()} samples "
          f"({len(unique_subs) - n_test} subjects)  |  "
          f"test={test_mask.sum()} samples ({n_test} subjects)")
    print(f"  Test subjects: {sorted(test_subs)}")

    return (X[train_mask], X[test_mask],
            y[train_mask], y[test_mask])


def plot_confusion_matrix(cm, title, save_path):
    """Save confusion matrix as a PNG."""
    labels = [CLASS_NAMES[c] for c in sorted(CLASS_NAMES)]
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=labels)
    disp.plot(ax=ax, colorbar=True, cmap='Blues')
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def run_svm(X_tr, X_te, y_tr, y_te, tag='handcrafted'):
    """
    Train SVM with GridSearchCV and evaluate on test set.

    Parameters
    ----------
    tag : str   label used for filenames  ('handcrafted' or 'pca')
    """
    RESULTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True,  exist_ok=True)

    # ── Pipeline: scaler + SVM ────────────────────────────────────────
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svm',    SVC(probability=True, random_state=42)),
    ])

    # ── Hyperparameter grid ───────────────────────────────────────────
    # RBF kernel is almost always best for spectral features
    # Poly kernel included for comparison
    param_grid = [
        {
            'svm__kernel': ['rbf'],
            'svm__C':      [0.1, 1, 10, 100],
            'svm__gamma':  ['scale', 'auto', 0.01, 0.001],
        },
        {
            'svm__kernel': ['poly'],
            'svm__C':      [0.1, 1, 10],
            'svm__degree': [2, 3],
            'svm__gamma':  ['scale'],
        },
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"\n[SVM / {tag}]  GridSearchCV  (5-fold CV on training set) …")
    t0     = time.time()
    search = GridSearchCV(pipe, param_grid, cv=cv,
                          scoring='accuracy', n_jobs=-1, verbose=1)
    search.fit(X_tr, y_tr)
    print(f"  Done in {(time.time()-t0)/60:.1f} min")
    print(f"  Best CV accuracy : {search.best_score_:.3f}")
    print(f"  Best params      : {search.best_params_}")

    # ── Evaluate on test set ──────────────────────────────────────────
    best_model = search.best_estimator_
    y_pred     = best_model.predict(X_te)
    acc        = (y_pred == y_te).mean()

    print(f"\n  Test accuracy : {acc:.3f}")
    print("\n" + classification_report(
        y_te, y_pred,
        target_names=[CLASS_NAMES[c] for c in sorted(CLASS_NAMES)]
    ))

    # ── Confusion matrix ──────────────────────────────────────────────
    cm = confusion_matrix(y_te, y_pred,
                          labels=sorted(CLASS_NAMES.keys()))
    plot_confusion_matrix(
        cm,
        title=f"SVM ({tag})  —  test accuracy {acc:.1%}",
        save_path=RESULTS / f'svm_{tag}_cm.png'
    )

    # ── Save model ────────────────────────────────────────────────────
    model_path = MODELS / f'svm_{tag}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({'model': best_model, 'best_params': search.best_params_,
                     'cv_score': search.best_score_, 'test_acc': acc}, f)
    print(f"  Saved model: {model_path}")

    return acc, search.best_params_


# ── Main ──────────────────────────────────────────────────────────────────

def main(feature_mode='handcrafted'):
    specs, labels, persons = load_data()

    results = {}

    if feature_mode in ('handcrafted', 'both'):
        print("\nExtracting hand-crafted features …")
        t0 = time.time()
        X = extract_all(specs)
        print(f"  Shape: {X.shape}  ({time.time()-t0:.1f}s)")
        print(f"  Features: {get_feature_names()}")

        X_tr, X_te, y_tr, y_te = subject_split(X, labels, persons)
        acc, params = run_svm(X_tr, X_te, y_tr, y_te, tag='handcrafted')
        results['handcrafted'] = acc

    if feature_mode in ('pca', 'both'):
        print("\nExtracting PCA features …")
        # Fit PCA on training split only to avoid leakage
        # Quick subject split for indices
        rng    = np.random.default_rng(42)
        unique = rng.permutation(np.unique(persons))
        n_test = max(1, int(len(unique) * 0.2))
        test_s = set(unique[:n_test])
        tr_idx = np.array([i for i, p in enumerate(persons) if p not in test_s])
        te_idx = np.array([i for i, p in enumerate(persons) if p in test_s])

        X_pca_tr, fitted_pca = build_pca_features(
            specs[tr_idx], n_components=64, pca=None
        )
        X_pca_te, _ = build_pca_features(
            specs[te_idx], n_components=64, pca=fitted_pca
        )
        # Save PCA for later use by evaluate.py
        with open(MODELS / 'pca_transform.pkl', 'wb') as f:
            pickle.dump(fitted_pca, f)

        acc, params = run_svm(
            X_pca_tr, X_pca_te,
            labels[tr_idx], labels[te_idx],
            tag='pca'
        )
        results['pca'] = acc

    print("\n" + "="*50)
    print("Summary:")
    for name, acc in results.items():
        print(f"  SVM ({name}): {acc:.3f}")
    print("="*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--features',
        choices=['handcrafted', 'pca', 'both'],
        default='handcrafted',
        help='Feature type to use (default: handcrafted)'
    )
    args = parser.parse_args()
    main(args.features)
