# knn_classifier.py
# KNN classification pipeline


import argparse
import sys
import time
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.neighbors       import KNeighborsClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics         import (classification_report,
                                     confusion_matrix,
                                     ConfusionMatrixDisplay)

# ── paths ─────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[2]
PREPROCESSED = ROOT / 'preprocessed'
RESULTS_CM   = ROOT / 'results' / 'confusion_matrices'
RESULTS_FI   = ROOT / 'results' / 'feature_importance'
MODELS       = ROOT / 'trained_models'

sys.path.insert(0, str(ROOT / 'code' / 'shared'))
from feature_extraction import (extract_all, build_pca_features,
                                 get_feature_names)
from utils import subject_split as _canonical_split

# ── class labels ──────────────────────────────────────────────────────────
CLASS_NAMES = {1: 'walking', 2: 'sitting', 3: 'standing',
               4: 'pick_up', 5: 'drink',   6: 'fall'}


# ── helpers ───────────────────────────────────────────────────────────────


def subject_split(X, y, persons, test_frac=0.2, seed=42):
    train_mask, test_mask = _canonical_split(persons, test_frac, seed)
    print(f"\nSplit:  train={train_mask.sum()}  |  test={test_mask.sum()}")
    return X[train_mask], X[test_mask], y[train_mask], y[test_mask]

def load_data():
    print("Loading preprocessed data …")
    specs   = np.load(PREPROCESSED / 'spectrograms.npy', allow_pickle=True)  # object array (N,), each (800, T_i)
    labels  = np.load(PREPROCESSED / 'labels.npy')                           # (N,)
    persons = np.load(PREPROCESSED / 'persons.npy')                          # (N,)
    print(f"  {len(specs)} samples  |  "
          f"classes: {np.unique(labels)}  |  "
          f"subjects: {len(np.unique(persons))}")
    return specs, labels, persons



def plot_confusion_matrix(cm, title, save_path):
    labels = [CLASS_NAMES[c] for c in sorted(CLASS_NAMES)]
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=True, cmap='Greens')
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_k_sensitivity(cv_results, save_path):
    params  = cv_results['params']
    scores  = cv_results['mean_test_score']

    # Group by metric
    metrics = sorted(set(p['knn__metric'] for p in params))
    weights = sorted(set(p['knn__weights'] for p in params))

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4),
                              sharey=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        for w in weights:
            mask = np.array([
                p['knn__metric'] == metric and p['knn__weights'] == w
                for p in params
            ])
            ks   = [p['knn__n_neighbors'] for p in np.array(params)[mask]]
            accs = scores[mask]
            order = np.argsort(ks)
            ax.plot(np.array(ks)[order], accs[order],
                    marker='o', label=f'weights={w}')
        ax.set_title(f'metric = {metric}')
        ax.set_xlabel('k (n_neighbors)')
        ax.set_ylabel('CV accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('KNN hyperparameter sensitivity', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def run_knn(X_tr, X_te, y_tr, y_te, tag='handcrafted'):
    """
    Train KNN with GridSearchCV and evaluate on test set.
    """
    RESULTS_CM.mkdir(parents=True, exist_ok=True)
    RESULTS_FI.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    # ── Pipeline: scaler + KNN ────────────────────────────────────────
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('knn',    KNeighborsClassifier(n_jobs=-1)),
    ])

    # ── Hyperparameter grid ───────────────────────────────────────────
    param_grid = {
        'knn__n_neighbors': [3, 5, 7, 9, 11, 15, 21],
        'knn__metric':      ['euclidean', 'manhattan', 'cosine'],
        'knn__weights':     ['uniform', 'distance'],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"\n[KNN / {tag}]  GridSearchCV  (5-fold CV on training set) …")
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
    cm = confusion_matrix(y_te, y_pred, labels=sorted(CLASS_NAMES.keys()))
    plot_confusion_matrix(
        cm,
        title=f"KNN ({tag})  —  test accuracy {acc:.1%}",
        save_path=RESULTS_CM / f'knn_{tag}_cm.png'
    )

    # ── k sensitivity plot ────────────────────────────────────────────
    plot_k_sensitivity(
        search.cv_results_,
        save_path=RESULTS_FI / f'knn_{tag}_k_sensitivity.png'
    )

    # ── Save model ────────────────────────────────────────────────────
    model_path = MODELS / f'knn_{tag}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model':       best_model,
            'best_params': search.best_params_,
            'cv_score':    search.best_score_,
            'test_acc':    acc,
        }, f)
    print(f"  Saved model: {model_path}")

    return acc, search.best_params_


# ── Main ──────────────────────────────────────────────────────────────────

def main(feature_mode='handcrafted'):
    specs, labels, persons = load_data()
    results = {}

    if feature_mode in ('handcrafted', 'both'):
        print("\nExtracting hand-crafted features …")
        t0 = time.time()
        X  = extract_all(specs)
        print(f"  Shape: {X.shape}  ({time.time()-t0:.1f}s)")

        X_tr, X_te, y_tr, y_te = subject_split(X, labels, persons)
        acc, params = run_knn(X_tr, X_te, y_tr, y_te, tag='handcrafted')
        results['handcrafted'] = acc

    if feature_mode in ('pca', 'both'):
        print("\nExtracting PCA features …")
        train_mask, test_mask = _canonical_split(persons)
        tr_idx = np.where(train_mask)[0]
        te_idx = np.where(test_mask)[0]

        # Reuse PCA fitted by SVM if available, else fit new
        pca_path = MODELS / 'pca_transform.pkl'
        if pca_path.exists():
            with open(pca_path, 'rb') as f:
                fitted_pca = pickle.load(f)
            X_pca_tr, _ = build_pca_features(specs[tr_idx],
                                              n_components=64,
                                              pca=fitted_pca)
        else:
            X_pca_tr, fitted_pca = build_pca_features(specs[tr_idx],
                                                       n_components=64)
        X_pca_te, _ = build_pca_features(specs[te_idx],
                                          n_components=64,
                                          pca=fitted_pca)

        acc, params = run_knn(
            X_pca_tr, X_pca_te,
            labels[tr_idx], labels[te_idx],
            tag='pca'
        )
        results['pca'] = acc

    print("\n" + "="*50)
    print("Summary:")
    for name, acc in results.items():
        print(f"  KNN ({name}): {acc:.3f}")
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
