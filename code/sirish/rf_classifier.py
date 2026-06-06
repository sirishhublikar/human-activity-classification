"""
rf_classifier.py
=====================================
Full Random Forest classification pipeline for Glasgow 848 dataset.

  1. Load preprocessed spectrograms from  preprocessed/
  2. Extract features  (hand-crafted or PCA)
  3. Subject-independent train/test split
  4. GridSearchCV over n_estimators, max_depth, min_samples_split, max_features
  5. Train final model, evaluate on held-out test set
  6. Save model to  trained_models/
  7. Save confusion matrix + feature importance to  results/

Note on scaling:
  Random Forest is a tree-based method and is invariant to monotonic feature
  transformations, so StandardScaler is NOT applied here (unlike SVM/KNN).
  This also means feature importances are directly comparable across features.

Note on PCA mode:
  PCA features work but lose physical interpretability, which is RF's main
  advantage over SVM/KNN. Hand-crafted mode is recommended for this classifier.

Usage:
  python code/sirish/rf_classifier.py
  python code/sirish/rf_classifier.py --features pca
  python code/sirish/rf_classifier.py --features both
"""

import argparse
import sys
import time
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics         import (classification_report,
                                     confusion_matrix,
                                     ConfusionMatrixDisplay)

# -- paths -----------------------------------------------------------------
ROOT         = Path(__file__).resolve().parents[2]
PREPROCESSED = ROOT / 'preprocessed'
RESULTS_CM   = ROOT / 'results' / 'confusion_matrices'
RESULTS_FI   = ROOT / 'results' / 'feature_importance'
MODELS       = ROOT / 'trained_models'

sys.path.insert(0, str(ROOT / 'code' / 'shared'))
from feature_extraction import (extract_all, build_pca_features,
                                 get_feature_names)

# -- class labels ----------------------------------------------------------
CLASS_NAMES = {1: 'walking', 2: 'sitting', 3: 'standing',
               4: 'pick_up', 5: 'drink',   6: 'fall'}


# -- helpers ---------------------------------------------------------------

def load_data():
    print("Loading preprocessed data ...")
    specs   = np.load(PREPROCESSED / 'spectrograms.npy', allow_pickle=True)
    labels  = np.load(PREPROCESSED / 'labels.npy',       allow_pickle=True)
    persons = np.load(PREPROCESSED / 'persons.npy',       allow_pickle=True)
    print(f"  {len(specs)} samples  |  "
          f"classes: {np.unique(labels)}  |  "
          f"subjects: {len(np.unique(persons))}")
    return specs, labels, persons


def subject_split(X, y, persons, test_frac=0.2, seed=42):
    """Subject-independent split"""
    rng         = np.random.default_rng(seed)
    unique_subs = np.unique(persons)
    shuffled    = rng.permutation(unique_subs)
    n_test      = max(1, int(len(shuffled) * test_frac))
    test_subs   = set(shuffled[:n_test])

    test_mask  = np.array([p in test_subs  for p in persons])
    train_mask = ~test_mask

    print(f"\nSplit:  train={train_mask.sum()}  |  test={test_mask.sum()}")
    return (X[train_mask], X[test_mask],
            y[train_mask], y[test_mask])


def plot_confusion_matrix(cm, title, save_path):
    labels = [CLASS_NAMES[c] for c in sorted(CLASS_NAMES)]
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=True, cmap='Oranges')
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_feature_importance(importances, feature_names, title, save_path):
    """
    Horizontal bar chart of RF feature importances (mean decrease in impurity).
    This is the key advantage of RF over SVM/KNN -- shows which physical
    properties of the micro-Doppler signature matter most for classification.
    """
    # Sort features by importance descending
    order = np.argsort(importances)[::-1]
    sorted_names  = [feature_names[i] for i in order]
    sorted_values = importances[order]

    fig, ax = plt.subplots(figsize=(9, 6))
    y_pos = np.arange(len(sorted_names))
    bars  = ax.barh(y_pos, sorted_values[::-1],
                    color='steelblue', edgecolor='white', height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names[::-1], fontsize=9)
    ax.set_xlabel('Mean decrease in impurity (normalised)', fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_n_estimators_sensitivity(cv_results, save_path):
    """
    Plot mean CV accuracy vs n_estimators for each max_features setting.
    Useful for the report -- shows how quickly RF converges as more trees are added.
    """
    params = cv_results['params']
    scores = cv_results['mean_test_score']

    max_features_vals = sorted(set(str(p['max_features']) for p in params))

    fig, ax = plt.subplots(figsize=(7, 4))
    for mf in max_features_vals:
        mask = np.array([str(p['max_features']) == mf for p in params])
        ns   = [p['n_estimators'] for p in np.array(params)[mask]]
        accs = scores[mask]
        order = np.argsort(ns)
        ax.plot(np.array(ns)[order], accs[order],
                marker='o', label=f'max_features={mf}')

    ax.set_xlabel('n_estimators')
    ax.set_ylabel('CV accuracy')
    ax.set_title('RF hyperparameter sensitivity: n_estimators vs accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")

def plot_diminishing_returns(cv_results, save_path):
    """
    Plot mean CV accuracy vs n_estimators aggregated across all other
    hyperparameter combinations. The shaded band shows +/- 1 std.
    The flattening of the curve visually shows diminishing returns.
    """
    params = cv_results['params']
    scores = cv_results['mean_test_score']

    n_vals = sorted(set(p['n_estimators'] for p in params))
    means, stds = [], []
    for n in n_vals:
        mask = np.array([p['n_estimators'] == n for p in params])
        means.append(scores[mask].mean())
        stds.append(scores[mask].std())

    means = np.array(means)
    stds  = np.array(stds)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(n_vals, means, marker='o', color='steelblue', label='mean CV accuracy')
    ax.fill_between(n_vals, means - stds, means + stds,
                    alpha=0.2, color='steelblue', label='±1 std')
    ax.set_xlabel('n_estimators')
    ax.set_ylabel('CV accuracy')
    ax.set_title('RF ensemble size: diminishing returns')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")

def run_rf(X_tr, X_te, y_tr, y_te, tag='handcrafted'):
    """
    Train Random Forest with GridSearchCV and evaluate on test set.

    No StandardScaler is used -- RF is scale-invariant and feature
    importances are more interpretable without any transformation.
    """
    RESULTS_CM.mkdir(parents=True, exist_ok=True)
    RESULTS_FI.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    # -- Hyperparameter grid -----------------------------------------------
    # n_estimators: more trees = more stable, diminishing returns after ~200
    # max_depth: None means trees grow until leaves are pure; limiting it
    #   reduces overfitting on small datasets like Glasgow 848
    # min_samples_split: controls minimum samples needed to split a node
    # max_features: 'sqrt' is the RF default and usually best for classification
    param_grid = {
        'n_estimators':    [100, 200, 300, 500],
        'max_depth':       [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'max_features':    ['sqrt', 'log2'],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',   # handles the Fall class imbalance
    )

    print(f"\n[RF / {tag}]  GridSearchCV  (5-fold CV on training set) ...")
    t0     = time.time()
    search = GridSearchCV(rf, param_grid, cv=cv,
                          scoring='accuracy', n_jobs=-1, verbose=1)
    search.fit(X_tr, y_tr)
    print(f"  Done in {(time.time()-t0)/60:.1f} min")
    print(f"  Best CV accuracy : {search.best_score_:.3f}")
    print(f"  Best params      : {search.best_params_}")

    # -- Evaluate on test set ----------------------------------------------
    best_model = search.best_estimator_
    y_pred     = best_model.predict(X_te)
    acc        = (y_pred == y_te).mean()

    print(f"\n  Test accuracy : {acc:.3f}")
    print("\n" + classification_report(
        y_te, y_pred,
        target_names=[CLASS_NAMES[c] for c in sorted(CLASS_NAMES)]
    ))

    # -- Confusion matrix --------------------------------------------------
    cm = confusion_matrix(y_te, y_pred, labels=sorted(CLASS_NAMES.keys()))
    plot_confusion_matrix(
        cm,
        title=f"RF ({tag})  --  test accuracy {acc:.1%}",
        save_path=RESULTS_CM / f'rf_{tag}_cm.png'
    )

    # -- Feature importance (hand-crafted mode only) -----------------------
    if tag == 'handcrafted':
        importances  = best_model.feature_importances_
        feature_names = get_feature_names()

        # Print ranked importances to console
        print("\n  Feature importances (ranked):")
        order = np.argsort(importances)[::-1]
        for rank, i in enumerate(order):
            print(f"    {rank+1:2d}. {feature_names[i]:<25s}  {importances[i]:.4f}")

        plot_feature_importance(
            importances,
            feature_names,
            title=f"RF feature importances ({tag})",
            save_path=RESULTS_FI / f'rf_{tag}_feature_importance.png'
        )

    # -- n_estimators sensitivity plot ------------------------------------
    plot_n_estimators_sensitivity(
        search.cv_results_,
        save_path=RESULTS_FI / f'rf_{tag}_n_estimators_sensitivity.png'
    )
    plot_diminishing_returns(
        search.cv_results_,
        save_path=RESULTS_FI / f'rf_{tag}_diminishing_returns.png'
    )
    # -- Save model --------------------------------------------------------
    model_path = MODELS / f'rf_{tag}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model':       best_model,
            'best_params': search.best_params_,
            'cv_score':    search.best_score_,
            'test_acc':    acc,
        }, f)
    print(f"  Saved model: {model_path}")

    return acc, search.best_params_


# -- Main ------------------------------------------------------------------

def main(feature_mode='handcrafted'):
    specs, labels, persons = load_data()
    results = {}

    if feature_mode in ('handcrafted', 'both'):
        print("\nExtracting hand-crafted features ...")
        t0 = time.time()
        X  = extract_all(specs)
        print(f"  Shape: {X.shape}  ({time.time()-t0:.1f}s)")

        X_tr, X_te, y_tr, y_te = subject_split(X, labels, persons)
        acc, params = run_rf(X_tr, X_te, y_tr, y_te, tag='handcrafted')
        results['handcrafted'] = acc

    if feature_mode in ('pca', 'both'):
        print("\nExtracting PCA features ...")
        rng    = np.random.default_rng(42)
        unique = rng.permutation(np.unique(persons))
        n_test = max(1, int(len(unique) * 0.2))
        test_s = set(unique[:n_test])
        tr_idx = np.array([i for i, p in enumerate(persons) if p not in test_s])
        te_idx = np.array([i for i, p in enumerate(persons) if p in test_s])

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

        acc, params = run_rf(
            X_pca_tr, X_pca_te,
            labels[tr_idx], labels[te_idx],
            tag='pca'
        )
        results['pca'] = acc

    print("\n" + "="*50)
    print("Summary:")
    for name, acc in results.items():
        print(f"  RF ({name}): {acc:.3f}")
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