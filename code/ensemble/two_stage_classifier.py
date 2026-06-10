# two_stage_classifier.py 

# Stage 1 : DCNN
# Stage 2 : Binary SVM (and KNN) - trained only on pick_up vs drink_water

import sys
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib    import Path
from scipy.ndimage import zoom
from sklearn.svm            import SVC
from sklearn.neighbors      import KNeighborsClassifier
from sklearn.preprocessing  import StandardScaler
from sklearn.pipeline       import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics        import (classification_report, confusion_matrix,
                                    ConfusionMatrixDisplay, accuracy_score)

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[2]
PREPROCESSED = ROOT / 'preprocessed'
MODELS       = ROOT / 'trained_models'
RESULTS_CM   = ROOT / 'results' / 'confusion_matrices'
RESULTS      = ROOT / 'results'

sys.path.insert(0, str(ROOT / 'code' / 'shared'))
from feature_extraction import extract_all
from utils import subject_split as _canonical_split

# ── Constants ──────────────────────────────────────────────────────────────
IMG_H = IMG_W = 128
CLASS_NAMES   = {1: 'walking', 2: 'sitting',  3: 'standing',
                 4: 'pick_up', 5: 'drink',    6: 'fall'}


# ── DCNN (Stage 1) ─────────────────────────────────────────────────────────

def _build_dcnn():
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return None, None

    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.MaxPool2d(2), nn.Dropout2d(0.25))
        def forward(self, x): return self.block(x)

    class DCNN(nn.Module):
        def __init__(self):
            super().__init__()
            layers, in_ch, out_ch, sz = [], 1, 64, 128
            for _ in range(5):
                layers.append(ConvBlock(in_ch, out_ch))
                in_ch = out_ch; out_ch = min(out_ch * 2, 512); sz //= 2
            self.features   = nn.Sequential(*layers)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_ch * sz * sz, 512),
                nn.ReLU(inplace=True), nn.Dropout(0.297),
                nn.Linear(512, 6))
        def forward(self, x): return self.classifier(self.features(x))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return DCNN(), device


def load_dcnn():
    pth = MODELS / 'dcnn_final_model.pth'
    if not pth.exists():
        raise FileNotFoundError(f"DCNN not found: {pth}")
    model, device = _build_dcnn()
    if model is None:
        raise ImportError("PyTorch not available")
    import torch
    model.load_state_dict(torch.load(pth, map_location=device))
    model.to(device).eval()
    print(f"  Loaded DCNN  device={device}")
    return model, device


def dcnn_probs(model, device, specs):
    import torch, torch.nn.functional as F
    imgs = []
    for spec in specs:
        p1, p99    = np.percentile(spec, [1, 99])
        spec_norm  = np.clip((spec - p1) / (p99 - p1 + 1e-12), 0.0, 1.0)
        img        = zoom(spec_norm, (IMG_H / spec_norm.shape[0],
                                      IMG_W / spec_norm.shape[1]), order=1)
        imgs.append(img.astype(np.float32))
    X = np.stack(imgs)[:, np.newaxis]            # (N, 1, 128, 128)

    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), 128):
            b = torch.tensor(X[i:i+128]).to(device)
            out.append(F.softmax(model(b), dim=1).cpu().numpy())
    return np.vstack(out).astype(np.float32)


# ── Features (Stage 2) ────────────────────────────────────────────────────

def get_all_features(specs):
    """
    Extract Stage 2 feature matrix for all N samples.
    Uses hand-crafted (20) + range features (31) if range_feats.npy exists.
    """
    X_hand = extract_all(specs)                   # (N, 20)

    range_path = PREPROCESSED / 'range_feats.npy'
    if range_path.exists():
        X_range = np.load(range_path).astype(np.float32)   # (N, 31)
        if X_range.shape[0] == len(specs):
            X = np.hstack([X_hand, X_range])
            print(f"  Stage 2 features: hand-crafted (20) + range (31) = {X.shape[1]}")
            return X
        print(f"  WARN: range_feats row count mismatch - hand-crafted only")

    print(f"  Stage 2 features: hand-crafted only (20)")
    return X_hand


# ── Stage 2 training ───────────────────────────────────────────────────────

def train_stage2(X_tr, y_tr, method='svm'):
    """
    Train a binary specialist on pick_up (4) vs drink_water (5) samples only.
    GridSearchCV with 5-fold stratified CV on the training subset.
    """
    mask   = np.isin(y_tr, [4, 5])
    X_s2   = X_tr[mask]
    y_s2   = y_tr[mask]
    n4, n5 = (y_s2 == 4).sum(), (y_s2 == 5).sum()
    print(f"  Stage 2 training samples:  pick_up={n4}  drink={n5}  total={len(y_s2)}")

    if method == 'svm':
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf',    SVC(probability=True, class_weight='balanced',
                           random_state=42)),
        ])
        grid = [
            {'clf__kernel': ['rbf'],
             'clf__C':      [0.1, 1, 10, 100],
             'clf__gamma':  ['scale', 'auto', 0.01]},
            {'clf__kernel': ['poly'],
             'clf__C':      [0.1, 1, 10],
             'clf__degree': [2, 3],
             'clf__gamma':  ['scale']},
        ]
    else:   # knn
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf',    KNeighborsClassifier(n_jobs=-1)),
        ])
        grid = {
            'clf__n_neighbors': [3, 5, 7, 9, 11],
            'clf__metric':      ['euclidean', 'manhattan', 'cosine'],
            'clf__weights':     ['uniform', 'distance'],
        }

    cv     = StratifiedKFold(5, shuffle=True, random_state=42)
    search = GridSearchCV(pipe, grid, cv=cv, scoring='accuracy', n_jobs=-1)
    search.fit(X_s2, y_s2)
    print(f"  Stage 2 ({method}) CV acc: {search.best_score_:.4f}  "
          f"params: {search.best_params_}")
    return search.best_estimator_


# ── Two-stage prediction ───────────────────────────────────────────────────

def predict_two_stage(stage1_probs, X_te, stage2_model, y_te=None):
    """
    Apply routing:
      Stage 1 predicts 4 or 5  →  Stage 2 decides
      Stage 1 predicts 1,2,3,6 →  keep Stage 1 prediction

    Also prints a routing breakdown when y_te is provided.

    Returns final predictions and routing mask.
    """
    stage1_preds = stage1_probs.argmax(axis=1) + 1      # 1-indexed
    route_mask   = np.isin(stage1_preds, [4, 5])
    final_preds  = stage1_preds.copy()

    n_routed = int(route_mask.sum())
    n_total = len(y_te) if y_te is not None else len(stage1_preds)
    print(f"\n  Samples routed to Stage 2: {n_routed}/{n_total} "
          f"({100*n_routed/len(stage1_preds):.1f}%)")

    if n_routed > 0:
        s2_preds             = stage2_model.predict(X_te[route_mask])
        final_preds[route_mask] = s2_preds

        # ── Routing breakdown ────────
        if y_te is not None:
            y_routed  = y_te[route_mask]
            s1_routed = stage1_preds[route_mask]

            s1_correct = (s1_routed == y_routed).sum()
            s2_correct = (s2_preds   == y_routed).sum()

            print(f"  On routed samples (n={n_routed}):")
            print(f"    Stage 1 would have got : {s1_correct}/{n_routed} "
                  f"({100*s1_correct/n_routed:.1f}%)")
            print(f"    Stage 2 got            : {s2_correct}/{n_routed} "
                  f"({100*s2_correct/n_routed:.1f}%)")
            print(f"    Delta                  : "
                  f"{'+' if s2_correct >= s1_correct else ''}"
                  f"{s2_correct - s1_correct} samples")

    return final_preds, route_mask


# ── Evaluation helpers ─────────────────────────────────────────────────────

def evaluate(y_true, y_pred, title, save_path):
    acc = accuracy_score(y_true, y_pred)
    print(f"\n  [{title}]  accuracy = {acc:.4f}")
    print(classification_report(
        y_true, y_pred,
        labels      = sorted(CLASS_NAMES),
        target_names= [CLASS_NAMES[c] for c in sorted(CLASS_NAMES)],
        zero_division=0,
    ))
    RESULTS_CM.mkdir(parents=True, exist_ok=True)
    cm   = confusion_matrix(y_true, y_pred, labels=sorted(CLASS_NAMES.keys()))
    lbls = [CLASS_NAMES[c] for c in sorted(CLASS_NAMES)]
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=lbls).plot(
        ax=ax, colorbar=True, cmap='Oranges')
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")
    return acc


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    RESULTS_CM.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ───────────────────────────────────────────────────
    print("Loading preprocessed data …")
    specs   = np.load(PREPROCESSED / 'spectrograms.npy', allow_pickle=True)
    labels  = np.load(PREPROCESSED / 'labels.npy').astype(int)
    persons = np.load(PREPROCESSED / 'persons.npy')

    train_mask, test_mask = _canonical_split(persons)
    print(f"  Canonical split:  train={train_mask.sum()}  test={test_mask.sum()}")

    y_tr = labels[train_mask]
    y_te = labels[test_mask]

    # ── 2. Load DCNN (Stage 1) ─────────────────────────────────────────
    print("\nLoading Stage 1 (DCNN) …")
    dcnn, device = load_dcnn()

    # ── 3. Extract Stage 2 features ────────────────────────────────────
    print("\nExtracting Stage 2 features …")
    X_all = get_all_features(specs)     # (N, 20 or 51)
    X_tr  = X_all[train_mask]
    X_te  = X_all[test_mask]

    # ── 4. Stage 1 baseline ────────────────────────────────────────────
    print("\nRunning Stage 1 on test set …")
    p1_te       = dcnn_probs(dcnn, device, specs[test_mask])
    stage1_preds = p1_te.argmax(axis=1) + 1

    acc_s1 = evaluate(
        y_te, stage1_preds,
        'Stage 1 only - DCNN',
        RESULTS_CM / 'stage1_dcnn_cm.png',
    )
    summary = {'Stage 1  (DCNN alone)': acc_s1}

    # ── 5. Train Stage 2 + evaluate ────────────────────────────────────
    print("\n" + "="*60)
    print("  STAGE 2 TRAINING AND EVALUATION")
    print("="*60)

    for method in ['svm', 'knn']:
        print(f"\n── Stage 2 : {method.upper()} specialist ──")
        s2_model = train_stage2(X_tr, y_tr, method=method)

        final_preds, _ = predict_two_stage(p1_te, X_te, s2_model, y_te)

        acc = evaluate(
            y_te, final_preds,
            f'Two-stage: DCNN + {method.upper()} specialist',
            RESULTS_CM / f'two_stage_{method}_cm.png',
        )
        summary[f'Two-stage (DCNN + {method.upper()})'] = acc

    # ── 6. Summary ─────────────────────────────────────────────────────
    best = max(summary.values())
    print("\n" + "="*55)
    print("  TWO-STAGE SUMMARY")
    print("="*55)
    for name, acc in summary.items():
        tag = "  ← best" if acc == best else ""
        print(f"  {name:<38}  {acc:.4f}{tag}")
    print("="*55)

    lines = ["="*50, "  TWO-STAGE CLASSIFIER SUMMARY", "="*50]
    for name, acc in summary.items():
        lines.append(f"  {name:<38}  {acc:.4f}")
    lines.append("="*50)
    out = RESULTS / 'two_stage_summary.txt'
    out.write_text('\n'.join(lines))
    print(f"\n  Saved: {out}")


if __name__ == '__main__':
    main()
