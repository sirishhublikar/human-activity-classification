# ensemble.py 
# Combines classical ML and DCNN predictions via soft & hard voting.

import sys
import pickle
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib    import Path
from scipy.ndimage import zoom
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                              classification_report, accuracy_score)

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[2]   # project root
PREPROCESSED = ROOT / 'preprocessed'
MODELS       = ROOT / 'trained_models'
RESULTS_CM   = ROOT / 'results' / 'confusion_matrices'
RESULTS      = ROOT / 'results'

sys.path.insert(0, str(ROOT / 'code' / 'shared'))
from feature_extraction import extract_all
from utils import subject_split as _canonical_split

# ── Constants ──────────────────────────────────────────────────────────────
SEED       = 42
TEST_FRAC  = 0.20
IMG_H      = 128
IMG_W      = 128
NUM_CLASSES = 6

CLASS_NAMES = {1: 'walking', 2: 'sitting',  3: 'standing',
               4: 'pick_up', 5: 'drink',    6: 'fall'}

# Accuracy each model achieved on its own test run - used for weighted vote.
KNOWN_ACCURACIES = {
    'svm':  0.853,
    'knn':  0.879,
    'rf':   None, 
    'dcnn': 0.9139,  
}



# ── DCNN architecture ─────────────────────

def _build_dcnn():
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return None, None

    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch, dropout=0.25):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(dropout),
            )
        def forward(self, x):
            return self.block(x)

    class DCNN(nn.Module):
        def __init__(self, num_classes=6, dropout=0.297,
                     n_conv_blocks=5, base_filters=64, fc_size=512):
            super().__init__()
            layers, in_ch, out_ch, img_size = [], 1, base_filters, 128
            for _ in range(n_conv_blocks):
                layers.append(ConvBlock(in_ch, out_ch, dropout=0.25))
                in_ch    = out_ch
                out_ch   = min(out_ch * 2, 512)
                img_size = img_size // 2
            self.features   = nn.Sequential(*layers)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_ch * img_size * img_size, fc_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(fc_size, num_classes),
            )
        def forward(self, x):
            return self.classifier(self.features(x))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = DCNN()
    return model, device


# ── Data loading ───────────────────────────────────────────────────────────

def load_preprocessed():
    specs   = np.load(PREPROCESSED / 'spectrograms.npy', allow_pickle=True)
    labels  = np.load(PREPROCESSED / 'labels.npy')       # 1-6, shape (N,)
    persons = np.load(PREPROCESSED / 'persons.npy')
    return specs, labels, persons


def canonical_split(persons):
    train_mask, test_mask = _canonical_split(persons)
    print(f"  Canonical split:  train={int(train_mask.sum())}  "
          f"test={int(test_mask.sum())}")
    return test_mask


# ── Model loading ──────────────────────────────────────────────────────────

def load_classical_model(name):
    path = MODELS / f'{name}_handcrafted.pkl'
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    bundle = pickle.load(open(path, 'rb'))
    model  = bundle['model']
    acc    = bundle.get('test_acc', KNOWN_ACCURACIES.get(name))
    print(f"  Loaded {name:<4}  test_acc={acc:.4f}")
    return model, acc


def load_dcnn_model():
    pth = MODELS / 'dcnn_final_model.pth'
    if not pth.exists():
        print("  WARN: dcnn_final_model.pth not found - DCNN excluded from ensemble.")
        return None, None, None

    model, device = _build_dcnn()
    if model is None:
        print("  WARN: PyTorch not installed - DCNN excluded from ensemble.")
        return None, None, None

    import torch
    model.load_state_dict(torch.load(pth, map_location=device))
    model.to(device).eval()
    acc = KNOWN_ACCURACIES['dcnn']
    print(f"  Loaded dcnn  test_acc={acc:.4f}  device={device}")
    return model, device, acc


# ── Feature / image preparation ────────────────────────────────────────────

def get_classical_features(specs):
    return extract_all(specs)


def specs_to_dcnn_images(specs):
    imgs = []
    for spec in specs:
        p1, p99    = np.percentile(spec, [1, 99])
        spec_norm  = np.clip((spec - p1) / (p99 - p1 + 1e-12), 0.0, 1.0)
        img        = zoom(spec_norm,
                          (IMG_H / spec_norm.shape[0],
                           IMG_W / spec_norm.shape[1]),
                          order=1)
        imgs.append(img.astype(np.float32))
    return np.stack(imgs)[:, np.newaxis, :, :]   # (N, 1, 128, 128)


# ── Probability extraction ─────────────────────────────────────────────────

def classical_probs(model, X):
    return model.predict_proba(X).astype(np.float32)


def dcnn_probs(model, device, X_images):
    import torch
    import torch.nn.functional as F

    BATCH = 128
    all_probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X_images), BATCH):
            batch  = torch.tensor(X_images[i:i + BATCH]).to(device)
            logits = model(batch)
            probs  = F.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
    return np.vstack(all_probs).astype(np.float32)   # (N, 6)


# ── Ensemble strategies ────────────────────────────────────────────────────

def soft_vote_uniform(prob_stack):
    mean_probs = prob_stack.mean(axis=0)              # (N, 6)
    return mean_probs.argmax(axis=1) + 1      


def soft_vote_weighted(prob_stack, weights):
    w          = np.array(weights, dtype=np.float32)
    w          = w / w.sum()                          # normalise
    weighted   = (prob_stack * w[:, None, None]).sum(axis=0)  # (N, 6)
    return weighted.argmax(axis=1) + 1


def hard_vote(prob_stack):
    votes     = prob_stack.argmax(axis=2) + 1         # (n_models, N)
    n_models  = prob_stack.shape[0]
    preds     = []
    for i in range(votes.shape[1]):
        counts = np.bincount(votes[:, i], minlength=7)[1:]   # classes 1-6
        if counts.max() > n_models // 2:
            preds.append(int(counts.argmax()) + 1)
        else:
            preds.append(int(prob_stack[:, i, :].sum(axis=0).argmax()) + 1)
    return np.array(preds)


# ── Evaluation & plotting ─────────────────────────────────────────────────

def evaluate(y_true, y_pred, label):
    acc = accuracy_score(y_true, y_pred)
    print(f"\n  [{label}]  accuracy = {acc:.4f}")
    print(classification_report(
        y_true, y_pred,
        labels=sorted(CLASS_NAMES), zero_division=0,
        target_names=[CLASS_NAMES[c] for c in sorted(CLASS_NAMES)]
    ))
    return acc


def plot_cm(y_true, y_pred, title, path):
    RESULTS_CM.mkdir(parents=True, exist_ok=True)
    cm   = confusion_matrix(y_true, y_pred, labels=sorted(CLASS_NAMES))
    lbls = [CLASS_NAMES[c] for c in sorted(CLASS_NAMES)]
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=lbls).plot(
        ax=ax, colorbar=True, cmap='Purples')
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    RESULTS_CM.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ───────────────────────────────────────────────────
    print("Loading preprocessed data …")
    specs, labels, persons = load_preprocessed()

    test_mask  = canonical_split(persons)
    specs_te   = specs[test_mask]
    y_te       = labels[test_mask].astype(int)   # 1-6

    # ── 2. Load models ─────────────────────────────────────────────────
    print("\nLoading models …")
    svm_model, svm_acc = load_classical_model('svm')
    knn_model, knn_acc = load_classical_model('knn')
    rf_model,  rf_acc  = load_classical_model('rf')
    dcnn_model, device, dcnn_acc = load_dcnn_model()

    # ── 3. Get probability predictions ────────────────────────────────
    print("\nExtracting features and running inference …")

    X_hand = get_classical_features(specs_te)
    print(f"  Hand-crafted features: {X_hand.shape}")

    p_svm  = classical_probs(svm_model, X_hand)
    print(f"  SVM   probs: {p_svm.shape}")

    p_knn  = classical_probs(knn_model, X_hand)
    print(f"  KNN   probs: {p_knn.shape}")

    p_rf   = classical_probs(rf_model, X_hand)
    print(f"  RF    probs: {p_rf.shape}")

    if dcnn_model is not None:
        X_imgs  = specs_to_dcnn_images(specs_te)
        p_dcnn  = dcnn_probs(dcnn_model, device, X_imgs)
        print(f"  DCNN  probs: {p_dcnn.shape}")
        prob_stack  = np.stack([p_svm, p_knn, p_rf, p_dcnn])  # (4, N, 6)
        model_names = ['SVM', 'KNN', 'RF', 'DCNN']
        weights     = [svm_acc, knn_acc, rf_acc, dcnn_acc]
    else:
        prob_stack  = np.stack([p_svm, p_knn, p_rf])           # (3, N, 6)
        model_names = ['SVM', 'KNN', 'RF']
        weights     = [svm_acc, knn_acc, rf_acc]

    print(f"\n  Ensemble over {len(model_names)} models: {model_names}")
    print(f"  Weights (accuracy): {[f'{w:.4f}' for w in weights]}")

    # ── 4. Individual model baselines ──────────────────────────────────
    print("\n" + "=" * 60)
    print("  INDIVIDUAL MODEL BASELINES")
    print("=" * 60)
    ind_accs = {}
    for i, name in enumerate(model_names):
        y_pred = prob_stack[i].argmax(axis=1) + 1
        ind_accs[name] = evaluate(y_te, y_pred, f'{name} individual')

    # ── 5. Ensemble strategies ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ENSEMBLE RESULTS")
    print("=" * 60)

    y_uniform  = soft_vote_uniform(prob_stack)
    y_weighted = soft_vote_weighted(prob_stack, weights)
    y_hard     = hard_vote(prob_stack)

    acc_uniform  = evaluate(y_te, y_uniform,  'Soft vote - uniform')
    acc_weighted = evaluate(y_te, y_weighted, 'Soft vote - weighted')
    acc_hard     = evaluate(y_te, y_hard,     'Hard majority vote')

    # ── 6. Confusion matrices ──────────────────────────────────────────
    plot_cm(y_te, y_uniform,
            f'Ensemble - uniform soft vote  ({acc_uniform:.1%})',
            RESULTS_CM / 'ensemble_uniform_cm.png')
    plot_cm(y_te, y_weighted,
            f'Ensemble - weighted soft vote  ({acc_weighted:.1%})',
            RESULTS_CM / 'ensemble_weighted_cm.png')
    plot_cm(y_te, y_hard,
            f'Ensemble - hard majority vote  ({acc_hard:.1%})',
            RESULTS_CM / 'ensemble_hard_cm.png')

    # ── 7. Summary table ───────────────────────────────────────────────
    summary_lines = [
        "=" * 50,
        "  ENSEMBLE SUMMARY",
        "=" * 50,
        "  Individual models:",
    ]
    for name, acc in ind_accs.items():
        summary_lines.append(f"    {name:<6}  {acc:.4f}")
    summary_lines += [
        "",
        "  Ensemble strategies:",
        f"    Uniform  soft vote   {acc_uniform:.4f}",
        f"    Weighted soft vote   {acc_weighted:.4f}",
        f"    Hard majority vote   {acc_hard:.4f}",
        "=" * 50,
    ]
    summary = "\n".join(summary_lines)
    print("\n" + summary)

    (RESULTS / 'ensemble_summary.txt').write_text(summary)
    print(f"\n  Summary saved: {RESULTS / 'ensemble_summary.txt'}")


if __name__ == '__main__':
    main()
