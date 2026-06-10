# utils.py  
# identical train / test subject sets for all models

import numpy as np

def subject_split(persons, test_frac=0.20, seed=42):
    """
    Parameters
    ----------
    persons   : array-like  int  subject ID per sample
    test_frac : float            fraction of subjects to hold out (default 0.20)
    seed      : int              random seed (default 42)

    Returns
    -------
    train_mask : ndarray bool (N,)
    test_mask  : ndarray bool (N,)
    """
    persons    = np.asarray(persons, dtype=int)
    rng        = np.random.default_rng(seed)
    unique     = np.unique(persons)
    shuffled   = rng.permutation(unique)
    n_test     = max(1, int(len(shuffled) * test_frac))
    test_subs  = set(shuffled[:n_test].tolist())

    test_mask  = np.array([int(p) in test_subs for p in persons])
    train_mask = ~test_mask

    return train_mask, test_mask
