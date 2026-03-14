import numpy as np


def compute_global_mask(X_train, max_df_ratio=0.7):
    """Return indices of features with document frequency <= max_df_ratio."""
    doc_freqs = np.array((X_train > 0).sum(axis=0)).flatten()
    max_count = max_df_ratio * X_train.shape[0]
    return np.where(doc_freqs <= max_count)[0]


def compute_class_mask(X_train, y_train, max_df_ratio=0.7):
    """
    Return indices of features that satisfy class‑specific max_df:
        df_pos <= max_df_ratio * n_pos  OR  df_neg <= max_df_ratio * n_neg
    """
    mask_pos = (y_train == 1)
    mask_neg = (y_train == 0)
    df_pos = np.array((X_train[mask_pos] > 0).sum(axis=0)).flatten()
    df_neg = np.array((X_train[mask_neg] > 0).sum(axis=0)).flatten()
    max_pos = max_df_ratio * mask_pos.sum()
    max_neg = max_df_ratio * mask_neg.sum()
    return np.where((df_pos <= max_pos) | (df_neg <= max_neg))[0]