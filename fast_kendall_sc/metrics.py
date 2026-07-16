import numpy as np
from ._fast_kendall_sc import _kendall_concordance_diff

def kendall_tau_score(x, y):
    """Calculate Kendall's tau-b correlation between two 1D vectors."""
    x = np.asarray(x)
    y = np.asarray(y, dtype=np.uint8)
    
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs x and y must be 1-dimensional arrays.")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Found input variables with inconsistent numbers of samples: {x.shape} and {y.shape}")
        
    N = x.shape[0]
    if N < 2:
        return 0.0

    sorted_idx = np.argsort(x)[::-1].astype(np.uintp)
    y_matrix = y.reshape(-1, 1)
    target_columns = np.array([0], dtype=np.uintp)
    
    raw_diff = _kendall_concordance_diff(y_matrix, sorted_idx, target_columns)[0]
    
    n_0 = (N * (N - 1)) / 2.0
    _, tie_counts_x = np.unique(x, return_counts=True)
    n_x = np.sum((tie_counts_x * (tie_counts_x - 1)) / 2.0)
    
    num_ones = np.sum(y)
    num_zeros = N - num_ones
    n_y = (num_ones * (num_ones - 1) / 2.0) + (num_zeros * (num_zeros - 1) / 2.0)
    
    denom = np.sqrt((n_0 - n_x) * (n_0 - n_y))
    return raw_diff / denom if denom > 0 else 0.0


def pairwise_kendall_tau(X, Y):
    """Compute pairwise Kendall's tau-b correlation between columns of X and Y."""
    X = np.asarray(X)
    Y = np.asarray(Y, dtype=np.uint8)
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
        
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"Inconsistent sample counts between X ({X.shape[0]}) and Y ({Y.shape[0]}).")
        
    n_samples, n_features_X = X.shape
    _, n_features_Y = Y.shape
    
    if n_samples < 2:
        return np.zeros((n_features_X, n_features_Y))
        
    n_0 = (n_samples * (n_samples - 1)) / 2.0
    num_ones = np.sum(Y, axis=0)
    num_zeros = n_samples - num_ones
    n_y = (num_ones * (num_ones - 1) / 2.0) + (num_zeros * (num_zeros - 1) / 2.0)
    
    correlations = np.zeros((n_features_X, n_features_Y))
    target_peaks = np.arange(n_features_Y, dtype=np.uintp)
    
    for g in range(n_features_X):
        x_col = X[:, g]
        _, tie_counts_x = np.unique(x_col, return_counts=True)
        n_x = np.sum((tie_counts_x * (tie_counts_x - 1)) / 2.0)
        
        sorted_idx = np.argsort(x_col)[::-1].astype(np.uintp)
        raw_diffs = np.array(_kendall_concordance_diff(Y, sorted_idx, target_peaks))
        
        denom = np.sqrt((n_0 - n_x) * (n_0 - n_y))
        valid_denom = denom > 0
        correlations[g, valid_denom] = raw_diffs[valid_denom] / denom[valid_denom]
        
    return correlations