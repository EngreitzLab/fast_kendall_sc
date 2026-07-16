import numpy as np
from ._fast_kendall_sc import _kendall_concordance_diff, _batch_kendall_tau


def _new_group_flags(x_sorted):
    """Boolean mask, True where a sorted (decreasing) x value starts a new tie run."""
    flags = np.empty(x_sorted.shape[0], dtype=bool)
    flags[0] = True
    flags[1:] = x_sorted[1:] != x_sorted[:-1]
    return flags


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
        return np.nan

    sorted_idx = np.argsort(x)[::-1].astype(np.uintp)
    new_group = _new_group_flags(x[sorted_idx])
    y_matrix = y.reshape(-1, 1)
    target_columns = np.array([0], dtype=np.uintp)

    raw_diff = _kendall_concordance_diff(y_matrix, sorted_idx, new_group, target_columns)[0]

    n_0 = (N * (N - 1)) / 2.0
    _, tie_counts_x = np.unique(x, return_counts=True)
    n_x = np.sum((tie_counts_x * (tie_counts_x - 1)) / 2.0)

    num_ones = int(np.sum(y))
    num_zeros = N - num_ones
    n_y = (num_ones * (num_ones - 1) / 2.0) + (num_zeros * (num_zeros - 1) / 2.0)

    denom = np.sqrt((n_0 - n_x) * (n_0 - n_y))
    return raw_diff / denom if denom > 0 else np.nan


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
        return np.full((n_features_X, n_features_Y), np.nan)

    n_0 = (n_samples * (n_samples - 1)) / 2.0
    num_ones = np.sum(Y, axis=0).astype(np.int64)
    num_zeros = n_samples - num_ones
    n_y = (num_ones * (num_ones - 1) / 2.0) + (num_zeros * (num_zeros - 1) / 2.0)

    correlations = np.full((n_features_X, n_features_Y), np.nan)
    target_peaks = np.arange(n_features_Y, dtype=np.uintp)

    for g in range(n_features_X):
        x_col = X[:, g]
        _, tie_counts_x = np.unique(x_col, return_counts=True)
        n_x = np.sum((tie_counts_x * (tie_counts_x - 1)) / 2.0)

        sorted_idx = np.argsort(x_col)[::-1].astype(np.uintp)
        new_group = _new_group_flags(x_col[sorted_idx])
        raw_diffs = np.array(_kendall_concordance_diff(Y, sorted_idx, new_group, target_peaks))

        denom = np.sqrt((n_0 - n_x) * (n_0 - n_y))
        valid_denom = denom > 0
        correlations[g, valid_denom] = raw_diffs[valid_denom] / denom[valid_denom]

    return correlations


def batch_kendall_tau(rna_matrix, atac_matrix, gene_indices, peak_indices):
    """Compute Kendall's tau-b for an explicit list of (gene, peak) candidate pairs.

    Unlike `pairwise_kendall_tau`, which scores every gene against every peak,
    this takes a possibly ragged set of candidate pairs (e.g. the specific
    enhancer-gene pairs from an ABC/E2G candidate list) and computes only
    those. The per-gene sorting and correlation work is done in Rust and
    parallelized across genes, so this is the entry point to use for large
    batches rather than looping over `kendall_tau_score` in Python.

    rna_matrix : (n_cells, n_genes) dense array of expression values.
    atac_matrix: (n_cells, n_peaks) dense 0/1 array of accessibility.
    gene_indices, peak_indices : 1D arrays of equal length; pair k is
        (gene_indices[k], peak_indices[k]).

    Returns a 1D array of tau values aligned with gene_indices/peak_indices.
    """
    rna_matrix = np.asarray(rna_matrix, dtype=np.float64)
    atac_matrix = np.asarray(atac_matrix, dtype=np.uint8)
    gene_indices = np.asarray(gene_indices)
    peak_indices = np.asarray(peak_indices)

    if gene_indices.shape != peak_indices.shape:
        raise ValueError("gene_indices and peak_indices must have the same shape.")
    if rna_matrix.shape[0] != atac_matrix.shape[0]:
        raise ValueError(
            f"rna_matrix and atac_matrix must have the same number of cells (rows): "
            f"{rna_matrix.shape[0]} vs {atac_matrix.shape[0]}."
        )

    n_pairs = gene_indices.shape[0]
    if n_pairs == 0:
        return np.array([], dtype=np.float64)

    n_cells = rna_matrix.shape[0]
    if n_cells < 2:
        return np.full(n_pairs, np.nan)

    # Group candidate pairs by gene (CSR-style, like scipy.sparse): sort the
    # pairs by gene id, then record where each gene's slice of peaks starts
    # and ends. `order` is kept so results can be scattered back into the
    # caller's original pair order at the end.
    order = np.argsort(gene_indices, kind="stable")
    gene_sorted = gene_indices[order].astype(np.uintp)
    peak_sorted = peak_indices[order].astype(np.uintp)

    unique_genes, counts = np.unique(gene_sorted, return_counts=True)
    offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.uintp)

    # The ATAC tie term depends only on the peak, not the gene, so it is
    # computed once for the whole matrix instead of once per gene.
    num_ones = atac_matrix.sum(axis=0).astype(np.int64)
    num_zeros = n_cells - num_ones
    n_y = (num_ones * (num_ones - 1) / 2.0) + (num_zeros * (num_zeros - 1) / 2.0)

    flat_results = np.asarray(
        _batch_kendall_tau(
            rna_matrix,
            atac_matrix,
            unique_genes.astype(np.uintp),
            offsets,
            peak_sorted,
            n_y,
        )
    )

    results = np.empty(n_pairs, dtype=np.float64)
    results[order] = flat_results
    return results
