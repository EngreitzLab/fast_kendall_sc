import numpy as np
import pytest
import scipy.sparse as sp

from fast_kendall_sc import kendall_tau_score, batch_kendall_tau


def _make_ragged_pairs(rng, n_genes, n_peaks, max_peaks_per_gene):
    gene_indices = []
    peak_indices = []
    for g in range(n_genes):
        k = rng.integers(1, max_peaks_per_gene + 1)
        peaks = rng.choice(n_peaks, size=k, replace=False)
        gene_indices.extend([g] * k)
        peak_indices.extend(peaks.tolist())
    order = rng.permutation(len(gene_indices))
    return np.array(gene_indices)[order], np.array(peak_indices)[order]


@pytest.mark.parametrize("seed", range(5))
def test_sparse_matches_dense(seed):
    rng = np.random.default_rng(seed)
    n_cells, n_genes, n_peaks = 300, 12, 20

    rna_dense = np.where(
        rng.random((n_cells, n_genes)) < 0.8, 0, rng.poisson(2, size=(n_cells, n_genes))
    ).astype(float)
    atac_dense = (rng.random((n_cells, n_peaks)) < 0.15).astype(np.uint8)

    gene_indices, peak_indices = _make_ragged_pairs(rng, n_genes, n_peaks, max_peaks_per_gene=6)

    dense_results = batch_kendall_tau(rna_dense, atac_dense, gene_indices, peak_indices)

    rna_sparse = sp.csr_matrix(rna_dense)
    atac_sparse = sp.csr_matrix(atac_dense)
    sparse_results = batch_kendall_tau(rna_sparse, atac_sparse, gene_indices, peak_indices)

    np.testing.assert_allclose(sparse_results, dense_results, atol=1e-9, equal_nan=True)


@pytest.mark.parametrize("seed", range(5))
def test_sparse_matches_single_gene_calls(seed):
    rng = np.random.default_rng(seed)
    n_cells, n_genes, n_peaks = 400, 10, 15

    rna_dense = np.where(
        rng.random((n_cells, n_genes)) < 0.75, 0, rng.poisson(3, size=(n_cells, n_genes))
    ).astype(float)
    atac_dense = (rng.random((n_cells, n_peaks)) < 0.2).astype(np.uint8)

    gene_indices, peak_indices = _make_ragged_pairs(rng, n_genes, n_peaks, max_peaks_per_gene=5)

    rna_sparse = sp.csc_matrix(rna_dense)
    atac_sparse = sp.csc_matrix(atac_dense)
    results = batch_kendall_tau(rna_sparse, atac_sparse, gene_indices, peak_indices)

    for k in range(len(gene_indices)):
        expected = kendall_tau_score(rna_dense[:, gene_indices[k]], atac_dense[:, peak_indices[k]])
        assert results[k] == pytest.approx(expected, abs=1e-9, nan_ok=True)


def test_sparse_input_formats_are_interchangeable():
    rng = np.random.default_rng(3)
    n_cells, n_genes, n_peaks = 200, 6, 8
    rna_dense = np.where(
        rng.random((n_cells, n_genes)) < 0.7, 0, rng.poisson(2, size=(n_cells, n_genes))
    ).astype(float)
    atac_dense = (rng.random((n_cells, n_peaks)) < 0.25).astype(np.uint8)
    gene_indices, peak_indices = _make_ragged_pairs(rng, n_genes, n_peaks, max_peaks_per_gene=4)

    csr_result = batch_kendall_tau(sp.csr_matrix(rna_dense), sp.csr_matrix(atac_dense), gene_indices, peak_indices)
    csc_result = batch_kendall_tau(sp.csc_matrix(rna_dense), sp.csc_matrix(atac_dense), gene_indices, peak_indices)
    coo_result = batch_kendall_tau(sp.coo_matrix(rna_dense), sp.coo_matrix(atac_dense), gene_indices, peak_indices)

    np.testing.assert_allclose(csr_result, csc_result, atol=1e-9, equal_nan=True)
    np.testing.assert_allclose(csr_result, coo_result, atol=1e-9, equal_nan=True)


def test_sparse_explicit_zero_entries_do_not_change_result():
    # A sparse matrix that explicitly stores some zero entries (not pruned)
    # must give the same answer as one where those zeros are implicit --
    # eliminate_zeros() in the wrapper is what guarantees this.
    rng = np.random.default_rng(5)
    n_cells, n_genes, n_peaks = 150, 3, 4
    rna_dense = np.where(
        rng.random((n_cells, n_genes)) < 0.6, 0, rng.poisson(2, size=(n_cells, n_genes))
    ).astype(float)
    atac_dense = (rng.random((n_cells, n_peaks)) < 0.3).astype(np.uint8)
    gene_indices, peak_indices = _make_ragged_pairs(rng, n_genes, n_peaks, max_peaks_per_gene=3)

    # Force some explicit zero entries into the sparse structure (still
    # unpruned -- these positions are "present" with value 0.0, rather than
    # simply absent). This changes some cells' true expression to 0, so the
    # correct reference is the dense array reflecting that same change, not
    # the original pre-zeroing matrix.
    rna_dirty = sp.csc_matrix(rna_dense)
    rna_dirty.data[: max(1, rna_dirty.data.size // 5)] = 0.0
    reference_dense = rna_dirty.toarray()

    dirty_results = batch_kendall_tau(rna_dirty, sp.csc_matrix(atac_dense), gene_indices, peak_indices)
    reference_results = batch_kendall_tau(reference_dense, atac_dense, gene_indices, peak_indices)

    np.testing.assert_allclose(dirty_results, reference_results, atol=1e-9, equal_nan=True)


def test_sparse_negative_rna_values_raise():
    rna = sp.csc_matrix(np.array([[1.0, -2.0], [3.0, 4.0], [0.0, 5.0]]))
    atac = sp.csc_matrix(np.array([[1, 0], [0, 1], [1, 1]], dtype=np.uint8))
    with pytest.raises(ValueError):
        batch_kendall_tau(rna, atac, np.array([0]), np.array([0]))


def test_sparse_only_one_side_still_uses_sparse_path():
    rng = np.random.default_rng(9)
    n_cells, n_genes, n_peaks = 120, 4, 5
    rna_dense = np.where(
        rng.random((n_cells, n_genes)) < 0.5, 0, rng.poisson(1, size=(n_cells, n_genes))
    ).astype(float)
    atac_dense = (rng.random((n_cells, n_peaks)) < 0.3).astype(np.uint8)
    gene_indices, peak_indices = _make_ragged_pairs(rng, n_genes, n_peaks, max_peaks_per_gene=3)

    # RNA dense, ATAC sparse
    mixed_results = batch_kendall_tau(rna_dense, sp.csc_matrix(atac_dense), gene_indices, peak_indices)
    dense_results = batch_kendall_tau(rna_dense, atac_dense, gene_indices, peak_indices)

    np.testing.assert_allclose(mixed_results, dense_results, atol=1e-9, equal_nan=True)
