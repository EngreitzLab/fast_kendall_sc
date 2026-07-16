import numpy as np
import pytest

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
    gene_indices = np.array(gene_indices)[order]
    peak_indices = np.array(peak_indices)[order]
    return gene_indices, peak_indices


@pytest.mark.parametrize("seed", range(5))
def test_batch_matches_single_gene_calls(seed):
    rng = np.random.default_rng(seed)
    n_cells, n_genes, n_peaks = 300, 12, 20

    rna_matrix = np.where(
        rng.random((n_cells, n_genes)) < 0.6, 0, rng.poisson(2, size=(n_cells, n_genes))
    ).astype(float)
    atac_matrix = (rng.random((n_cells, n_peaks)) < 0.3).astype(np.uint8)

    gene_indices, peak_indices = _make_ragged_pairs(rng, n_genes, n_peaks, max_peaks_per_gene=6)

    batch_results = batch_kendall_tau(rna_matrix, atac_matrix, gene_indices, peak_indices)

    for k in range(len(gene_indices)):
        expected = kendall_tau_score(rna_matrix[:, gene_indices[k]], atac_matrix[:, peak_indices[k]])
        assert batch_results[k] == pytest.approx(expected, abs=1e-9, nan_ok=True)


def test_batch_handles_duplicate_pairs_and_repeated_genes():
    rng = np.random.default_rng(7)
    n_cells, n_genes, n_peaks = 150, 4, 6
    rna_matrix = np.where(
        rng.random((n_cells, n_genes)) < 0.5, 0, rng.poisson(1, size=(n_cells, n_genes))
    ).astype(float)
    atac_matrix = (rng.random((n_cells, n_peaks)) < 0.4).astype(np.uint8)

    # same (gene, peak) pair requested twice, plus a gene queried against
    # multiple peaks and a peak queried against multiple genes
    gene_indices = np.array([0, 0, 1, 1, 2, 3, 0])
    peak_indices = np.array([0, 0, 0, 1, 5, 5, 3])

    results = batch_kendall_tau(rna_matrix, atac_matrix, gene_indices, peak_indices)

    assert results[0] == pytest.approx(results[1], abs=1e-9)
    for k in range(len(gene_indices)):
        expected = kendall_tau_score(rna_matrix[:, gene_indices[k]], atac_matrix[:, peak_indices[k]])
        assert results[k] == pytest.approx(expected, abs=1e-9)


def test_batch_empty_pairs_returns_empty_array():
    rna_matrix = np.zeros((10, 3))
    atac_matrix = np.zeros((10, 4), dtype=np.uint8)
    results = batch_kendall_tau(rna_matrix, atac_matrix, np.array([]), np.array([]))
    assert results.shape == (0,)


def test_batch_mismatched_pair_lengths_raise():
    rna_matrix = np.zeros((10, 3))
    atac_matrix = np.zeros((10, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        batch_kendall_tau(rna_matrix, atac_matrix, np.array([0, 1]), np.array([0]))


def test_batch_mismatched_cell_counts_raise():
    rna_matrix = np.zeros((10, 3))
    atac_matrix = np.zeros((11, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        batch_kendall_tau(rna_matrix, atac_matrix, np.array([0]), np.array([0]))
