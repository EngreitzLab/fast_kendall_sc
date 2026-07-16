import numpy as np
import pytest
from scipy.stats import kendalltau

from fast_kendall_sc import kendall_tau_score, pairwise_kendall_tau


def _scipy_tau_b(x, y):
    return kendalltau(x, y, variant="b").statistic


@pytest.mark.parametrize("seed", range(10))
def test_matches_scipy_no_ties_in_x(seed):
    rng = np.random.default_rng(seed)
    n = 200
    x = rng.random(n)  # continuous, effectively no ties
    y = (rng.random(n) < 0.3).astype(np.uint8)

    assert kendall_tau_score(x, y) == pytest.approx(_scipy_tau_b(x, y), abs=1e-9)


@pytest.mark.parametrize("seed", range(10))
def test_matches_scipy_with_ties_in_x(seed):
    rng = np.random.default_rng(seed)
    n = 200
    x = rng.poisson(1.5, size=n).astype(float)  # small-integer counts -> many ties
    y = (rng.random(n) < 0.3).astype(np.uint8)

    assert kendall_tau_score(x, y) == pytest.approx(_scipy_tau_b(x, y), abs=1e-9)


@pytest.mark.parametrize("seed", range(10))
def test_matches_scipy_under_scrna_sparsity(seed):
    # ~70% dropout zeros, typical scRNA-seq expression sparsity.
    rng = np.random.default_rng(seed)
    n = 500
    x = np.where(rng.random(n) < 0.7, 0, rng.poisson(3, size=n)).astype(float)
    y = (rng.random(n) < 0.25).astype(np.uint8)

    assert kendall_tau_score(x, y) == pytest.approx(_scipy_tau_b(x, y), abs=1e-9)


def test_pairwise_matches_single_gene_calls():
    rng = np.random.default_rng(42)
    n, n_genes, n_peaks = 150, 5, 8
    X = np.where(rng.random((n, n_genes)) < 0.6, 0, rng.poisson(2, size=(n, n_genes)))
    Y = (rng.random((n, n_peaks)) < 0.3).astype(np.uint8)

    pairwise = pairwise_kendall_tau(X, Y)

    for g in range(n_genes):
        for p in range(n_peaks):
            expected = kendall_tau_score(X[:, g], Y[:, p])
            assert pairwise[g, p] == pytest.approx(expected, abs=1e-9)


def test_undefined_correlation_returns_nan():
    n = 50
    x_constant = np.zeros(n)
    y = np.zeros(n, dtype=np.uint8)
    y[: n // 2] = 1

    assert np.isnan(kendall_tau_score(x_constant, y))

    y_constant = np.ones(n, dtype=np.uint8)
    x = np.arange(n, dtype=float)
    assert np.isnan(kendall_tau_score(x, y_constant))


def test_fewer_than_two_samples_returns_nan():
    assert np.isnan(kendall_tau_score(np.array([1.0]), np.array([1], dtype=np.uint8)))
    assert np.isnan(kendall_tau_score(np.array([]), np.array([], dtype=np.uint8)))


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError):
        kendall_tau_score(np.array([1.0, 2.0, 3.0]), np.array([1, 0], dtype=np.uint8))
