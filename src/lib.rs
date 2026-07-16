use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use rayon::prelude::*;

// Computes concordant-discordant diffs for y_matrix columns against x sorted
// in decreasing order (via sorted_idx), with a correction for ties in x.
//
// `new_group[i]` must be true iff sorted position `i` starts a new run of
// equal x values (i.e. x_sorted[i] != x_sorted[i-1], and true at i == 0).
// Pairs within the same x-tie group are neither concordant nor discordant,
// but the global pass counts them arbitrarily based on tie-breaking order in
// sorted_idx. We correct for that by also accumulating a "local" diff that
// resets at each tie-group boundary, and subtracting it from the global diff
// (mirrors the two-pass approach in the original R implementation, fused
// into a single O(N) pass per column).
#[pyfunction]
fn _kendall_concordance_diff(
    y_matrix: PyReadonlyArray2<'_, u8>,
    sorted_idx: PyReadonlyArray1<'_, usize>,
    new_group: PyReadonlyArray1<'_, bool>,
    peak_columns: PyReadonlyArray1<'_, usize>,
) -> PyResult<Vec<f64>> {
    let y = y_matrix.as_array();
    let idx = sorted_idx.as_array();
    let groups = new_group.as_array();
    let peaks = peak_columns.as_array();

    let mut result = Vec::with_capacity(peaks.len());

    for &j in peaks.iter() {
        let mut concordant: u64 = 0;
        let mut discordant: u64 = 0;
        let mut cumsum: u64 = 0;

        let mut tie_concordant: u64 = 0;
        let mut tie_discordant: u64 = 0;
        let mut tie_cumsum: u64 = 0;
        let mut tie_rank: u64 = 0;

        for (i, &original_row) in idx.iter().enumerate() {
            let acc = y[[original_row, j]] as u64;
            cumsum += acc;

            let i_u64 = i as u64;
            discordant += acc * (i_u64 + 1 - cumsum);
            concordant += (1 - acc) * cumsum;

            if groups[i] {
                tie_cumsum = 0;
                tie_rank = 0;
            }
            tie_cumsum += acc;
            tie_rank += 1;
            tie_discordant += acc * (tie_rank - tie_cumsum);
            tie_concordant += (1 - acc) * tie_cumsum;
        }

        let global_diff = (concordant as f64) - (discordant as f64);
        let tie_diff = (tie_concordant as f64) - (tie_discordant as f64);
        result.push(global_diff - tie_diff);

    }

    Ok(result)
}

// Computes tau-b for many (gene, peak) candidate pairs at once, grouped by
// gene in CSR/CSC-style form (same layout as scipy.sparse: `gene_columns` is
// the list of distinct genes involved, `peak_offsets` marks where each
// gene's slice of `peak_columns` starts/ends). Unlike `_kendall_concordance_diff`,
// which handles one already-sorted gene per call, this function does the
// per-gene sorting itself and runs the per-gene work in parallel across CPU
// cores via rayon, so it must be handed the raw matrices instead of
// pre-sorted indices.
//
// `n_y` is the ATAC tie-correction term for every peak in the full matrix
// (depends only on the peak, not the gene), computed once by the Python
// caller with a single vectorized pass over the whole matrix.
#[pyfunction]
fn _batch_kendall_tau(
    py: Python<'_>,
    rna_matrix: PyReadonlyArray2<'_, f64>,
    atac_matrix: PyReadonlyArray2<'_, u8>,
    gene_columns: PyReadonlyArray1<'_, usize>,
    peak_offsets: PyReadonlyArray1<'_, usize>,
    peak_columns: PyReadonlyArray1<'_, usize>,
    n_y: PyReadonlyArray1<'_, f64>,
) -> PyResult<Vec<f64>> {
    let rna = rna_matrix.as_array();
    let atac = atac_matrix.as_array();
    let gene_columns = gene_columns.as_array().to_vec();
    let peak_offsets = peak_offsets.as_array().to_vec();
    let peak_columns = peak_columns.as_array().to_vec();
    let n_y = n_y.as_array().to_vec();

    let n_cells = rna.shape()[0];
    let n_genes = gene_columns.len();
    let n_0 = (n_cells * (n_cells - 1)) as f64 / 2.0;

    // From here on we only touch plain Rust data (Vecs) and read-only
    // ndarray views, never Python objects, so it's safe to release the GIL
    // for the whole parallel section: other Python threads can run while
    // rayon spreads the per-gene loop below across CPU cores.
    let per_gene_results: Vec<Vec<f64>> = py.allow_threads(|| {
        (0..n_genes)
            .into_par_iter()
            .map(|g| {
                let gene_col = gene_columns[g];
                let start = peak_offsets[g];
                let end = peak_offsets[g + 1];

                // Sort this gene's cells by expression, descending -- same
                // idea as `np.argsort(x)[::-1]` in the single-gene path, but
                // done here so every gene's sort can run concurrently
                // instead of serially in a Python loop.
                let mut order: Vec<usize> = (0..n_cells).collect();
                order.sort_by(|&a, &b| {
                    rna[[b, gene_col]]
                        .partial_cmp(&rna[[a, gene_col]])
                        .expect("RNA expression values must not be NaN")
                });

                // Tie-run boundaries in the sorted expression values, and the
                // RNA tie-correction term n_x, derived together in one pass
                // (same n_x definition as the single-gene Python path).
                let mut new_group = vec![false; n_cells];
                new_group[0] = true;
                let mut n_x = 0.0f64;
                let mut run_len: u64 = 1;
                for i in 1..n_cells {
                    if rna[[order[i], gene_col]] != rna[[order[i - 1], gene_col]] {
                        new_group[i] = true;
                        n_x += (run_len * (run_len - 1)) as f64 / 2.0;
                        run_len = 1;
                    } else {
                        run_len += 1;
                    }
                }
                n_x += (run_len * (run_len - 1)) as f64 / 2.0;

                (start..end)
                    .map(|k| {
                        let peak = peak_columns[k];

                        let mut concordant: u64 = 0;
                        let mut discordant: u64 = 0;
                        let mut cumsum: u64 = 0;

                        let mut tie_concordant: u64 = 0;
                        let mut tie_discordant: u64 = 0;
                        let mut tie_cumsum: u64 = 0;
                        let mut tie_rank: u64 = 0;

                        for (i, &row) in order.iter().enumerate() {
                            let acc = atac[[row, peak]] as u64;
                            cumsum += acc;

                            let i_u64 = i as u64;
                            discordant += acc * (i_u64 + 1 - cumsum);
                            concordant += (1 - acc) * cumsum;

                            if new_group[i] {
                                tie_cumsum = 0;
                                tie_rank = 0;
                            }
                            tie_cumsum += acc;
                            tie_rank += 1;
                            tie_discordant += acc * (tie_rank - tie_cumsum);
                            tie_concordant += (1 - acc) * tie_cumsum;
                        }

                        let diff = (concordant as f64 - discordant as f64)
                            - (tie_concordant as f64 - tie_discordant as f64);
                        let denom = ((n_0 - n_x) * (n_0 - n_y[peak])).sqrt();
                        if denom > 0.0 {
                            diff / denom
                        } else {
                            f64::NAN
                        }
                    })
                    .collect()
            })
            .collect()
    });

    Ok(per_gene_results.concat())
}

#[pymodule]
fn _fast_kendall_sc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_kendall_concordance_diff, m)?)?;
    m.add_function(wrap_pyfunction!(_batch_kendall_tau, m)?)?;
    Ok(())
}