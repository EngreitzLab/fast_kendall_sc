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

// Per-gene precomputation for the fully-sparse batch path below. Built once
// per gene in O(nnz_gene log nnz_gene) -- never O(n_cells) -- and then reused
// for every peak assigned to that gene.
//
// The key idea: we never materialize this gene's full sort order over all
// n_cells cells. We only need two things, both answerable in O(log nnz_gene)
// from data structures sized to the gene's nonzero count:
//   1. `rank_of(cell)`: this cell's 0-indexed position in the (conceptual)
//      full descending sort by expression.
//   2. `group_start_of(rank)` / `group_size_of(rank)`: the RNA tie-group that
//      rank belongs to, needed for the tie-correction term.
//
// Cells absent from the gene's sparse column have implicit expression 0 and
// all sort after every explicit (necessarily positive -- enforced by the
// Python wrapper) value. Within that implicit-zero block we're free to
// assign ranks in any fixed, consistent order (they're all mutually tied, so
// the choice can't affect the final tie-corrected result) -- we pick natural
// cell-id order, which is exactly what makes `rank_of` computable by binary
// search instead of needing a full n_cells array.
struct GeneSortInfo {
    n_cells: usize,
    nnz: usize,
    n_x: f64,
    // Explicit cells sorted by cell id (not by expression), paired with the
    // rank each was assigned by the value-descending sort. Enables
    // `rank_of` via binary search: found => explicit, use its rank;
    // not found => implicit zero, derive its rank from the insertion point.
    by_cell_id: Vec<(usize, usize)>,
    // Indexed by rank (only ranks < nnz are ever explicit).
    group_start_by_rank: Vec<usize>,
    group_size_by_rank: Vec<usize>,
}

impl GeneSortInfo {
    fn build(
        gene_col: usize,
        n_cells: usize,
        rna_data: &[f64],
        rna_indices: &[usize],
        rna_indptr: &[usize],
    ) -> Self {
        let mut explicit: Vec<(usize, f64)> = (rna_indptr[gene_col]..rna_indptr[gene_col + 1])
            .map(|k| (rna_indices[k], rna_data[k]))
            .collect();
        // Rank = position in this sort (0 = highest expression).
        explicit.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("RNA expression must not be NaN"));
        let nnz = explicit.len();

        // Tie-run boundaries among the explicit ranks, and n_x's contribution
        // from them -- same logic as the dense/per-column paths, just scoped
        // to nnz explicit entries instead of n_cells total.
        let mut group_start_by_rank = vec![0usize; nnz];
        let mut group_size_by_rank = vec![0usize; nnz];
        let mut n_x = 0.0f64;
        let mut group_begin = 0usize;
        for r in 1..nnz {
            if explicit[r].1 != explicit[r - 1].1 {
                let size = r - group_begin;
                for rr in group_begin..r {
                    group_start_by_rank[rr] = group_begin;
                    group_size_by_rank[rr] = size;
                }
                n_x += (size * (size - 1)) as f64 / 2.0;
                group_begin = r;
            }
        }
        if nnz > 0 {
            let size = nnz - group_begin;
            for rr in group_begin..nnz {
                group_start_by_rank[rr] = group_begin;
                group_size_by_rank[rr] = size;
            }
            n_x += (size * (size - 1)) as f64 / 2.0;
        }
        // The implicit-zero block is one single tied group of known size --
        // its n_x contribution is a closed form, no scan needed.
        let n_zeros = n_cells - nnz;
        if n_zeros > 0 {
            n_x += (n_zeros * (n_zeros - 1)) as f64 / 2.0;
        }

        let mut by_cell_id: Vec<(usize, usize)> = explicit
            .iter()
            .enumerate()
            .map(|(rank, &(cell_id, _))| (cell_id, rank))
            .collect();
        by_cell_id.sort_unstable_by_key(|&(cell_id, _)| cell_id);

        GeneSortInfo { n_cells, nnz, n_x, by_cell_id, group_start_by_rank, group_size_by_rank }
    }

    fn rank_of(&self, cell_id: usize) -> usize {
        match self.by_cell_id.binary_search_by_key(&cell_id, |&(id, _)| id) {
            Ok(pos) => self.by_cell_id[pos].1,
            // `idx` explicit cells have id < cell_id, so among the
            // implicit-zero cells (ordered by their own cell id) there are
            // exactly (cell_id - idx) of them before this one.
            Err(idx) => self.nnz + (cell_id - idx),
        }
    }

    fn group_start_of(&self, rank: usize) -> usize {
        if rank < self.nnz { self.group_start_by_rank[rank] } else { self.nnz }
    }

    fn group_size_of(&self, rank: usize) -> usize {
        if rank < self.nnz { self.group_size_by_rank[rank] } else { self.n_cells - self.nnz }
    }
}

// Scores one peak against one gene using only the peak's accessible
// (nonzero) cells -- O(nnz_peak log(nnz_gene + nnz_peak)) instead of the
// O(n_cells) a full sweep would need. This is the same tau-b computation as
// `_batch_kendall_tau_sparse`'s old per-peak loop, re-derived so it never
// touches a "zero" row directly.
//
// Both concordant/discordant and their tie-corrected counterparts are sums
// where each accessible cell's contribution depends only on its rank among
// the *other accessible cells* (`k`, its 0-indexed position in ascending
// rank order) and the length of the zero-run before it. Since all of that
// is determined by the sorted list of ranks alone, we never need to visit
// the (usually far more numerous) zero rows individually -- their aggregate
// effect is captured by the gaps between consecutive ranks.
fn score_peak(gene_info: &GeneSortInfo, atac_indices: &[usize], peak_start: usize, peak_end: usize) -> f64 {
    let n_cells = gene_info.n_cells as i64;

    let mut ranks: Vec<usize> = atac_indices[peak_start..peak_end]
        .iter()
        .map(|&cell_id| gene_info.rank_of(cell_id))
        .collect();
    ranks.sort_unstable();
    let m = ranks.len();

    let mut discordant: i64 = 0;
    let mut concordant: i64 = 0;
    let mut prev_rank: i64 = -1;
    for (k, &r) in ranks.iter().enumerate() {
        let r = r as i64;
        discordant += r - k as i64;
        concordant += k as i64 * (r - prev_rank - 1);
        prev_rank = r;
    }
    concordant += m as i64 * (n_cells - 1 - prev_rank);

    // Same formulas again, but restricted independently to each RNA
    // tie-group: since groups are contiguous in rank-space and `ranks` is
    // sorted, each group's accessible ranks form one contiguous run here.
    let mut tie_discordant: i64 = 0;
    let mut tie_concordant: i64 = 0;
    let mut i = 0usize;
    while i < m {
        let group_start = gene_info.group_start_of(ranks[i]) as i64;
        let group_size = gene_info.group_size_of(ranks[i]) as i64;
        let mut j = i;
        while j < m && gene_info.group_start_of(ranks[j]) as i64 == group_start {
            j += 1;
        }

        let mut prev_local: i64 = -1;
        for (k, &r) in ranks[i..j].iter().enumerate() {
            let local_r = r as i64 - group_start;
            tie_discordant += local_r - k as i64;
            tie_concordant += k as i64 * (local_r - prev_local - 1);
            prev_local = local_r;
        }
        tie_concordant += (j - i) as i64 * (group_size - 1 - prev_local);

        i = j;
    }

    (concordant as f64 - discordant as f64) - (tie_concordant as f64 - tie_discordant as f64)
}

// Same interface and semantics as the earlier dense-per-column version of
// this function: reads CSC sparse arrays directly, so a dense (n_cells x
// n_genes) or (n_cells x n_peaks) matrix is never required. The difference
// is algorithmic, not memory-related -- see `GeneSortInfo` and `score_peak`
// above. Each peak now costs O(nnz_peak) instead of O(n_cells), which both
// speeds up the common case (real ATAC accessibility is typically a few
// percent open) and turns this from a memory-bandwidth-bound sweep into a
// compute-bound one that actually benefits from more CPU cores.
//
// Preconditions enforced by the Python wrapper, not re-checked here: sparse
// zero entries have been eliminated (`.eliminate_zeros()`), and RNA values
// are all >= 0 (see `GeneSortInfo`'s doc comment for why that matters).
#[pyfunction]
fn _batch_kendall_tau_sparse(
    py: Python<'_>,
    n_cells: usize,
    rna_data: PyReadonlyArray1<'_, f64>,
    rna_indices: PyReadonlyArray1<'_, usize>,
    rna_indptr: PyReadonlyArray1<'_, usize>,
    atac_indices: PyReadonlyArray1<'_, usize>,
    atac_indptr: PyReadonlyArray1<'_, usize>,
    gene_columns: PyReadonlyArray1<'_, usize>,
    peak_offsets: PyReadonlyArray1<'_, usize>,
    peak_columns: PyReadonlyArray1<'_, usize>,
    n_y: PyReadonlyArray1<'_, f64>,
) -> PyResult<Vec<f64>> {
    // These borrow directly from the numpy buffers rather than copying: RNA
    // and ATAC are the full matrices, so copying them here would cost O(total
    // matrix nnz) on every call regardless of how few genes/peaks a given
    // batch actually touches -- easily dwarfing the O(nnz_gene + nnz_peak)
    // work this function exists to make cheap. Only the small,
    // pairs-sized bookkeeping arrays are worth materializing as owned Vecs.
    let rna_data = rna_data.as_slice().expect("rna_data must be contiguous");
    let rna_indices = rna_indices.as_slice().expect("rna_indices must be contiguous");
    let rna_indptr = rna_indptr.as_slice().expect("rna_indptr must be contiguous");
    let atac_indices = atac_indices.as_slice().expect("atac_indices must be contiguous");
    let atac_indptr = atac_indptr.as_slice().expect("atac_indptr must be contiguous");
    let gene_columns = gene_columns.as_array().to_vec();
    let peak_offsets = peak_offsets.as_array().to_vec();
    let peak_columns = peak_columns.as_array().to_vec();
    let n_y = n_y.as_array().to_vec();

    let n_genes = gene_columns.len();
    let n_0 = (n_cells * (n_cells - 1)) as f64 / 2.0;

    let per_gene_results: Vec<Vec<f64>> = py.allow_threads(|| {
        (0..n_genes)
            .into_par_iter()
            .map(|g| {
                let gene_col = gene_columns[g];
                let gene_info = GeneSortInfo::build(gene_col, n_cells, rna_data, rna_indices, rna_indptr);

                let start = peak_offsets[g];
                let end = peak_offsets[g + 1];
                (start..end)
                    .map(|k| {
                        let peak = peak_columns[k];
                        let diff = score_peak(&gene_info, atac_indices, atac_indptr[peak], atac_indptr[peak + 1]);
                        let denom = ((n_0 - gene_info.n_x) * (n_0 - n_y[peak])).sqrt();
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
    m.add_function(wrap_pyfunction!(_batch_kendall_tau_sparse, m)?)?;
    Ok(())
}