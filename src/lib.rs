use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};

#[pyfunction]
fn _kendall_concordance_diff(
    y_matrix: PyReadonlyArray2<'_, u8>,
    sorted_idx: PyReadonlyArray1<'_, usize>,
    peak_columns: PyReadonlyArray1<'_, usize>,
) -> PyResult<Vec<f64>> {
    let y = y_matrix.as_array();
    let idx = sorted_idx.as_array();
    let peaks = peak_columns.as_array();

    let mut result = Vec::with_capacity(peaks.len());

    for &j in peaks.iter() {
        let mut concordant: u64 = 0;
        let mut discordant: u64 = 0;
        let mut cumsum: u64 = 0;

        for (i, &original_row) in idx.iter().enumerate() {
            let acc = y[[original_row, j]] as u64; 
            cumsum += acc;
            
            let i_u64 = i as u64;
            discordant += acc * (i_u64 + 1 - cumsum);
            concordant += (1 - acc) * cumsum;
        }
        
        result.push((concordant as f64) - (discordant as f64));
    }

    Ok(result)
}

#[pymodule]
fn _fast_kendall_sc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_kendall_concordance_diff, m)?)?;
    Ok(())
}