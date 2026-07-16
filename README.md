fast-kendall-sc is a Python package with a Rust backend designed to calculate Kendall's tau rank correlation between continuous and binary variables in single-cell datasets. The implementation is designed to minimize memory overhead and execution time when correlating gene expression with binarized chromatin accessibility.


## Technical Details

**Indirect Sorting:** The program sorts the continuous variable (RNA) once per gene to obtain a vector of sorted indices. These indices are passed directly to the Rust backend.

**Linear-Time Processing:** Using the sorted row index map, the Rust backend computes the concordant and discordant pairs in a single linear pass over the binary matrix, resulting in $O(N)$ time complexity per peak.

**In-Memory Operations:** The Rust library accesses Python memory directly via PyO3 and rust-numpy, avoiding the allocation of intermediate sorted matrices.

**Branchless Execution:** The inner loop avoids conditional branching, enabling consistent execution times and optimization by the compiler.
