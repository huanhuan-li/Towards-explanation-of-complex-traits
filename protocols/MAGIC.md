[MAGIC(Markov affinitybased graph imputation of cells)](https://doi.org/10.1016/j.cell.2018.05.061)  
MAGIC restores noisy and sparse single-cell data using diffusion geometry.   
### pre-requests of input data 
1. need normalization! To ensure that distances between cells reflect biology rather than experimental artifact.  
2. square root transform is recommended. Log transformation is frequently used for single-cell RNA-seq, however, this requires the addition of a pseudocount to avoid infinite values at zero. We instead use a square root transform, which has similar properties to the log transform but has no problem with zeroes.
