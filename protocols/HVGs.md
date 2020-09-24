1. If data is count matrix, use Seurat v3 FindVariableFeatures function.  
Computed the mean and variance of each gene using the unnormalized data (i.e., UMI or counts matrix)  
First, fits a line to the relationship of log(variance) and log(mean) using local polynomial regression (loess).   
Then standardizes the feature values using the observed mean and expected variance (given by the fitted line).   
Feature variance is then calculated on the standardized values after clipping to a maximum (see clip.max parameter).  
This variance represents a measure of single-cell dispersion after controlling for mean expression, and was used directly to rank the features. 

2. data imputated with MAGIC
calculate CV of each gene, and kept all genes of 'CV'>1