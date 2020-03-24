# UMAP-kNN
Repository for the batch-incremental UMAP-kNN algorithm implemented in Scikit-multiflow.

For more informations about Scikit-multiflow, check out the official website: 
https://scikit-multiflow.github.io/
<img src="https://scikit-multiflow.github.io/scikit-multiflow/_images/skmultiflow-logo-wide.png" height="80"/>


## Citing Uniform Manifold Approximation and Projection-Based kNN
To cite the UMAP-kNN in a publication, please cite the following paper:

> Maroua Bahri, Bernhard Pfahringer, Albert Bifet, Silviu Maniu.
> Efficient Batch-Incremental Classification for Evolving Data Streams. In the Symposium on Intelligent Data Analysis (IDA), 2020.

## Important source files
The implementation used in this work is the following: 
* batchIncrementalUMAP.py: an example of the batch-incremental UMAP-kNN application.
 

## How to execute it
If you wish to test the UMAP-kNN, you can update its parameters:
* k: the number of neigbors for kNN
* batch: the batch size
* d: the output dimensionality
* w: the maximum number of instances to store inside the sliding window
* stream: the data stream

## Datasets used in the original paper
The real datasets are compressed and available at the root directory. 

