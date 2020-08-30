# Data Mining - Clustering

This is an Python implementation of Bradley-Fayyad-Reina (BFR)[^1] algorithm using Spark. 

Python command: $ python3 bfr\.py <input_path> <n_cluster> <out_file1> <out_file2>

Input file format: index, axis, ...
 - an example with two points in three dimensions
    > 0,-51.718,-16.014,-32.745

    > 1,-94.947,378.966,-277.579

Output file1(json) format: a point index with its corresponding cluster index
 - an example of three clustered points
    > {"0": 0, "1": 0, "2": 2}

Output file2(csv) format: intermidiate results including “round id” (starting from 1), “the number of clusters in the discard set”, “the total number of the discarded points”, “the number of clusters in the compression set”, “the total number of the compressed points”, and “the number of points in the retained set”.
 - an example of two points
    > round id, nof_cluster_discard, nof_point_discard, nof_cluster_compression, nof_point_compression, nof_point_retained
    
    > 1, 10, 2898, 20, 147, 82
    
    > 2, 10, 5426, 14, 256, 15


---
[^1] BFR algorithm is a variant of K-Means desinated to handle very large data sets. It assumes that clusters are normally distributed around a centroid in a Euclidean space. You can find more information [here](https://www.aaai.org/Papers/KDD/1998/KDD98-002.pdf).