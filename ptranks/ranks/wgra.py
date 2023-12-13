"""
Implements Weighted Global Ranking Approach (WGRA) method.
"""
import numpy as np


def wgra(array, weights):
    """
    Implements Weighted Global Ranking Approach (WGRA) method described in:
      A New Kind of Nonparametric Test for Statistical Comparison of Multiple Classifiers Over Multiple Datasets

    The method follows scipy convention: Higher rank means higher value of criterion.

    Arguments:
    ----------
    array -- numpy array containing experiment results with dimensions (n_datasets, n_methods, n_runs)
    weights -- numpy array containing dataset-related  weights. (n_datasets, )

    Returns:
    ---------
    Numpy array -- average ranks vector of size (n_methods, )
    """
    n_datasets, n_methods, n_runs = array.shape

    v_j_h_bar = np.mean(array, axis=1)  # (n_datasets, n_runs )
    sigma_j_h = np.std(array, axis=1)  # (n_datasets, n_runs )
    set_idx_mean = np.mean(np.arange(1, n_methods + 1))

    sigma_r_j_h = weights * set_idx_mean  # (n_datasets,)

    ranks = np.zeros((n_datasets, n_methods, n_runs))

    for j in range(n_datasets):
        for h in range(n_runs):
            if np.allclose(sigma_j_h[j, h], 0.0):
                ranks[j, :, h] = set_idx_mean
                continue

            for i in range(n_methods):
                ranks[j, i, h] = (
                    n_methods
                    + 1
                    - (
                        set_idx_mean
                        - ((array[j, i, h] - v_j_h_bar[j, h]) * sigma_r_j_h[j])
                        / sigma_j_h[j, h]
                    )
                )

    tmp_ranks = np.mean(ranks, axis=2)  # (n_datasets, n_methods)
    avg_ranks = np.mean(tmp_ranks, axis=0)

    return avg_ranks
