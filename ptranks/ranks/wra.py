import numpy as np


def wra(array, weights):
    """
    Implements Weighted Ranking Approach (WRA) method described in:
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
    pi_w_1 = np.mean(array, axis=2)  # (n_datasets, n_methods)
    v_j_bar_bar = np.mean(pi_w_1, axis=1)  # (n_datasets, )
    sigma_j = np.std(pi_w_1, axis=1)  # (n_datasets, )
    set_idx_mean = np.mean(np.arange(1, n_methods + 1))
    sigma_r_j = weights * set_idx_mean  # (n_datasets,)

    ranks = np.zeros((n_datasets, n_methods))

    for j in range(n_datasets):

        if np.allclose(sigma_j[j], 0.0):
            ranks[j, :] = set_idx_mean
            continue

        for i in range(n_methods):
            ranks[j, i] = n_methods + 1 - \
                (set_idx_mean -
                 ((pi_w_1[j, i] - v_j_bar_bar[j]) * sigma_r_j[j]) / sigma_j[j])

    avg_ranks = np.mean(ranks, axis=0)

    return avg_ranks
