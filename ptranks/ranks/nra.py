import numpy as np
from scipy.stats import rankdata


def nra(array, weights):
    """
    Implements Naive Ranking Approach (NRA) method described in:
      A New Kind of Nonparametric Test for Statistical Comparison of Multiple Classifiers Over Multiple Datasets

    The method follows scipy convention: Higher rank means higher value of criterion.
    Function added for consistency. It simply calls method from scipy.

    Arguments:
    ----------
    array -- numpy array containing experiment results with dimensions (n_datasets, n_methods, n_runs)
    weights -- numpy array containing dataset-related  weights. (n_datasets, )

    Returns:
    ---------
    Numpy array -- average ranks vector of size (n_methods, )
    """
    pi_0_1 = np.mean(array, axis=2)  # (n_datasets, n_methods)
    ranks = rankdata(pi_0_1, axis=1)
    avg_ranks = np.mean(ranks, axis=0)

    return avg_ranks