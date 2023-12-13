"""
Implements Global Ranking Approach (GRA) method
"""
import numpy as np
from scipy.stats import rankdata


def gra(array, weights):
    """
    Implements Global Ranking Approach (GRA) method described in:
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
    ranks_global = rankdata(array, axis=1)  # (n_datasets, n_methods, n_runs)
    ranks_global_average = np.mean(ranks_global, axis=2)  # (n_datasets, n_methods,)
    avg_ranks = np.mean(ranks_global_average, axis=0)  # (n_methods,)

    return avg_ranks
