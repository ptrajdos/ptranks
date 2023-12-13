"""
Testing module
"""
import numpy as np
from tests.ranks.rank_test import RankTest
from ptranks.ranks.nra import nra


class NRATest(RankTest):
    """
    Testing class for nra function
    """

    __test__ = True

    def get_method(self):
        return nra

    def test_nra_paper(self):
        """
        Reproduces example from paper
        """
        nra_fun = self.get_method()

        array = self.get_table_1()
        weights = np.asanyarray([0.5])

        ranks = nra_fun(array, weights)

        expected_ranks = 5 - np.asanyarray([2, 1, 3, 4])

        self.assertTrue(np.allclose(ranks, expected_ranks), "Wrong ranks!")
