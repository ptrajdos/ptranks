"""
Testing module
"""
import numpy as np
from tests.ranks.rank_test import RankTest
from ptranks.ranks.gra import gra


class GRATest(RankTest):
    """
    Testing class for gra function
    """

    __test__ = True

    def get_method(self):
        return gra

    def test_gra_paper(self):
        """
        Reproduces example from paper
        """
        gra_fun = self.get_method()

        array = self.get_table_1()
        weights = np.asanyarray([0.5])

        ranks = gra_fun(array, weights)

        expected_ranks = 5 - np.asanyarray([1.8, 1.2, 3.8, 3.2])

        self.assertTrue(np.allclose(ranks, expected_ranks), "Wrong ranks!")
