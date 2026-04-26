"""
Testing module
"""
import numpy as np
from tests.ranks.rank_test import RankTest
from ptranks.ranks.wra import wra


class WRATest(RankTest):
    """
    Testing class for wra function
    """

    __test__ = True

    def get_method(self):
        return wra

    def test_wra_paper(self):
        """
        Reproduces example from paper
        """
        wra_fun = self.get_method()

        array = self.get_table_1()
        weights = None
        ranks = wra_fun(array, weights)

        # A bit different from paper due to different rounding
        expected_ranks = 5 - np.asanyarray(
            [1.315, 1.024, 3.637, 4.024]
        )
        self.assertTrue(np.allclose(ranks, expected_ranks, atol=1e-3), "Wrong ranks!")
