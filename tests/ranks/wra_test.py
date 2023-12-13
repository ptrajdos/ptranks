
from tests.ranks.rank_test import RankTest
from ptranks.ranks.wra import wra

import numpy as np

class WRATest(RankTest):

    __test__ = True
    
    def get_method(self):
        return wra
    
    def test_wra_paper(self):

        wra = self.get_method()

        array = self.get_table_1()
        weights = np.asanyarray( [0.61] )

        ranks = wra(array, weights)

        #A bit different from paper due to different rounding 
        expected_ranks = 5 - np.asanyarray( [1.152454731, 0.8224436447, 3.7925434213, 4.232558203] )

        self.assertTrue( np.allclose(ranks, expected_ranks), "Wrong ranks!")
