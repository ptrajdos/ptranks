"""
Testing module
"""
from tests.ranks.rank_test import RankTest
from ptranks.ranks.wgra import wgra


class WGRATest(RankTest):
    """
    Testing class for wgra function
    """

    __test__ = True

    def get_method(self):
        return wgra
