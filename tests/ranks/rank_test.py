"""
Testing module
"""
import unittest
import numpy as np


class RankTest(unittest.TestCase):
    """
    Base class for ranking function testing
    """

    __test__ = False

    @classmethod
    def setUpClass(cls):
        if not cls.__test__:
            raise unittest.SkipTest("Skipping")

    def get_method(self):
        """
        Returns method to be tested

        """
        raise unittest.SkipTest("Skipping")

    def get_table_1(self):
        """
        Generates example table from paper
        """
        table = np.asanyarray(
            [
                [0.94, 0.95, 0.81, 0.85],  # run 1
                [0.95, 0.96, 0.85, 0.87],  # run 2
                [0.93, 0.94, 0.82, 0.85],  # run 3
                [0.95, 0.94, 0.93, 0.71],  # run 4
                [0.93, 0.97, 0.81, 0.86],  # run 5
            ]
        ).transpose()
        table = np.expand_dims(table, axis=0)
        return table

    def get_table_1_col_means(self):
        """
        Generates column means from paper example table
        """
        return np.mean(self.get_table_1, axis=1)

    def test_sanity(self):
        """
        Tests sanity of the ranking method
        """
        exp_repetitions = 5
        method = self.get_method()
        np.random.seed(0)

        for n_datasets in [1, 3, 5, 10]:
            for n_methods in [1, 3, 5]:
                for n_runs in [1, 3, 5, 10]:
                    for _ in range(exp_repetitions):
                        array = np.random.random((n_datasets, n_methods, n_runs))
                        weights = np.random.random((n_datasets,))

                        ranks = method(array, weights)

                        self.assertIsNotNone(ranks, "Ranks array is None")
                        self.assertIsInstance(
                            ranks, np.ndarray, "Result is not an numpy array"
                        )
                        self.assertFalse(any(np.isnan(ranks)), "Ranks contain NaNs")
                        self.assertFalse(
                            any(np.isinf(ranks)), "Ranks with infinite values"
                        )
                        self.assertTrue(
                            ranks.shape == (n_methods,), "Wrong size of rank array"
                        )

                        desired_rank_sum = n_methods * (n_methods + 1.0) / 2.0
                        rank_sum = np.sum(ranks)

                        self.assertTrue(
                            np.allclose(rank_sum, desired_rank_sum), "Wrong ranks sum"
                        )

    def test_the_same_performance(self):
        """
        Testing ranking methods against uniform criterion value
        """
        exp_repetitions = 5
        method = self.get_method()

        np.random.seed(10)

        for n_datasets in [1, 3, 5, 10]:
            for n_methods in [1, 3, 5]:
                for n_runs in [1, 3, 5, 10]:
                    for _ in range(exp_repetitions):
                        array = np.zeros((n_datasets, n_methods, n_runs))
                        array[:] = np.random.random((1,))
                        weights = np.random.random((n_datasets,))

                        ranks = method(array, weights)

                        self.assertIsNotNone(ranks, "Ranks array is None")
                        self.assertIsInstance(
                            ranks, np.ndarray, "Result is not an numpy array"
                        )
                        self.assertFalse(any(np.isnan(ranks)), "Ranks contain NaNs")
                        self.assertFalse(
                            any(np.isinf(ranks)), "Ranks with infinite values"
                        )
                        self.assertTrue(
                            ranks.shape == (n_methods,), "Wrong size of rank array"
                        )

                        desired_rank_sum = n_methods * (n_methods + 1.0) / 2.0
                        rank_sum = np.sum(ranks)

                        self.assertTrue(
                            np.allclose(rank_sum, desired_rank_sum), "Wrong ranks sum"
                        )

    def test_consistency(self):
        """
        Test rank consistency
        """
        method = self.get_method()
        exp_repetitions = 5

        np.random.seed(11)
        for n_methods in [1, 3, 5]:
            for _ in range(exp_repetitions):
                array = np.zeros((1, n_methods, 1))
                array[0, :, 0] = np.random.random((n_methods,))
                weights = np.asanyarray([0.5])

                ranks = method(array, weights)

                desired_order = (-array[0, :, 0]).argsort()
                rank_order = (-ranks).argsort()

                self.assertTrue(np.allclose(rank_order, desired_order))
