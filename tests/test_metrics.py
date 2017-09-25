# -*- coding: utf-8 -*-
import inspect
import os
import sys
import unittest

import numpy as np
import pandas as pd

# Workaround to import tools module
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import metrics


class TestMetricBehavior(unittest.TestCase):
    """
    Test cases:
        - There should be no NaN in the output of *_ignoring_nans metrics
    """

    def setUp(self):
        """
        Loads data for testing
        Returns:
            None
        """
        self.all_metrics = inspect.getmembers(metrics, inspect.isfunction)
        # Tuples of vectors to test (y_true, y_estimated)
        self.data_tuples = [
            # Data Types
            # Only numpy one
            (np.arange(1, 6), np.array([6, 4, 7, 1, 2])),
            # Only pandas
            (pd.Series(data=[2, 3, 5, 3, 2]), pd.Series(data=[2, 13, 53, 34, 2])),
            # Mixed
            (np.arange(1, 6), pd.Series(data=[2, 13, 53, 34, 2])),

            # Zeros
            # Zeros in target
            (np.array([1, 2, 0]), np.array([2, 3, 4])),
            # Zeros in estimated
            (np.array([1, 2, 1]), np.array([2, 0, 4])),
            # Zeros in target and estimated
            (np.array([1, 0, 1]), np.array([2, 0, 4])),

            # NaNs
            # NaNs in target
            (np.array([1, 2, np.nan]), np.array([2, 3, 4])),
            # NaNs in estimated
            (np.array([1, 2, 4]), np.array([2, 3, np.nan])),
            # NaNs in target and estimated
            (np.array([1, 2, np.nan]), np.array([2, 3, np.nan])),

        ]

    def tearDown(self):
        """
        Delete loaded data
        Returns:
            None
        """
        pass

    # Test if we return np.nan
    def test_all_ignoring_nans_metrics_for_numpy_nans(self):
        for metric_name, metric_callable in self.all_metrics:
            if 'ignoring_nans' in metric_name and metric_name != 'mean_absolute_scaled_error_ignoring_nans':
                for i, (y_true, y_estimated) in enumerate(self.data_tuples):
                    with self.subTest(i=i):
                        # Percentage errors will return NaN for 0/0
                        # noinspection PyTypeChecker
                        # Find if common zeros
                        common_zeros_indices = np.intersect1d(np.nonzero(y_true == 0), np.nonzero(y_estimated == 0))
                        # Disable inspection if common zeros for percentage metrics
                        if 'percentage' in metric_name and common_zeros_indices.size != 0:
                            continue
                        return_is_not_nan = ~np.isnan(metric_callable(y_true=y_true,
                                                                      y_estimated=y_estimated))
                        self.assertTrue(expr=return_is_not_nan, msg='{} returns NaN'
                                                                    ' for y_true: {},'
                                                                    ' y_estimated: {}'.format(metric_name,
                                                                                              y_true,
                                                                                              y_estimated))

    # # Test if we return None
    def test_all_ignoring_nans_metrics_for_None(self):
        for metric_name, metric_callable in self.all_metrics:
            if 'ignoring_nans' in metric_name and metric_name != 'mean_absolute_scaled_error_ignoring_nans':
                for i, (y_true, y_estimated) in enumerate(self.data_tuples):
                    with self.subTest(i=i):
                        self.assertIsNotNone(obj=metric_callable(y_true=y_true,
                                                                 y_estimated=y_estimated),
                                             msg='{} returns None'
                                                 ' for y_true: {},'
                                                 ' y_estimated: {}'.format(metric_name,
                                                                           y_true,
                                                                           y_estimated))


if __name__ == '__main__':
    unittest.main()
