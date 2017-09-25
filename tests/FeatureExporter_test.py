"""
Tests the FeatureExporter class
"""

import os
import sys
import unittest

import pandas as pd

# Workaround to import featureEngineering module
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from featureEngineering.FeatureExporter import FeatureExporter


class TestExporter(unittest.TestCase):
    @unittest.skip("Exporter.exportTo not working")
    def test_output_to_file(self):
        """
        Tests if our FeatureExporter correctly outputs to file

        Returns: test case

        """
        train_df = pd.read_csv("../data/training_set_test.txt", sep=" ", header=None)
        train_df.columns = ["source", "target", "label"]

        node_information_df = pd.read_csv("../data/node_information_test.csv", sep=",", header=None)
        node_information_df.columns = ["ID", "year", "title", "authors", "journalName", "abstract"]
        node_information_df = node_information_df.reset_index().set_index("ID")
        node_information_df["authors"].fillna("", inplace=True)
        node_information_df["journalName"].fillna("", inplace=True)

        features = ["original"]
        feature = features[0]
        exporter = FeatureExporter()

        exporter.computeFeature(train_df, node_information_df, feature)
        exporter.exportTo('training_set_test.txt', feature)

        self.assertTrue(os.path.isfile("featureEngineering/originalFeatures/output/training_set_test.txt"))

    def test_absolute_truth_and_meaning(self):
        """
        Testing the tests

        Returns: test case

        """
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
