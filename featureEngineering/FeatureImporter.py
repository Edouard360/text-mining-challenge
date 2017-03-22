from featureEngineering.FeatureExporter import FeatureExporter
import pandas as pd
import os.path


class FeatureImporter:
    """
    The FeatureImporter class with only static functions as attributes.
    Given a filename, FeatureImporter can first check whether the file exist.
    If not, it will be computed and exported by the FeatureExtractor class.
    Then, FeatureImporter can simply import the file. The path can be like:
    'featureEngineering/lsaFeatures/output/testing_set.txt'
    """
    @staticmethod
    def importFromFile(filename, features=FeatureExporter.available_features.keys(), **kargs):
        path_list = FeatureExporter.pathListBuilder(filename, features=features, **kargs)
        return pd.concat(tuple([pd.read_csv(path, index_col=0) for path in path_list]), axis=1).values

    @staticmethod
    def check(filename, features=FeatureExporter.available_features.keys(), **kargs):
        path_list = FeatureExporter.pathListBuilder(filename, features=features, **kargs)
        return all(os.path.isfile(path) for path in path_list)
