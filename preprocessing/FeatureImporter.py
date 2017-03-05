from preprocessing.FeatureExporter import FeatureExporter
import pandas as pd
import os.path

class FeatureImporter:
    @staticmethod
    def importFromFile(filename,features = FeatureExporter.available_features.keys(),**kargs):
        path_list = FeatureExporter.pathListBuilder(filename,features=features,**kargs)
        return pd.concat(tuple([pd.read_csv(path, index_col=0) for path in path_list]),axis=1).values

    @staticmethod
    def check(filename, features = FeatureExporter.available_features.keys(),**kargs):
        path_list = FeatureExporter.pathListBuilder(filename,features=features,**kargs)
        return all(os.path.isfile(path) for path in path_list)