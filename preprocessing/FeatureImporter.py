from preprocessing.FeatureExporter import FeatureExporter
import pandas as pd
import os.path

class FeatureImporter:
    @staticmethod
    def importFromFile(filename,features = FeatureExporter.available_features.keys()):
        path_list = ["preprocessing/"+value["path"]+"output/"+filename for key, value in FeatureExporter.available_features.items() if key in features]
        assert len(path_list)>0, "You should select existing features among \n:"+str(FeatureExporter.available_features.keys())
        return pd.concat(tuple([pd.read_csv(path, index_col=0) for path in path_list]),axis=1).values

    @staticmethod
    def check(filename, features = FeatureExporter.available_features.keys()):
        path_list = ["preprocessing/" + value["path"] + "output/" + filename for key, value in
                     FeatureExporter.available_features.items() if key in features]
        return all(os.path.isfile(path) for path in path_list)


