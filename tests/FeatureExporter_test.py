from time import localtime, strftime
import pandas as pd
from featureEngineering.FeatureExporter import FeatureExporter
#from featureEngineering.FeatureImporter import FeatureImporter
import os


def test_output_to_file():
    train_df = pd.read_csv("data/training_set_test.txt", sep=" ", header=None)
    train_df.columns = ["source", "target", "label"]

    node_information_df = pd.read_csv("data/node_information_test.csv", sep=",", header=None)
    node_information_df.columns = ["ID", "year", "title", "authors", "journalName", "abstract"]
    node_information_df = node_information_df.reset_index().set_index("ID")
    node_information_df["authors"].fillna("", inplace=True)
    node_information_df["journalName"].fillna("", inplace=True)

    features = ["original"]
    feature = features[0]
    exporter = FeatureExporter()

    #assert FeatureImporter.check('training_set_test.txt', features=features) is False

    exporter.computeFeature(train_df, node_information_df, feature)
    exporter.exportTo('training_set_test.txt', feature)

    assert os.path.isfile("featureEngineering/originalFeatures/output/training_set_test.txt")

#training_features = FeatureImporter.importFromFile(df_dict["train"]["filename"], features=features, **parameters)
