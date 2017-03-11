from preprocessing.FeatureExtractor import FeatureExtractor
import numpy as np
import pandas as pd
from tools import build_graph
import igraph


class CommonNeighboursFeatureExtractor(FeatureExtractor):
    def __init__(self, node_information_df, verbose=False, freq=10000, **kargs):
        super(CommonNeighboursFeatureExtractor, self).__init__(node_information_df, verbose=verbose, freq=freq)
        self.id_to_index = dict(zip(self.node_information_df.index.values, range(self.node_information_df.index.size)))
        try:
            common_neighbours_df = pd.read_csv("preprocessing/commonNeighboursFeatures/common_neighbours.csv")
        except Exception:
            print("Building graph for the commonNeighboursFeature")
            g = build_graph()

            train_df = pd.read_csv("data/training_set.txt", sep=" ", header=None, usecols=[0, 1])
            train_df.columns = ["source", "target"]
            test_df = pd.read_csv("data/testing_set.txt", sep=" ", header=None)
            test_df.columns = ["source", "target"]
            concatenated_df = pd.concat((train_df, test_df), axis=0)
            list_concat = concatenated_df.values.tolist()
            list_concat = [tuple(concat) for concat in list_concat]
            print("Building list_concat")
            list_concat = [tuple([self.id_to_index[concat[0]], self.id_to_index[concat[1]]]) for concat in list_concat]
            print("Similarity Dice computation")
            concatenated_df["commonNeighbours"] = g.similarity_dice(pairs=list_concat)
            concatenated_df.to_csv("preprocessing/commonNeighboursFeatures/common_neighbours.csv", index=False)
            print("Exporting graph to preprocessing/commonNeighboursFeatures/common_neighbours.csv")
            common_neighbours_df = pd.read_csv("preprocessing/commonNeighboursFeatures/common_neighbours.csv")
        self.common_neighbours_df = common_neighbours_df.set_index(["source", "target"])
        self.commonNeighbours = []

    def reset(self):
        self.commonNeighbours = []

    def extractStep(self, source, target):
        self.commonNeighbours.append(self.common_neighbours_df.loc[(source, target)].values[0])

    def concatFeature(self):
        return np.array([self.commonNeighbours]).T
