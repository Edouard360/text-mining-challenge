from preprocessing.FeatureExtractor import FeatureExtractor
import numpy as np
import pandas as pd
from tools import build_graph


class InOutFeatureExtractor(FeatureExtractor):
    def __init__(self, node_information_df, verbose=False, freq=10000, **kargs):
        super(InOutFeatureExtractor, self).__init__(node_information_df, verbose=verbose, freq=freq)
        try:
            node_degree_df = pd.read_csv("preprocessing/inOutFeatures/node_degree.csv", sep=",", header=None)
        except Exception:
            print("Building graph for the inOutFeature")
            g = build_graph()
            node_degree = pd.DataFrame(index=g.vs["name"], data={"indegree": g.indegree(), "outdegree": g.outdegree()})
            node_degree.to_csv("preprocessing/inOutFeatures/node_degree.csv", header=0)
            print("Exporting graph to preprocessing/inOutFeatures/node_degree.csv")
            node_degree_df = pd.read_csv("preprocessing/inOutFeatures/node_degree.csv", sep=",", header=None)
        node_degree_df.columns = ["ID", "indegree", "outdegree"]
        node_degree_df = node_degree_df.reset_index().set_index("ID")
        self.node_degree_df = node_degree_df[["indegree", "outdegree"]].values
        self.indegree = []
        self.outdegree = []
        self.id_to_index = dict(zip(self.node_information_df.index.values, range(self.node_information_df.index.size)))

    def reset(self):
        self.indegree = []
        self.outdegree = []

    def extractStep(self, source, target):
        self.indegree.append(self.node_degree_df[self.id_to_index[target], 0])
        self.outdegree.append(self.node_degree_df[self.id_to_index[target], 1])

    def concatFeature(self):
        return np.array([self.indegree, self.outdegree]).T
