from preprocessing.FeatureExtractor import FeatureExtractor
import numpy as np
import pandas as pd
from tools import build_graph


class PageRankFeatureExtractor(FeatureExtractor):
    columns = ["pageRank"]

    def __init__(self, node_information_df, verbose=False, freq=10000, **kargs):
        super(PageRankFeatureExtractor, self).__init__(node_information_df, verbose=verbose, freq=freq)
        try:
            page_rank_df = pd.read_csv("preprocessing/pageRankFeatures/page_rank.csv", sep=",")
        except Exception:
            print("Building graph for the pageRankFeature")
            g = build_graph()
            page_rank_df = pd.DataFrame(index=g.vs["name"], data={"pageRank": g.pagerank(directed=True)})
            page_rank_df.to_csv("preprocessing/pageRankFeatures/page_rank.csv")
            print("Exporting graph to preprocessing/pageRankFeatures/page_rank.csv")
            page_rank_df = pd.read_csv("preprocessing/pageRankFeatures/page_rank.csv", sep=",")
        page_rank_df.columns = ["ID", "pageRank"]
        self.page_rank_df = page_rank_df.set_index("ID")
        self.pagerank = []
        self.id_to_index = dict(zip(self.node_information_df.index.values, range(self.node_information_df.index.size)))

    def reset(self):
        self.pagerank = []

    def extractStep(self, source, target):
        self.pagerank.append(self.page_rank_df.iloc[self.id_to_index[target]].values[0])

    def concatFeature(self):
        return np.array([self.pagerank]).T
