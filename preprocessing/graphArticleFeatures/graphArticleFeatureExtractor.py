from preprocessing.FeatureExtractor import FeatureExtractor
import numpy as np
import pandas as pd
from tools import articles_graph


class GraphArticleFeatureExtractor(FeatureExtractor):
    columns = ["indegree", "outdegree", "commonNeighbours", "coreness_in_f", "coreness_out_f", "coreness_all_f",
               "id_to_cluster_max", "pagerank"]

    def __init__(self, node_information_df, verbose=False, freq=10000, **kargs):
        super(GraphArticleFeatureExtractor, self).__init__(node_information_df, verbose=verbose, freq=freq)
        self.id_to_index = dict(zip(self.node_information_df.index.values, range(self.node_information_df.index.size)))
        self.index_to_cluster = dict(
            zip(range(self.node_information_df.index.size), np.zeros(len(self.node_information_df.index))))
        self.articles_graph = articles_graph()
        for i, id_articles_list in enumerate(self.articles_graph.clusters(mode="WEAK")):
            for id_article in id_articles_list:
                self.index_to_cluster[id_article] = i
        self.node_degree_df = pd.DataFrame(
            data={"ID": self.articles_graph.vs["name"],
                  "indegree": self.articles_graph.indegree(),
                  "outdegree": self.articles_graph.outdegree()}
        )
        # self.node_degree_df.columns = ["ID", "indegree", "outdegree"]
        self.node_degree_df = self.node_degree_df.reset_index().set_index("ID")

        self.pagerank = self.articles_graph.pagerank()
        self.coreness_in = self.articles_graph.coreness(mode="in")
        self.coreness_out = self.articles_graph.coreness(mode="out")
        self.coreness_all = self.articles_graph.coreness(mode="all")
        self.node_degree_df = self.node_degree_df[["indegree", "outdegree"]].values
        self.reset()

    def reset(self):
        self.indegree = []
        self.outdegree = []
        self.commonNeighbours = []
        self.coreness_in_f = []
        self.coreness_out_f = []
        self.coreness_all_f = []
        self.id_to_cluster_max = []
        self.pagerank_f = []

    def extractStep(self, source, target):
        index_source = self.articles_graph["articles_to_index"][source]
        index_target = self.articles_graph["articles_to_index"][target]

        self.id_to_cluster_max.append(max(self.index_to_cluster[index_source], self.index_to_cluster[index_target]))
        self.coreness_in_f.append(self.coreness_in[index_source] - self.coreness_in[index_target])
        self.coreness_out_f.append(self.coreness_out[index_source] - self.coreness_out[index_target])
        self.coreness_all_f.append(self.coreness_all[index_source] - self.coreness_all[index_target])
        self.commonNeighbours.append(self.articles_graph.similarity_dice(pairs=[(index_source, index_target)])[0])
        self.pagerank_f.append(max(self.pagerank[index_source], self.pagerank[index_target]))
        self.indegree.append(self.node_degree_df[self.id_to_index[target], 0])
        self.outdegree.append(self.node_degree_df[self.id_to_index[target], 1])

    def concatFeature(self):
        return np.array([self.commonNeighbours, self.coreness_in_f, self.coreness_out_f, self.coreness_all_f,
                         self.id_to_cluster_max, self.pagerank_f, self.indegree, self.outdegree]).T
