from preprocessing.FeatureExtractor import FeatureExtractor
import numpy as np
import pandas as pd
import igraph
import csv


def build_graph(path=""):
    with open(path+"data/training_set.txt", "r") as f:
        reader = csv.reader(f)
        training_set  = list(reader)

    training_set = [element[0].split(" ") for element in training_set]

    with open(path+"data/node_information.csv", "r") as f:
        reader = csv.reader(f)
        node_info  = list(reader)

    IDs = [element[0] for element in node_info]
    edges = [(element[0],element[1]) for element in training_set if element[2]=="1"]
    ## some nodes may not be connected to any other node
    ## hence the need to create the nodes of the graph from node_info.csv,
    ## not just from the edge list
    nodes = IDs
    ## create empty directed graph
    g = igraph.Graph(directed=True)
    ## add vertices
    g.add_vertices(nodes)
    ## add edges
    g.add_edges(edges)
    return g

class InOutFeatureExtractor(FeatureExtractor):
    def __init__(self,node_information,**kargs):
        super(InOutFeatureExtractor, self).__init__(node_information)
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
        self.overlap_title = []
        self.indegree = []
        self.outdegree = []
        self.id_to_index = dict(zip(self.node_information_df.index.values, range(self.node_information_df.index.size)))

    def extractStep(self, source, target):
        self.indegree.append(self.node_degree_df[self.id_to_index[target], 0])
        self.outdegree.append(self.node_degree_df[self.id_to_index[target], 1])

    def concatFeature(self):
        return np.array([self.indegree, self.outdegree]).T

