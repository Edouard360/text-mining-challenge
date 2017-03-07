from preprocessing.FeatureExtractor import FeatureExtractor
import numpy as np
import pandas as pd
from tools import build_authors_graph

class AuthorsFeatureExtractor(FeatureExtractor):
    def __init__(self, node_information_df, **kargs):
        super(AuthorsFeatureExtractor, self).__init__(node_information_df)
        self.id_to_index = dict(zip(self.node_information_df.index.values, range(self.node_information_df.index.size)))
        try:
            meanAuthorsCitation_df = pd.read_csv("preprocessing/authorsFeatures/meanAuthorsCitation.csv", header=None)
        except FileNotFoundError:
            print("Building authors graph for the authorFeature")
            dict_authors_edges = build_authors_graph()
            train_df = pd.read_csv("data/training_set.txt", sep=" ", header=None, usecols=[0,1])
            train_df.columns = ["source", "target"]
            test_df = pd.read_csv("data/testing_set.txt", sep=" ", header=None)
            test_df.columns = ["source", "target"]
            concatenated_df = pd.concat((train_df,test_df),axis=0)
            list_concat = concatenated_df.values.tolist()
            authors = node_information_df["authors"].values.tolist()
            authors = [author_list.split(", ") for author_list in authors]
            authors = [list(filter(None, author_list)) for author_list in authors]
            meanAuthorsCitation = []
            print("Computing meanAuthorsCitation list")
            for source, target in list_concat:
                quotes = []
                source_authors = authors[id_to_index[source]]
                target_authors = authors[id_to_index[target]]
                for source_author in source_authors:
                    for target_author in target_authors:
                        if (source_author, target_author) in dict_authors_edges:
                            quotes.append(dict_authors_edges[(source_author, target_author)])
                if quotes:
                    meanAuthorsCitation.append(np.mean(quotes))
                else:
                    meanAuthorsCitation.append(0)
            concatenated_df["meanAuthorsCitation"] = meanAuthorsCitation
            print("Exporting authorsFeatures to preprocessing/authorsFeatures/meanAuthorsCitation.csv")
            concatenated_df.to_csv("preprocessing/authorsFeatures/meanAuthorsCitation.csv", index=False)
            meanAuthorsCitation_df = pd.read_csv("preprocessing/authorsFeatures/meanAuthorsCitation.csv")
        self.meanAuthorsCitation_df = meanAuthorsCitation_df.set_index(["source","target"])
        self.mean_authors_citation = []

    def extractStep(self, source, target):
        self.mean_authors_citation.append(self.meanAuthorsCitation_df.loc[(source, target)].values[0])

    def concatFeature(self):
        return np.array([self.mean_authors_citation]).T
