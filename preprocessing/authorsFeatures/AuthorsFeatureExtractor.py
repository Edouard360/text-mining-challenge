from preprocessing.FeatureExtractor import FeatureExtractor
import numpy as np
import pandas as pd
from tools import authors_citation_dict,authors_citation_graph

class AuthorsFeatureExtractor(FeatureExtractor):
    def __init__(self, node_information_df, verbose = False, freq = 10000, **kargs):
        super(AuthorsFeatureExtractor, self).__init__(node_information_df,verbose = verbose, freq = freq)
        self.id_to_index = dict(zip(self.node_information_df.index.values, range(self.node_information_df.index.size)))
        print("Building authors citation graph")
        self.authors_citation_graph = authors_citation_graph()




        # try:
        #     meanWsAuthorsCitation_df = pd.read_csv("preprocessing/authorsFeatures/meanWsAuthorsCitation.csv")
        # except FileNotFoundError:
        #     print("Building authors graph for the authorFeature")
        #     dict_authors_edges = authors_citation_dict()
        #     train_df = pd.read_csv("data/training_set.txt", sep=" ", header=None, usecols=[0,1])
        #     train_df.columns = ["source", "target"]
        #     test_df = pd.read_csv("data/testing_set.txt", sep=" ", header=None)
        #     test_df.columns = ["source", "target"]
        #     concatenated_df = pd.concat((train_df,test_df),axis=0)
        #     list_concat = concatenated_df.values.tolist()
        #     authors = node_information_df["authors"].values.tolist()
        #     authors = [author_list.split(", ") for author_list in authors]
        #     authors = [list(filter(None, author_list)) for author_list in authors]
        #     meanWsAuthorsCitation = []
        #     print("Computing meanWsAuthorsCitation list")
        #     for source, target in list_concat:
        #         quotes = []
        #         source_authors = authors[self.id_to_index[source]]
        #         target_authors = authors[self.id_to_index[target]]
        #         for source_author in source_authors:
        #             for target_author in target_authors:
        #                 if (source_author!= target_author) and (source_author, target_author) in dict_authors_edges:
        #                     quotes.append(dict_authors_edges[(source_author, target_author)])
        #
        #         if quotes:
        #             meanWsAuthorsCitation.append(np.mean(quotes))
        #         else:
        #             meanWsAuthorsCitation.append(0)
        #     concatenated_df["meanWsAuthorsCitation"] = meanWsAuthorsCitation
        #     print("Exporting concatenated authorsFeatures to preprocessing/authorsFeatures/meanWsAuthorsCitation.csv")
        #     concatenated_df.to_csv("preprocessing/authorsFeatures/meanWsAuthorsCitation.csv", index=False)
        #     meanWsAuthorsCitation_df = pd.read_csv("preprocessing/authorsFeatures/meanWsAuthorsCitation.csv")
        # self.meanWsAuthorsCitation_df = meanWsAuthorsCitation_df.set_index(["source","target"])


        authors = self.node_information_df["authors"].values.tolist()
        print("Splitting authors list")
        authors = [author_list.split(", ") for author_list in authors]
        print("Filtering authors")
        authors = [list(filter(None, author_list)) for author_list in authors]
        print("Authors name to id")
        self.authors_list_id = [[self.authors_citation_graph["authors_to_index"][author] for author in author_list] for author_list in authors]
        self.mean_authors_citation = []

    def reset(self):
        self.mean_authors_citation = []

    def extractStep(self, source, target):
        source_authors = self.authors_list_id[self.id_to_index[source]]
        target_authors = self.authors_list_id[self.id_to_index[target]]
        #pairs = [(source,target) for source in source_authors for target in target_authors if source!=target]

        edgeSeq = self.authors_citation_graph.es.select(_between=(source_authors, target_authors))

        #np.mean(self.authors_citation_graph.similarity_dice(pairs=pairs))
        #self.authors_citation_graph.get_eids()
        if(len(edgeSeq["weight"])>0):
            mean = np.mean(edgeSeq["weight"])
        else:
            mean = 0

        self.mean_authors_citation.append(mean)

        #self.mean_authors_citation.append(self.meanWsAuthorsCitation_df.loc[(source, target)].values[0])

    def concatFeature(self):
        return np.array([self.mean_authors_citation]).T
