from preprocessing.FeatureExtractor import FeatureExtractor
import numpy as np
import pandas as pd
from tools import authors_citation_dict, authors_citation_graph, authors_collaboration_graph


class GraphAuthorsFeatureExtractor(FeatureExtractor):
    columns = ["meanACiteB_col", "maxACiteB_col",
               "AOut_col", "BIn_col",
               "ACiteAMean_col", "ACiteASum_col",
               "BOut_col"
               # "meanACollaborateWithB_col", "maxACollaborateWithB_col",
               # "dice_collaboration_mean_col", "dice_collaboration_max_col",
               # "jaccard_collaboration_mean_col", "jaccard_collaboration_max_col"
               ]

    def __init__(self, node_information_df, verbose=False, freq=10000, **kargs):
        super(GraphAuthorsFeatureExtractor, self).__init__(node_information_df, verbose=verbose, freq=freq)
        self.id_to_index = dict(zip(self.node_information_df.index.values, range(self.node_information_df.index.size)))
        print("Building authors citation graph")
        self.authors_citation_graph = authors_citation_graph()
        # self.authors_collaboration_graph = authors_collaboration_graph()
        authors = self.node_information_df["authors"].values.tolist()
        print("Splitting authors list")
        authors = [author_list.split(", ") for author_list in authors]
        print("Filtering authors")
        authors = [list(filter(None, author_list)) for author_list in authors]
        print("Authors name to id")
        self.authors_list_id = [[self.authors_citation_graph["authors_to_index"][author] for author in author_list] for
                                author_list in authors]
        self.meanACiteB_col, self.maxACiteB_col, self.meanACollaborateWithB_col, self.maxACollaborateWithB_col = [], [], [], []
        self.meanACiteB_col, self.maxACiteB_col, self.meanACollaborateWithB_col, self.maxACollaborateWithB_col = [], [], [], []
        self.AOut_col, self.BIn_col = [], []
        self.ACiteASum_col, self.ACiteAMean_col = [], []
        self.BOut_col = []
        # self.dice_collaboration_mean_col, self.dice_collaboration_max_col,\
        # self.jaccard_collaboration_mean_col,self.jaccard_collaboration_max_col = [],[],[],[]

    def reset(self):
        self.meanACiteB_col, self.maxACiteB_col, self.meanACollaborateWithB_col, self.maxACollaborateWithB_col = [], [], [], []
        self.AOut_col, self.BIn_col = [], []
        self.ACiteASum_col, self.ACiteAMean_col = [], []
        self.BOut_col = []
        # self.dice_collaboration_mean_col, self.dice_collaboration_max_col,\
        # self.jaccard_collaboration_mean_col,self.jaccard_collaboration_max_col = [],[],[],[]

    def extractStep(self, source, target):
        source_authors = self.authors_list_id[self.id_to_index[source]]
        target_authors = self.authors_list_id[self.id_to_index[target]]

        ACiteB = self.authors_citation_graph.es.select(_between=(source_authors, target_authors))
        AOut = self.authors_citation_graph.outdegree(source_authors)
        BIn = self.authors_citation_graph.indegree(target_authors)
        BOut = self.authors_citation_graph.outdegree(target_authors)
        # edgeSeqCollaboration = self.authors_collaboration_graph.es.select(_between=(source_authors, target_authors))

        # pairs = [(source,target) for source in source_authors for target in target_authors if source!=target]
        # dice_collaboration = self.authors_collaboration_graph.similarity_dice(pairs=pairs)
        # jaccard_collaboration = self.authors_collaboration_graph.similarity_jaccard(pairs=pairs)

        common_authors = list(set(source_authors) & set(target_authors))
        ACiteA = self.authors_citation_graph.vs[common_authors]["weight"]
        self.ACiteAMean_col.append(0 if len(ACiteA) == 0 else np.mean(ACiteA))
        self.ACiteASum_col.append(0 if len(ACiteA) == 0 else np.sum(ACiteA))

        if (len(ACiteB) > 0):
            meanACiteB = np.mean(ACiteB["weight"])
            maxACiteB = np.max(ACiteB["weight"])
        else:
            meanACiteB = 0
            maxACiteB = 0
        self.meanACiteB_col.append(meanACiteB)
        self.maxACiteB_col.append(maxACiteB)

        self.AOut_col.append(0 if len(AOut) == 0 else np.mean(AOut))
        self.BIn_col.append(0 if len(BIn) == 0 else np.mean(BIn))
        self.BOut_col.append(0 if len(BOut) == 0 else np.mean(BOut))

        # if (len(edgeSeqCollaboration) > 0):
        #     meanACollaborateWithB = np.mean(edgeSeqCollaboration["weight"])
        #     maxACollaborateWithB = np.max(edgeSeqCollaboration["weight"])
        # else:
        #     meanACollaborateWithB = 0
        #     maxACollaborateWithB = 0


        # self.meanACollaborateWithB_col.append(meanACollaborateWithB)
        # self.maxACollaborateWithB_col.append(maxACollaborateWithB)

        # self.dice_collaboration_mean_col.append(0 if len(dice_collaboration)==0 else np.mean(dice_collaboration))
        # self.dice_collaboration_max_col.append(0 if len(dice_collaboration)==0 else np.max(dice_collaboration))
        # self.jaccard_collaboration_mean_col.append(0 if len(dice_collaboration)==0 else np.mean(jaccard_collaboration))
        # self.jaccard_collaboration_max_col.append(0 if len(dice_collaboration)==0 else np.max(jaccard_collaboration))

        # self.mean_authors_citation.append(self.meanWsAuthorsCitation_df.loc[(source, target)].values[0])

    def concatFeature(self):
        return np.array([self.meanACiteB_col, self.maxACiteB_col,
                         self.AOut_col, self.BIn_col,
                         self.ACiteAMean_col, self.ACiteASum_col,
                         self.BOut_col
                         # self.meanACollaborateWithB_col, self.maxACollaborateWithB_col,
                         # self.dice_collaboration_mean_col, self.dice_collaboration_max_col,
                         # self.jaccard_collaboration_mean_col, self.jaccard_collaboration_max_col
                         ]).T
