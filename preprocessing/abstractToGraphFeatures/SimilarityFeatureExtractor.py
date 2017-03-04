from preprocessing.FeatureExtractor import FeatureExtractor
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing.abstractToGraphFeatures.abstract_to_graph import abstractToGraph

class SimilarityFeatureExtractor(FeatureExtractor):
    def __init__(self, node_information,**kargs):
        super(SimilarityFeatureExtractor, self).__init__(node_information)
        self.similarity = []
        self.id_to_index = dict(zip(self.node_information_df.index.values, range(self.node_information_df.index.size)))

        if not ("metric" in kargs):
            print("Default metric chosen: 'degrees'")
            metric = 'degrees'
        else:
            metric = kargs["metric"]

        if not ("percentile" in kargs):
            print("Default percentile chosen: 95")
            percentile = 95
        else:
            percentile = kargs["percentile"]

        assert metric in ["closeness","degrees","w_closeness","w_degrees"] ,"You should select an available metric"
        try:
            loader = np.load("preprocessing/abstractToGraphFeatures/metrics/" + metric + ".npz")
        except Exception:
            print("Metrics have probably never been created.\nMetrics initialization...")
            abstractToGraph()
            loader = np.load("preprocessing/abstractToGraphFeatures/metrics/" + metric + ".npz")

        features = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
        E_M2 = np.array((features).multiply(features).mean(axis=0))
        E_M_2 = np.array((features).mean(axis=0)) ** 2
        var = (E_M2 - E_M_2)[0]

        features_to_keep = (var >= np.percentile(var, percentile))
        self.features = features[:, features_to_keep].toarray()

    def extractStep(self, source, target):
        self.similarity.append(cosine_similarity(self.features[self.id_to_index[source], :].reshape(1, -1),
                                            self.features[self.id_to_index[target], :].reshape(1, -1))[0, 0])

    def concatFeature(self):
        return np.array([self.similarity]).T
