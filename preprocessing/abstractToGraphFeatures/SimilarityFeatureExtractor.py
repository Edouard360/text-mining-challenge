import os.path
import numpy as np
from scipy import sparse
from preprocessing.FeatureExtractor import FeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing.abstractToGraphFeatures.abstract_to_graph import abstractToGraph, tfIdfFeatures
from preprocessing.abstractToGraphFeatures.weighting_scheme import keepHighVarianceFeatures


class SimilarityFeatureExtractor(FeatureExtractor):
    columns = ["degree", "w_degree", "closeness", "w_closeness", "tfidf"]

    def __init__(self, node_information_df, verbose=False, freq=10000, **kargs):
        super(SimilarityFeatureExtractor, self).__init__(node_information_df, verbose=verbose, freq=freq)
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

        assert metric in ["closeness", "degrees", "w_closeness", "w_degrees",
                          "tfidf"], "You should select an available metric"

        print("All metrics are going to be used")
        print("Metrics have probably never been created.\nMetrics initialization...")
        metrics = ["degrees", "w_degrees", "closeness", "w_closeness"]
        for metric in metrics:
            if not os.path.isfile("preprocessing/abstractToGraphFeatures/metrics/" + metric + ".npz"):
                abstractToGraph()
                tfIdfFeatures()
                break

        degrees_loader = np.load("preprocessing/abstractToGraphFeatures/metrics/degrees.npz")
        w_degrees_loader = np.load("preprocessing/abstractToGraphFeatures/metrics/w_degrees.npz")
        closeness_loader = np.load("preprocessing/abstractToGraphFeatures/metrics/closeness.npz")
        w_closeness_loader = np.load("preprocessing/abstractToGraphFeatures/metrics/w_closeness.npz")
        tfidf_loader = np.load("preprocessing/abstractToGraphFeatures/metrics/w_closeness.npz")

        self.degrees_matrix = keepHighVarianceFeatures(
            sparse.csr_matrix((degrees_loader['data'], degrees_loader['indices'], degrees_loader['indptr']),
                              shape=degrees_loader['shape']), percentile=percentile)
        self.w_degrees_matrix = keepHighVarianceFeatures(
            sparse.csr_matrix((w_degrees_loader['data'], w_degrees_loader['indices'], w_degrees_loader['indptr']),
                              shape=w_degrees_loader['shape']), percentile=percentile)
        self.closeness_matrix = keepHighVarianceFeatures(
            sparse.csr_matrix((closeness_loader['data'], closeness_loader['indices'], closeness_loader['indptr']),
                              shape=closeness_loader['shape']), percentile=percentile)
        self.w_closeness_matrix = keepHighVarianceFeatures(
            sparse.csr_matrix((w_closeness_loader['data'], w_closeness_loader['indices'], w_closeness_loader['indptr']),
                              shape=w_closeness_loader['shape']), percentile=percentile)
        self.tfidf_matrix = keepHighVarianceFeatures(
            sparse.csr_matrix((tfidf_loader['data'], tfidf_loader['indices'], tfidf_loader['indptr']),
                              shape=tfidf_loader['shape']), percentile=percentile)
        self.reset()

    def reset(self):
        self.degrees_feature, self.w_degrees_feature = [], []
        self.closeness_feature, self.w_closeness_feature = [], []
        self.tfidf_feature = []

    def extractStep(self, source, target):
        self.degrees_feature.append(cosine_similarity(self.degrees_matrix[self.id_to_index[source], :].reshape(1, -1),
                                                      self.degrees_matrix[self.id_to_index[target], :].reshape(1, -1))[
                                        0, 0])
        self.w_degrees_feature.append(
            cosine_similarity(self.w_degrees_matrix[self.id_to_index[source], :].reshape(1, -1),
                              self.w_degrees_matrix[self.id_to_index[target], :].reshape(1, -1))[0, 0])
        self.closeness_feature.append(
            cosine_similarity(self.closeness_matrix[self.id_to_index[source], :].reshape(1, -1),
                              self.closeness_matrix[self.id_to_index[target], :].reshape(1, -1))[0, 0])
        self.w_closeness_feature.append(
            cosine_similarity(self.w_closeness_matrix[self.id_to_index[source], :].reshape(1, -1),
                              self.w_closeness_matrix[self.id_to_index[target], :].reshape(1, -1))[0, 0])
        self.tfidf_feature.append(cosine_similarity(self.tfidf_matrix[self.id_to_index[source], :].reshape(1, -1),
                                                    self.tfidf_matrix[self.id_to_index[target], :].reshape(1, -1))[
                                      0, 0])

    def concatFeature(self):
        return np.array([self.degrees_feature, self.w_degrees_feature,
                         self.closeness_feature, self.w_closeness_feature,
                         self.tfidf_feature]).T
