import pandas as pd
from preprocessing.abstractToGraphFeatures.SimilarityFeatureExtractor import SimilarityFeatureExtractor
from preprocessing.originalFeatures.OriginalFeatureExtractor import OriginalFeatureExtractor
from preprocessing.inOutFeatures.InOutFeatureExtractor import InOutFeatureExtractor
from preprocessing.commonNeighboursFeatures.commonNeighboursExtractor import CommonNeighboursFeatureExtractor
from preprocessing.intersectionFeatures.IntersectionFeatureExtractor import IntersectionFeatureExtractor
from preprocessing.pageRankFeatures.PageRankFeatureExtractor import PageRankFeatureExtractor
from preprocessing.tfidfFeatures.TfidfFeatureExtractor import TfidfFeatureExtractor
from preprocessing.authorsFeatures.AuthorsFeatureExtractor import AuthorsFeatureExtractor


class FeatureExporter:
    available_features = {
        "original": {
            "columns": OriginalFeatureExtractor.columns,
            "path": "originalFeatures/",
            "extractor": OriginalFeatureExtractor,
            "default_args": {}
        },
        "inOutDegree": {
            "columns": InOutFeatureExtractor.columns,  # A changer avec target_indegree for clarity (EDOUARD T NUL)
            "path": "inOutFeatures/",
            "extractor": InOutFeatureExtractor,
            "default_args": {}
        },
        "similarity": {
            "columns": SimilarityFeatureExtractor.columns,
            "path": "abstractToGraphFeatures/",
            "extractor": SimilarityFeatureExtractor,
            "default_args": {"metric": "degrees", "percentile": 0.95}
        },
        "intersection": {
            "columns": IntersectionFeatureExtractor.columns,
            "path": "intersectionFeatures/",
            "extractor": IntersectionFeatureExtractor,
            "default_args": {}
        },
        "commonNeighbours": {
            "columns": CommonNeighboursFeatureExtractor.columns,
            "path": "commonNeighboursFeatures/",
            "extractor": CommonNeighboursFeatureExtractor,
            "default_args": {}
        },
        "tfidf": {
            "columns": TfidfFeatureExtractor.columns,
            "path": "tfidfFeatures/",
            "extractor": TfidfFeatureExtractor,
            "default_args": {}
        },
        "authors": {
            "columns": AuthorsFeatureExtractor.columns,
            "path": "authorsFeatures/",
            "extractor": AuthorsFeatureExtractor,
            "default_args": {}
        },
        "pageRank": {
            "columns": PageRankFeatureExtractor.columns,
            "path": "pageRankFeatures/",
            "extractor": PageRankFeatureExtractor,
            "default_args": {}
        }
    }

    def __init__(self, verbose=False, freq=10000):
        self.verbose = verbose
        self.freq = freq
        self.extractor = None
        self.current_feature = None
        self.current_feature_name = None

    @staticmethod
    def pathListBuilder(filename, features=available_features.keys(), **kargs):
        path_list = []
        for key, value in FeatureExporter.available_features.items():
            if key in features:
                keys_to_keep = list(set(value["default_args"].keys()) & set(kargs.keys()))
                keys_to_keep.sort()
                suffix = "".join([key_str + "_" + str(kargs[key_str]) + "_" for key_str in keys_to_keep])
                path_list.append("preprocessing/" + value["path"] + "output/" + suffix + filename)
        assert len(path_list) > 0, "You should select existing features among \n:" + str(
            FeatureExporter.available_features.keys())
        return path_list

    def exportAllTo(self, df, node_information_df, filename):
        for key, value in FeatureExporter.available_features.items():
            self.computeFeature(df, node_information_df, key, **(value["default_args"]))
            self.exportTo(filename)

    def computeFeature(self, df, node_information_df, feature, **kargs):
        keys = FeatureExporter.available_features.keys()
        assert feature in keys, "Choose among those features :" + str(keys)
        if not (self.current_feature_name == feature):
            self.current_feature_name = feature
            self.current_feature = FeatureExporter.available_features[feature]
            self.extractor = self.current_feature["extractor"](node_information_df, verbose=self.verbose,
                                                               freq=self.freq, **kargs)
        self.feature = self.extractor.extractFeature(df)
        self.extractor.reset()

    def exportTo(self, filename, feature, **kargs):
        self.feature = pd.DataFrame(self.feature)
        self.feature.columns = self.current_feature["columns"]
        self.feature.to_csv(FeatureExporter.pathListBuilder(filename, feature, **kargs)[0])
