import pandas as pd
from featureEngineering.originalFeatures.OriginalFeatureExtractor import OriginalFeatureExtractor
# from featureEngineering.graphArticleFeatures.graphArticleFeatureExtractor import GraphArticleFeatureExtractor
# from featureEngineering.graphAuthorsFeatures.GraphAuthorsFeatureExtractor import GraphAuthorsFeatureExtractor
# from featureEngineering.lsaFeatures.lsaFeatureExtractor import LsaFeatureExtractor
# from featureEngineering.journalFeatures.journalFeatureExtractor import JournalFeatureExtractor
# from featureEngineering.abstractFeatures.SimilarityFeatureExtractor import SimilarityFeatureExtractor


class FeatureExporter:
    """
    The FeatureExporter class.
    This class exports features as "*.txt" files in the /output folder of the corresponding feature.
    This coding scheme helps to compute a feature only once.
    All available features are given as a static member to the FeatureExporter object.
    Each feature comes with its own Extractor, its columns names, and the path to the folder containing the feature.
    """
    available_features = {
        "original": {
            "columns": OriginalFeatureExtractor.columns,
            "path": "originalFeatures/",
            "extractor": OriginalFeatureExtractor,
            "default_args": {}
        }
        # "lsa": {
        #     "columns": LsaFeatureExtractor.columns,
        #     "path": "lsaFeatures/",
        #     "extractor": LsaFeatureExtractor,
        #     "default_args": {}
        # },
        # "journal": {
        #     "columns": JournalFeatureExtractor.columns,
        #     "path": "journalFeatures/",
        #     "extractor": JournalFeatureExtractor,
        #     "default_args": {}
        # },
        # "similarity": {
        #     "columns": SimilarityFeatureExtractor.columns,
        #     "path": "abstractFeatures/",
        #     "extractor": SimilarityFeatureExtractor,
        #     "default_args": {"metric": "degrees", "percentile": 0.95}
        # },
        # "graphArticle": {
        #     "columns": GraphArticleFeatureExtractor.columns,
        #     "path": "graphArticleFeatures/",
        #     "extractor": GraphArticleFeatureExtractor,
        #     "default_args": {}
        # },
        # "graphAuthors": {
        #     "columns": GraphAuthorsFeatureExtractor.columns,
        #     "path": "graphAuthorsFeatures/",
        #     "extractor": GraphAuthorsFeatureExtractor,
        #     "default_args": {}
        # }
    }

    def __init__(self, verbose=False, freq=10000):
        self.verbose = verbose
        self.freq = freq
        self.extractor = None
        self.current_feature = None
        self.current_feature_name = None

    @staticmethod
    def pathListBuilder(filename, features=available_features.keys(), **kargs):
        """
        :param filename: a string like "training_set.txt" or "testing_set.txt"
        :param feature: the name of the feature like "lsa"
        This function computes the path to the file, for the export. It can be for instance:
        "featureEngineering/lsaFeatures/output/testing_set.txt"
        """
        path_list = []
        for key, value in FeatureExporter.available_features.items():
            if key in features:
                keys_to_keep = list(set(value["default_args"].keys()) & set(kargs.keys()))
                keys_to_keep.sort()
                suffix = "".join([key_str + "_" + str(kargs[key_str]) + "_" for key_str in keys_to_keep])
                path_list.append("featureEngineering/" + value["path"] + "output/" + suffix + filename)
        assert len(path_list) > 0, "You should select existing features among \n:" + str(
            FeatureExporter.available_features.keys())
        return path_list

    def computeFeature(self, df, node_information_df, feature, **kargs):
        """
        :param df: A dataframe object created from testing_set and training_set.
        :param node_information_df: The node information dataframe (which is used in almost all features)
        :param feature:
        The computeFeature function.
        Computes the feature given as string if it is in the available feature list.
        """
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
        """
        :param filename: a string like "training_set.txt" or "testing_set.txt"
        :param feature: the name of the feature like "lsa"
        :param kargs: Only in case additional parameters are used for the extractor.
        After the computeFeature function, the exportTo function exports the feature as a "*.txt" file.
        The path to this file is computed using the pathListBuilder function, and can be for instance:
        "featureEngineering/lsaFeatures/output/testing_set.txt"
        """
        self.feature = pd.DataFrame(self.feature)
        self.feature.columns = self.current_feature["columns"]
        self.feature.to_csv(FeatureExporter.pathListBuilder(filename, feature, **kargs)[0])
