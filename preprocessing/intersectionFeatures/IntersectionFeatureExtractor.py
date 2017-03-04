from preprocessing.FeatureExtractor import FeatureExtractor
from tools import remove_stopwords_and_stem
import numpy as np

class IntersectionFeatureExtractor(FeatureExtractor):
    def __init__(self, node_information_df, **kargs):
        super(IntersectionFeatureExtractor, self).__init__(node_information_df)
        self.overlap_title_target = []

    def extractStep(self, source, target):
        source_info = self.node_information_df.ix[source, :]
        target_info = self.node_information_df.ix[target, :]

        source_title = source_info["title"].lower().split(" ")
        source_title = remove_stopwords_and_stem(source_title)

        target_abstract = target_info["abstract"].lower().split(" ")
        target_abstract = remove_stopwords_and_stem(target_abstract)

        self.overlap_title_target.append(len(set(source_title).intersection(set(target_abstract))))

    def concatFeature(self):
        return np.array([self.overlap_title_target]).T
