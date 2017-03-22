from featureEngineering.FeatureExtractor import FeatureExtractor
from tools import remove_stopwords_and_stem
import numpy as np


class OriginalFeatureExtractor(FeatureExtractor):
    columns = ["overlap_title", "temp_diff", "comm_auth", "overlap_journal"]

    def __init__(self, node_information_df, verbose=False, freq=10000, **kargs):
        super(OriginalFeatureExtractor, self).__init__(node_information_df, verbose=verbose, freq=freq)
        self.overlap_title = []
        self.temp_diff = []
        self.comm_auth = []
        self.overlap_journal = []

    def extractStep(self, source, target):
        source_info = self.node_information_df.ix[source, :]
        target_info = self.node_information_df.ix[target, :]

        source_title = source_info["title"].lower().split(" ")
        source_title = remove_stopwords_and_stem(source_title)

        target_title = target_info["title"].lower().split(" ")
        target_title = remove_stopwords_and_stem(target_title)

        # Just saying
        source_auth = source_info["authors"].split(", ")
        target_auth = target_info["authors"].split(", ")

        source_journal = source_info["journalName"].lower().split(".")

        target_journal = source_info["journalName"].lower().split(".")

        self.overlap_title.append(len(set(source_title).intersection(set(target_title))))
        self.temp_diff.append(int(source_info['year']) - int(target_info["year"]))
        self.comm_auth.append(len(set(source_auth).intersection(set(target_auth))))
        self.overlap_journal.append(len(set(source_journal).intersection(set(target_journal))))

    def reset(self):
        self.overlap_title = []
        self.temp_diff = []
        self.comm_auth = []
        self.overlap_journal = []

    def concatFeature(self):
        return np.array([self.overlap_title, self.temp_diff, self.comm_auth, self.overlap_journal]).T
