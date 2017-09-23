import numpy as np

from featureEngineering.FeatureExtractor import FeatureExtractor
from tools import journals_citation_graph


class JournalFeatureExtractor(FeatureExtractor):
    columns = ["ACiteB", "meanJournalACiteJournalB", "dice_similarity"]

    def __init__(self, node_information_df, verbose=False, freq=10000, **kargs):
        super(JournalFeatureExtractor, self).__init__(node_information_df, verbose=verbose, freq=freq)
        self.g, self.g_sep = journals_citation_graph()
        self.id_to_index = dict(zip(self.node_information_df.index.values, range(self.node_information_df.index.size)))
        self.id_to_journals = self.node_information_df["journalName"].values.tolist()
        self.reset()

    def reset(self):
        self.ACiteB = []
        self.meanACiteB = []
        self.dice_similarity = []

    def extractStep(self, source, target):
        source_journal = self.id_to_journals[self.id_to_index[source]]
        target_journal = self.id_to_journals[self.id_to_index[target]]
        source_journal_sep = filter(None, source_journal.split("."))
        target_journal_sep = filter(None, target_journal.split("."))

        id_source_journal = self.g["journals_to_index"][source_journal]
        id_target_journal = self.g["journals_to_index"][target_journal]

        id_source_sep_journal = [self.g_sep["journals_sep_to_index"][journal] for journal in source_journal_sep]
        id_target_sep_journal = [self.g_sep["journals_sep_to_index"][journal] for journal in target_journal_sep]

        dice_similarity = self.g.similarity_dice(pairs=[(id_source_journal, id_target_journal)])[0]
        ACiteB = self.g.es.select(_between=([id_source_journal], [id_target_journal]))["weight"]
        meanACiteB = self.g_sep.es.select(_between=(id_source_sep_journal, id_target_sep_journal))["weight"]

        if (len(source_journal) == 0 or len(target_journal) == 0):
            dice_similarity = 0
            ACiteB = [0]

        self.dice_similarity.append(dice_similarity)
        self.ACiteB.append(0 if len(ACiteB) == 0 else np.mean(ACiteB))
        self.meanACiteB.append(0 if len(meanACiteB) == 0 else np.mean(meanACiteB))

    def concatFeature(self):
        return np.array([self.ACiteB, self.meanACiteB, self.dice_similarity]).T
