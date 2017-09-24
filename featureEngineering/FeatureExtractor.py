class FeatureExtractor:
    """
    An abstract class for the feature extractors. Derived classes are in the subfolders.
    This parent class enables, for all Extractors, to have an insight into the speed of the computation,
    via the "freq" parameter. By default, it will print: 'n samples processed' every 10000 samples.
    """

    def __init__(self, node_information_df, verbose=False, freq=10000):
        self.counter = 0
        self.freq = freq
        self.verbose = verbose
        self.node_information_df = node_information_df
        self.feature = None

    def extractFeature(self, df):
        for source, target in zip(df["source"], df["target"]):
            self.extractStep(source, target)
            self.counter += 1
            if self.verbose and (self.counter % self.freq == 1):
                print(self.counter, " samples processed")
        self.counter = 0
        return self.concatFeature()

    def extractStep(self, source, target):
        raise NotImplementedError("Please implement extractStep for a subclass")

    def reset(self):
        raise NotImplementedError("Please implement reset for a subclass")

    def concatFeature(self):
        raise NotImplementedError("Please implement concatFeature for a subclass")
