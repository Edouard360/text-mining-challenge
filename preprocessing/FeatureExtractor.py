
class FeatureExtractor:
    """
    An abstract class for the feature extractors.
    Derived classes are in the subfolders
    """
    def __init__(self, node_information_df):
        self.counter = 0
        self.node_information_df = node_information_df
        self.feature = None

    def extractFeature(self, df, verbose=False):
        for source, target in zip(df["source"], df["target"]):
            self.extractStep(source, target)
            self.counter += 1
            if verbose and (self.counter % 10000 == 1):
                print(self.counter, " examples processed")
        return self.concatFeature()

    def extractStep(self,source,target):
        raise NotImplementedError("Please implement extractStep for a subclass")

    def concatFeature(self):
        raise NotImplementedError("Please implement concatFeature for a subclass")
