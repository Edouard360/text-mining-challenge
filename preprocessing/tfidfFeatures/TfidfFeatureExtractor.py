from preprocessing.FeatureExtractor import FeatureExtractor
from tools import remove_stopwords_and_stem
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from tools import build_graph
import numpy as np
import pandas as pd


class TfidfFeatureExtractor(FeatureExtractor):
    '''
    We are going to compute cosine_similarity for tfidf representation between :
    - source's targets
    - target
    And take the average as feature (or not ? I am so lost I have no idea what I'm doing)
    '''

    def __init__(self, node_information_df, verbose=False, freq=10000, **kargs):
        super(TfidfFeatureExtractor, self).__init__(node_information_df, verbose=verbose, freq=freq)
        self.id_to_index = dict(zip(self.node_information_df.index.values, range(self.node_information_df.index.size)))
        try:
            tfidf_similarity_df = pd.read_csv("preprocessing/tfidfFeatures/tfidf_similarity.csv")
        except FileNotFoundError:
            try:
                ddsim_matrix = np.load("preprocessing/tfidfFeatures/ddsim_matrix.npy")
            except FileNotFoundError:
                print("Building tfidf matrix for the tfidfFeature")
                try:
                    with open('preprocessing/abstractToGraphFeatures/abstract_list.csv', 'r') as f:
                        abstract_list = []
                        for line in f:
                            abstract_list = abstract_list + [line[:-1].split(",")]
                except FileNotFoundError:
                    with open('preprocessing/abstractToGraphFeatures/abstract_list.csv', "w") as f:
                        abstract_list = node_information_df["abstract"].values
                        abstract_list = [remove_stopwords_and_stem(abstract.split(" ")) for abstract in abstract_list]
                        writer = csv.writer(f)
                        writer.writerows(abstract_list)
                # compute TFIDF vector of each paper
                vectorizer = TfidfVectorizer(stop_words="english")
                # TfidfVectorizer prend des listes de string comme argument et non des listes de listes de string
                abstract_list = [' '.join(abstract_list_elt) for abstract_list_elt in abstract_list]
                features_TFIDF = vectorizer.fit_transform(abstract_list)
                # define the doc­doc similarity matrix based on the cosine distance print
                print("Computing the doc­doc similarity matrix based on the cosine distance")
                print("It takes time... But we're not fucking losers eh")
                ddsim_matrix = cosine_similarity(features_TFIDF[:], features_TFIDF)
                print("Doc­doc similarity matrix has been computed and saved, booyah !")
                np.save("preprocessing/tfidfFeatures/ddsim_matrix.npy", ddsim_matrix)
            print("Building graph for the tfidfFeature")
            g = build_graph()
            print("Building tfidf_similarity")
            train_df = pd.read_csv("data/training_set.txt", sep=" ", header=None, usecols=[0, 1])
            train_df.columns = ["source", "target"]
            test_df = pd.read_csv("data/testing_set.txt", sep=" ", header=None)
            test_df.columns = ["source", "target"]
            concatenated_df = pd.concat((train_df, test_df), axis=0)
            list_concat = concatenated_df.values.tolist()
            tfidf_similarity = []
            for source, target in list_concat:
                cosine_similarities = []
                source_targets = g.neighbors(vertex=self.id_to_index[source], mode='out')
                if source_targets:
                    for source_target in source_targets:
                        cosine_similarities.append(ddsim_matrix[self.id_to_index[target], source_target])
                    tfidf_similarity.append(np.mean(cosine_similarities))
                else:
                    tfidf_similarity.append(0)
            concatenated_df["tfidf_similarity"] = tfidf_similarity
            print("Exporting tfidf_similarity to preprocessing/tfidfFeatures/tfidf_similarity.csv")
            concatenated_df.to_csv("preprocessing/tfidfFeatures/tfidf_similarity.csv", index=False)
            tfidf_similarity_df = pd.read_csv("preprocessing/tfidfFeatures/tfidf_similarity.csv")
        self.tfidf_similarity_df = tfidf_similarity_df.set_index(["source", "target"])
        self.tfidf_mean = []

    def reset(self):
        self.tfidf_mean = []

    def extractStep(self, source, target):
        self.tfidf_mean.append(self.tfidf_similarity_df.loc[(source, target)].values[0])

    def concatFeature(self):
        return np.array([self.tfidf_mean]).T
