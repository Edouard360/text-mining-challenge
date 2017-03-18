import numpy as np
from preprocessing.FeatureExtractor import FeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


class LsaFeatureExtractor(FeatureExtractor):
    columns = ["topic_abstract", "topic_title"]

    def __init__(self, node_information_df, verbose=False, freq=10000, **kargs):
        super(LsaFeatureExtractor, self).__init__(node_information_df, verbose=verbose, freq=freq)

        self.id_to_index = dict(zip(self.node_information_df.index.values, range(self.node_information_df.index.size)))

        abstracts = self.node_information_df["abstract"].values.tolist()
        titles = self.node_information_df["title"].values.tolist()

        vectorizer_title = TfidfVectorizer(stop_words='english', max_df=0.01)
        vectorizer_abstract = TfidfVectorizer(stop_words='english', max_df=0.01)

        print("Vectorizing titles")
        M_title = vectorizer_title.fit_transform(titles)

        M_title = M_title.toarray()
        np.random.shuffle(M_title)
        Mt = M_title[:1000, :]
        print("SVD on titles")
        u_t, s_t, v_t = np.linalg.svd(Mt)
        print("Projecting titles")
        self.titles_topics = np.dot(M_title, v_t[:100, :].transpose())

        print("Vectorizing abstracts")
        M_abstract = vectorizer_abstract.fit_transform(abstracts)
        M_abstract = M_abstract.toarray()
        np.random.shuffle(M_abstract)
        print("SVD on abstracts")
        Ma = M_abstract[:1500, :]
        print("Projecting abstracts")
        u_a, s_a, v_a = np.linalg.svd(Ma)
        self.abstracts_topics = np.dot(M_abstract, v_a[:100, :].transpose())
        self.reset()

    def reset(self):
        self.topic_abstract = []
        self.topic_title = []

    def extractStep(self, source, target):
        source_vector = self.titles_topics[self.id_to_index[source], :].reshape(1, -1)
        target_vector = self.titles_topics[self.id_to_index[target], :].reshape(1, -1)
        if (np.linalg.norm(source_vector) == 0) or (np.linalg.norm(target_vector) == 0):
            self.topic_title.append(0)
        else:
            self.topic_title.append(cosine_similarity(source_vector, target_vector)[0, 0])

        source_vector = self.abstracts_topics[self.id_to_index[source], :].reshape(1, -1)
        target_vector = self.abstracts_topics[self.id_to_index[target], :].reshape(1, -1)
        if (np.linalg.norm(source_vector) == 0) or (np.linalg.norm(target_vector) == 0):
            self.topic_abstract.append(0)
        else:
            self.topic_abstract.append(cosine_similarity(source_vector, target_vector)[0, 0])

    def concatFeature(self):
        return np.array([self.topic_title, self.topic_abstract]).T
