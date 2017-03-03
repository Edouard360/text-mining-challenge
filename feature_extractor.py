import numpy as np
import nltk
# Git test

stpwds = set(nltk.corpus.stopwords.words("english"))
stemmer = nltk.stem.PorterStemmer()

# from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer = TfidfVectorizer(stop_words="english")
# features_TFIDF = vectorizer.fit_transform(node_information_df["abstract"])

def featuresFromDataFrame(df,node_information_df,verbose = True):
    '''
    :param df: The train_df or test_df with fields "source" and "target"
    :param node_information_df: The information on which to build the features
    :return: nd-array of shape (df.shape[0], n_features)
    '''
    counter = 0
    overlap_title = []
    temp_diff = []
    comm_auth = []
    for source, target in zip(df["source"], df["target"]):
        source_info = node_information_df.ix[source, :]
        target_info = node_information_df.ix[target, :]

        source_title = source_info["title"].lower().split(" ")
        source_title = [token for token in source_title if token not in stpwds]
        source_title = [stemmer.stem(token) for token in source_title]

        target_title = target_info["title"].lower().split(" ")
        target_title = [token for token in target_title if token not in stpwds]
        target_title = [stemmer.stem(token) for token in target_title]

        source_auth = source_info["authors"].split(",")
        target_auth = target_info["authors"].split(",")

        overlap_title.append(len(set(source_title).intersection(set(target_title))))
        temp_diff.append(int(source_info['year']) - int(target_info["year"]))
        comm_auth.append(len(set(source_auth).intersection(set(target_auth))))
        counter += 1
        if verbose and (counter % 1000 == True):
            print(counter, " examples processed")
    return np.array([overlap_title, temp_diff, comm_auth]).T
