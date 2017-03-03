import numpy as np
from tools import remove_stopwords_and_stem
import csv
import pandas as pd
from scipy import sparse
from library_moodle import terms_to_graph,compute_node_centrality
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

# vectorizer = TfidfVectorizer(stop_words="english")
# features_TFIDF = vectorizer.fit_transform(df["abstract"].values)

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
        source_title = remove_stopwords_and_stem(source_title)

        target_title = target_info["title"].lower().split(" ")
        target_title = remove_stopwords_and_stem(target_title)

        source_auth = source_info["authors"].split(",")
        target_auth = target_info["authors"].split(",")

        overlap_title.append(len(set(source_title).intersection(set(target_title))))
        temp_diff.append(int(source_info['year']) - int(target_info["year"]))
        comm_auth.append(len(set(source_auth).intersection(set(target_auth))))
        counter += 1
        if verbose and (counter % 1000 == True):
            print(counter, " examples processed")
    return np.array([overlap_title, temp_diff, comm_auth]).T

def addFeaturesFromMetric(node_information_df, metric = "degrees", percentile = 95):
    loader = np.load("preprocessing/node-abstract/node-feature-"+metric+".npz")
    features = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                          shape=loader['shape'])
    E_M2 = np.array((features).multiply(features).mean(axis=0))
    E_M_2 = np.array((features).mean(axis=0))**2
    var = (E_M2-E_M_2)[0]

    # plt.plot(range(100), [np.percentile(var, i) for i in range(100)])
    features_to_keep = (var >= np.percentile(var, percentile))
    features = features[:, features_to_keep]

    return pd.concat((node_information_df,pd.DataFrame(index = node_information_df.index, data = features.toarray(),columns = [metric+"_"+str(i) for i in range(features.shape[1])])),axis=1)