import numpy as np
import pandas as pd
import csv
from scipy import sparse
from featureEngineering.abstractFeatures.terms_to_graph import terms_to_graph, compute_node_centrality
from tools import remove_stopwords_and_stem
from featureEngineering.abstractFeatures.weighting_scheme import weightingScheme, computeTfidf


def getAbstractList():
    abstract_information_df = pd.read_csv("data/node_information.csv", sep=",", header=None, usecols=[0, 5])
    abstract_information_df.columns = ["ID", "abstract"]
    abstract_information_df = abstract_information_df.reset_index().set_index("ID")
    try:
        with open('featureEngineering/abstractFeatures/abstract_list.csv', 'r') as f:
            abstract_list = []
            for line in f:
                abstract_list = abstract_list + [line[:-1].split(",")]
    except IOError:
        with open("featureEngineering/abstractFeatures/abstract_list.csv", "w") as f:
            print("It seems abstract_list.csv has never been created.\nCreating abstract_list.csv")
            abstract_list = abstract_information_df["abstract"].values
            abstract_list = [remove_stopwords_and_stem(abstract.split(" ")) for abstract in abstract_list]
            abstract_list = [[w for w in abstract if (w.isalpha())] for abstract in abstract_list]
            writer = csv.writer(f)
            writer.writerows(abstract_list)
    return abstract_list


def abstractToGraph(path=""):
    abstract_list = getAbstractList()
    # All the unique  words
    concatenated_abstracts = np.concatenate(tuple(abstract_list))
    unique_words = list(set(concatenated_abstracts))
    index_dict = dict(zip(unique_words, range(len(unique_words))))
    doc_len = np.array([len(abstract) for abstract in abstract_list])

    window_size = 3
    print("Creating Graphs")
    graphs = [terms_to_graph(abstract, window_size) for abstract in abstract_list]
    print("Graphs created")
    # Keys are degrees, w_degrees, closeness, w_closeness
    features_dict = dict()
    metrics = ["degrees", "w_degrees", "closeness", "w_closeness"]
    for m in metrics:
        features_dict[m] = sparse.lil_matrix((len(abstract_list), len(unique_words)))

    print("Computing metrics")
    for i, graph in enumerate(graphs):
        for row in compute_node_centrality(graph):
            col_index = index_dict[row[0]]
            for index_m, m in enumerate(metrics):
                features_dict[m][i, col_index] = row[index_m + 1]

    print("Saving metrics to featureEngineering/abstractFeatures/metrics/ folder")
    _, idf = computeTfidf(abstract_list, index_dict, unique_words)
    for m in metrics:
        csr_matrix = features_dict[m].tocsr()
        csr_matrix = weightingScheme(csr_matrix, idf, doc_len)
        np.savez("featureEngineering/abstractFeatures/metrics/" + m, data=csr_matrix.data, indices=csr_matrix.indices,
                 indptr=csr_matrix.indptr, shape=csr_matrix.shape)


def tfIdfFeatures(path=""):
    abstract_list = getAbstractList()

    # All the unique  words
    concatenated_abstracts = np.concatenate(tuple(abstract_list))
    unique_words = list(set(concatenated_abstracts))
    index_dict = dict(zip(unique_words, range(len(unique_words))))
    doc_len = np.array([len(abstract) for abstract in abstract_list])

    tf, idf = computeTfidf(abstract_list, index_dict)
    csr_tfidf = weightingScheme(tf, idf, doc_len, tf_scheme="BM25")

    np.savez("featureEngineering/abstractFeatures/metrics/tfidf", data=csr_tfidf.data, indices=csr_tfidf.indices,
             indptr=csr_tfidf.indptr, shape=csr_tfidf.shape)
