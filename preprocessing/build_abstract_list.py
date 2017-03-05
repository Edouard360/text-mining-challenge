import numpy as np
import pandas as pd
import nltk
import csv
from scipy import sparse
from library_moodle import terms_to_graph,compute_node_centrality
from tools import remove_stopwords_and_stem

stpwds = set(nltk.corpus.stopwords.words("english"))
stemmer = nltk.stem.PorterStemmer()

node_information_df = pd.read_csv("data/node_information.csv", sep=",",header=None)
node_information_df.columns = ["ID","year","title","authors","journalName","abstract"]
node_information_df = node_information_df.reset_index().set_index("ID")

try:
    with open('abstract_list.csv', 'r') as f:
        abstract_list = []
        for line in f:
            abstract_list = abstract_list + [line[:-1].split(",")]
except FileNotFoundError:
    with open("abstract_list.csv", "w") as f:
        abstract_list = node_information_df["abstract"].values
        abstract_list = [remove_stopwords_and_stem(abstract.split(" ")) for abstract in abstract_list]
        writer = csv.writer(f)
        writer.writerows(abstract_list)

# All the unique  words
concatenated_abstracts = np.concatenate(tuple(abstract_list))
unique_words = list(set(concatenated_abstracts))
index_dict = dict(zip(unique_words,range(len(unique_words))))

window_size = 3
graphs = [terms_to_graph(abstract,min(window_size,len(abstract))) for abstract in abstract_list]

# Keys are degrees, w_degrees, closeness, w_closeness
features_dict = dict()
metrics = ["degrees","w_degrees","closeness","w_closeness"]
for m in metrics:
    features_dict[m] = sparse.lil_matrix((len(abstract_list),len(unique_words)))

for i,graph in enumerate(graphs):
    for row in compute_node_centrality(graph):
        col_index = index_dict[row[0]]
        for index_m, m in enumerate(metrics):
            features_dict[m][i,col_index]= row[index_m+1]

for m in metrics:
    csr_matrix = features_dict[m].tocsr()
    np.savez("node-abstract/node-feature-"+m, data=csr_matrix.data, indices=csr_matrix.indices,
             indptr=csr_matrix.indptr, shape=csr_matrix.shape)
