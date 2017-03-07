import numpy as np
import nltk
import igraph
import csv
import pandas as pd
from collections import defaultdict
stpwds = set(nltk.corpus.stopwords.words("english"))
stemmer = nltk.stem.PorterStemmer()

def build_graph(path=""):
    with open(path+"data/training_set.txt", "r") as f:
        reader = csv.reader(f)
        training_set  = list(reader)

    training_set = [element[0].split(" ") for element in training_set]

    with open(path+"data/node_information.csv", "r") as f:
        reader = csv.reader(f)
        node_info  = list(reader)

    IDs = [element[0] for element in node_info]
    edges = [(element[0],element[1]) for element in training_set if element[2]=="1"]
    ## some nodes may not be connected to any other node
    ## hence the need to create the nodes of the graph from node_info.csv,
    ## not just from the edge list
    nodes = IDs
    ## create empty directed graph
    g = igraph.Graph(directed=True)
    ## add vertices
    g.add_vertices(nodes)
    ## add edges
    g.add_edges(edges)
    return g

def build_authors_graph(path=""):
    with open(path+"data/training_set.txt", "r") as f:
        reader = csv.reader(f)
        training_set  = list(reader)

    training_set = [element[0].split(" ") for element in training_set]
    node_information_df = pd.read_csv("data/node_information.csv",header = None)

    node_information_df.columns = ["ID", "year", "title", "authors", "journalName", "abstract"]
    node_information_df = node_information_df.reset_index().set_index("ID")
    node_information_df["authors"].fillna("", inplace=True)
    authors = node_information_df["authors"].values.tolist()
    authors = [author_list.split(", ") for author_list in authors]
    authors = [list(filter(None, author_list)) for author_list in authors]
    concatenated_authors = np.concatenate(tuple(authors))
    unique_authors = list(set(concatenated_authors))

    igraph.Graph()
    authors_to_index = dict(zip(unique_authors, range(len(unique_authors))))
    id_to_index = dict(zip(node_information_df.index.values, range(node_information_df.index.size)))

    dict_edges = defaultdict(int)
    for element in training_set:
        if element[2] == "1":
            for author_source in authors[id_to_index[int(element[0])]]:
                for author_target in authors[id_to_index[int(element[1])]]:
                    dict_edges[(author_source,author_target)]+=1

    '''a = list(dict_edges.keys())
    b = [(y,x) for x,y in a]
    tmp = list(set(a) & set(b))
    authors_graph = igraph.Graph(directed=True)
    authors_graph.add_vertices(unique_authors)
    for source, target in dict_edges.keys():
        authors_graph.add_edge(source,target,weight = dict_edges[(source,target)])
    assert authors_graph.is_weighted(), "The graph is not weighted"
    '''
    return dict_edges

def remove_stopwords_and_stem(words):
    words = [token for token in words if (len(token)>2 and (token not in stpwds))]
    return [stemmer.stem(token) for token in words]

def random_sample(df, p = 0.05,seed=42):
    '''
    Randomly samples a proportion 'p' of rows of a dataframe
    '''
    size = df.shape[0]
    np.random.seed(seed)
    return df.ix[np.random.randint(0, size, int(size*p)), :]

def stats_df(df):
    '''
    Gives stats about the dataframe
    '''
    print("Nb lines in the train : ",len(df["from"]))
    print("Nb of unique nodes : ",len(df["from"].unique()))
    print("The document that cites the most, cites : ",df.groupby(["from"]).sum()["y"].max()," document(s).")
    print("The document with no citation : ",sum(df.groupby(["from"]).sum()["y"] == 0),"\n")
    print("The most cited document, is cited : ",df.groupby(["to"]).sum()["y"].max()," times.")
    print("Nb of documents never cited  : ",sum(df.groupby(["to"]).sum()["y"] == 0),"\n")
    print("There are NaN to handle for authors and journalName :")
