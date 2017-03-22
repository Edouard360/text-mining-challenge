import numpy as np
import nltk
import igraph
import csv
import pandas as pd
from collections import defaultdict
from itertools import combinations
from sklearn.metrics import f1_score
from math import log

# Build a cost dictionary, assuming Zipf's law and cost = -math.log(probability).
# words = open("data/words-by-frequency.txt").read().split()
# wordcost = dict((k, log((i + 1) * log(len(words)))) for i, k in enumerate(words))
# maxword = max(len(x) for x in words)

stpwds = set(nltk.corpus.stopwords.words("english"))
stemmer = nltk.stem.PorterStemmer()


def articles_graph(path=""):
    with open(path + "data/training_set.txt", "r") as f:
        reader = csv.reader(f)
        training_set = list(reader)
    training_set = [element[0].split(" ") for element in training_set]
    node_id_df = pd.read_csv("data/node_information.csv", header=None, usecols=[0]).values.reshape(-1)

    g = igraph.Graph(directed=True)

    g["articles_to_index"] = dict(zip(node_id_df, range(len(node_id_df))))
    g.add_vertices([i for i in range(len(node_id_df))])
    edges = []
    for element in training_set:
        if element[2] == "1":
            edges.append((g["articles_to_index"][int(element[0])], g["articles_to_index"][int(element[1])]))
    g.add_edges(edges)
    return g


def journals_citation_graph(path=""):
    with open(path + "data/training_set.txt", "r") as f:
        reader = csv.reader(f)
        training_set = list(reader)

    training_set = [element[0].split(" ") for element in training_set]
    node_information_df = pd.read_csv("data/node_information.csv", header=None)

    node_information_df.columns = ["ID", "year", "title", "authors", "journalName", "abstract"]
    node_information_df = node_information_df.reset_index().set_index("ID")
    node_information_df["journalName"].fillna("", inplace=True)

    journals = node_information_df["journalName"].values.tolist()
    unique_journals = list(set(journals))

    journals_sep = [journal.split(".") for journal in journals]
    journals_sep = [list(filter(None, journal)) for journal in journals_sep]
    concatenated_journals_sep = np.concatenate(tuple(journals_sep))
    unique_journals_sep = list(set(concatenated_journals_sep))

    g = igraph.Graph(directed=True)
    g_sep = igraph.Graph(directed=True)

    g.add_vertices([i for i in range(len(unique_journals))])
    g.vs["weight"] = np.zeros(len(unique_journals))
    g["journals_to_index"] = dict(zip(unique_journals, range(len(unique_journals))))

    g_sep.add_vertices([i for i in range(len(unique_journals_sep))])
    g_sep.vs["weight"] = np.zeros(len(unique_journals_sep))
    g_sep["journals_sep_to_index"] = dict(zip(unique_journals_sep, range(len(unique_journals_sep))))

    id_to_index = dict(zip(node_information_df.index.values, range(node_information_df.index.size)))

    edges = []
    edges_sep = []
    for element in training_set:
        if element[2] == "1":
            journal_source = g["journals_to_index"][journals[id_to_index[int(element[0])]]]
            journal_target = g["journals_to_index"][journals[id_to_index[int(element[1])]]]
            edges.append((journal_source, journal_target))
            for journal_sep_source in journals_sep[id_to_index[int(element[0])]]:
                for journal_sep_target in journals_sep[id_to_index[int(element[1])]]:
                    if (journal_sep_source != journal_sep_target):
                        edges_sep.append((g_sep["journals_sep_to_index"][journal_sep_source],
                                          g_sep["journals_sep_to_index"][journal_sep_target]))
                    else:
                        g_sep.vs[g_sep["journals_sep_to_index"][journal_sep_source]]["weight"] += 1

    g.add_edges(edges)
    g.es["weight"] = np.ones(len(edges))
    g = g.simplify(combine_edges='sum')

    g_sep.add_edges(edges_sep)
    g_sep.es["weight"] = np.ones(len(edges_sep))
    g_sep = g_sep.simplify(combine_edges='sum')
    return g, g_sep


def authors_citation_dict(path=""):
    with open(path + "data/training_set.txt", "r") as f:
        reader = csv.reader(f)
        training_set = list(reader)

    training_set = [element[0].split(" ") for element in training_set]
    node_information_df = pd.read_csv("data/node_information.csv", header=None)

    node_information_df.columns = ["ID", "year", "title", "authors", "journalName", "abstract"]
    node_information_df = node_information_df.reset_index().set_index("ID")
    node_information_df["authors"].fillna("", inplace=True)
    authors = node_information_df["authors"].values.tolist()
    authors = [author_list.split(", ") for author_list in authors]
    authors = [list(filter(None, author_list)) for author_list in authors]

    id_to_index = dict(zip(node_information_df.index.values, range(node_information_df.index.size)))

    dict_edges = defaultdict(int)
    for element in training_set:
        if element[2] == "1":
            for author_source in authors[id_to_index[int(element[0])]]:
                for author_target in authors[id_to_index[int(element[1])]]:
                    dict_edges[(author_source, author_target)] += 1

    return dict_edges


def authors_citation_graph(path=""):
    with open(path + "data/training_set.txt", "r") as f:
        reader = csv.reader(f)
        training_set = list(reader)

    training_set = [element[0].split(" ") for element in training_set]
    node_information_df = pd.read_csv("data/node_information.csv", header=None)

    node_information_df.columns = ["ID", "year", "title", "authors", "journalName", "abstract"]
    node_information_df = node_information_df.reset_index().set_index("ID")
    node_information_df["authors"].fillna("", inplace=True)
    authors = node_information_df["authors"].values.tolist()
    authors = [author_list.split(", ") for author_list in authors]
    authors = [list(filter(None, author_list)) for author_list in authors]
    concatenated_authors = np.concatenate(tuple(authors))
    unique_authors = list(set(concatenated_authors))

    g = igraph.Graph(directed=True)
    g.add_vertices([i for i in range(len(unique_authors))])
    g.vs["weight"] = np.zeros(len(unique_authors))
    g["authors_to_index"] = dict(zip(unique_authors, range(len(unique_authors))))

    id_to_index = dict(zip(node_information_df.index.values, range(node_information_df.index.size)))

    edges = []
    for element in training_set:
        if element[2] == "1":
            for author_source in authors[id_to_index[int(element[0])]]:
                for author_target in authors[id_to_index[int(element[1])]]:
                    if (author_source != author_target):
                        edges.append((g["authors_to_index"][author_source], g["authors_to_index"][author_target]))
                    else:
                        g.vs[g["authors_to_index"][author_source]]["weight"] += 1

    g.add_edges(edges)
    g.es["weight"] = np.ones(len(edges))
    g = g.simplify(combine_edges='sum')
    return g


def authors_collaboration_graph():
    node_information_df = pd.read_csv("data/node_information.csv", header=None)

    node_information_df.columns = ["ID", "year", "title", "authors", "journalName", "abstract"]
    node_information_df = node_information_df.reset_index().set_index("ID")
    node_information_df["authors"].fillna("", inplace=True)
    authors = node_information_df["authors"].values.tolist()
    authors = [author_list.split(", ") for author_list in authors]
    authors = [list(filter(None, author_list)) for author_list in authors]
    concatenated_authors = np.concatenate(tuple(authors))
    unique_authors = list(set(concatenated_authors))

    g = igraph.Graph(directed=False)
    g.add_vertices([i for i in range(len(unique_authors))])
    g["authors_to_index"] = dict(zip(unique_authors, range(len(unique_authors))))
    authors_list_ids = [[g["authors_to_index"][author] for author in author_list] for author_list in authors]
    edges = []
    for author_list_id in authors_list_ids:
        edges += list(combinations(author_list_id, 2))

    g.add_edges(edges)
    g.es["weight"] = np.ones(len(edges))
    g = g.simplify(combine_edges='sum')
    return g


def remove_stopwords_and_stem(words, split_more=False):
    words = [token for token in words if (len(token) > 2 and (token not in stpwds))]
    if split_more:
        more = []
        for word in words:
            split_word = infer_spaces(word)
            if (len(split_word) > 1):
                more += split_word
            more = [w for w in more if len(w) > 3]
            print(more)
        words += more
    return [stemmer.stem(token) for token in words]


def random_sample(df, p=0.05, seed=42):
    '''
    Randomly samples a proportion 'p' of rows of a dataframe
    '''
    size = df.shape[0]
    np.random.seed(seed)
    return df.ix[np.random.randint(0, size, int(size * p)), :]


def stats_df(df):
    '''
    Gives stats about the dataframe
    '''
    print("Nb lines in the train : ", len(df["from"]))
    print("Nb of unique nodes : ", len(df["from"].unique()))
    print("The document that cites the most, cites : ", df.groupby(["from"]).sum()["y"].max(), " document(s).")
    print("The document with no citation : ", sum(df.groupby(["from"]).sum()["y"] == 0), "\n")
    print("The most cited document, is cited : ", df.groupby(["to"]).sum()["y"].max(), " times.")
    print("Nb of documents never cited  : ", sum(df.groupby(["to"]).sum()["y"] == 0), "\n")
    print("There are NaN to handle for authors and journalName :")


def xgb_f1(y, t):
    '''
    :param y: true labels
    :param t: predicted labels
    :return: f1 score
    '''
    # t = t.get_label()
    y_bin = [1. if y_cont > 0.5 else 0. for y_cont in y]  # binaryzing your output
    return 'f1', f1_score(t, y_bin)


def infer_spaces(s):
    """Uses dynamic programming to infer the location of spaces in a string
    without spaces."""

    # Find the best match for the i first characters, assuming cost has
    # been built for the i-1 first characters.
    # Returns a pair (match_cost, match_length).
    def best_match(i):
        candidates = enumerate(reversed(cost[max(0, i - maxword):i]))
        return min((c + wordcost.get(s[i - k - 1:i], 9e999), k + 1) for k, c in candidates)

    # Build the cost array.
    cost = [0]
    for i in range(1, len(s) + 1):
        c, k = best_match(i)
        cost.append(c)

    # Backtrack to recover the minimal-cost string.
    out = []
    i = len(s)
    while i > 0:
        c, k = best_match(i)
        assert c == cost[i]
        out.append(s[i - k:i])
        i -= k

    return list(reversed(out))
