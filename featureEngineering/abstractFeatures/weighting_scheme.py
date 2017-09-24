import numpy as np
from scipy import sparse


def computeTfidf(abstract_list, index_dict, unique_words):
    print("Now computing tf-idf metric")
    idf = np.zeros(len(unique_words))
    tf = sparse.lil_matrix((len(abstract_list), len(unique_words)))
    count = 1
    for i, abstract in enumerate(abstract_list):
        for word in set(abstract):
            idf[index_dict[word]] += 1
        for word in abstract:
            tf[i, index_dict[word]] += 1
        if (count % 10000 == 1):
            print(count, "abstracts treated")
        count += 1
    return tf, idf


def weightingScheme(tf, idf, doc_len, tf_scheme="BM25"):
    assert tf_scheme in ["classic", "BM25", "pl"], "Not a valid scheme"
    idf_n = np.log10(len(doc_len) / idf)
    tf_n = tf.copy()
    K = 1.2;
    b = 0.75
    if tf_scheme == "classic":
        tf_n[tf_n.nonzero()] = 1 + np.log10(tf_n[tf_n.nonzero()].toarray())
    elif tf_scheme == "BM25":
        tf_n = sparse.csr_matrix((K + 1) * (tf_n.toarray()) / \
                                 (np.reshape(K * (1 - b + b * doc_len / np.mean(doc_len)), (-1, 1)) + tf_n.toarray()))
    elif tf_scheme == "pl":
        tf_n[tf_n.nonzero()] = 1 + np.log10(1 + np.log10(tf_n[tf_n.nonzero()].toarray()))
        composition1 = sparse.diags(1 / (1 - b + b * doc_len / np.mean(doc_len)))
        tf_n = (composition1 @ tf_n).tocsr()
    return (tf_n @ (sparse.diags(idf_n))).tocsr()


def keepHighVarianceFeatures(features, percentile=95):
    E_M2 = np.array((features).multiply(features).mean(axis=0))
    E_M_2 = np.array((features).mean(axis=0)) ** 2
    var = (E_M2 - E_M_2)[0]
    var[np.isnan(var)] = 0
    return features[:, (var >= np.percentile(var, percentile))].toarray()
