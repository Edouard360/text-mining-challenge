import pandas as pd
from preprocessing.abstractToGraphFeatures.SimilarityFeatureExtractor import SimilarityFeatureExtractor
from preprocessing.originalFeatures.OriginalFeatureExtractor import OriginalFeatureExtractor
from preprocessing.inOutFeatures.InOutFeatureExtractor import InOutFeatureExtractor
# SimilarityFeatureExtractor=[]
# OriginalFeatureExtractor=[]
# InOutFeatureExtractor=[]

class FeatureExporter:
    available_features = {
        "original":{
            "columns":["overlap_title", "temp_diff", "comm_auth"],
            "path":"originalFeatures/",
            "extractor":OriginalFeatureExtractor,
            "default_args":{}
        },
        "inOutDegree":{
            "columns":["indegree","outdegree"],
            "path":"inOutFeatures/",
            "extractor": InOutFeatureExtractor,
            "default_args":{}
        },
        "similarity":{
            "columns":["similarity"],
            "path":"abstractToGraphFeatures/",
            "extractor": SimilarityFeatureExtractor,
            "default_args": {"metric":"degrees", "percentile" : 0.95}
        }
    }
    def __init__(self, verbose=False):
        self.verbose = verbose

    def exportAllTo(self,df,node_information_df,filename):
        for key,value in FeatureExporter.available_features.items():
            self.computeFeature(df, node_information_df, key,**(value["default_args"]))
            self.exportTo(filename)

    def computeFeature(self,df,node_information_df,feature,**kargs):
        keys = FeatureExporter.available_features.keys()
        assert feature in keys,"Choose among those features :"+str(keys)
        self.current_feature = FeatureExporter.available_features[feature]
        extractor = self.current_feature["extractor"](node_information_df, **kargs)
        self.feature = extractor.extractFeature(df,verbose = self.verbose)

    def exportTo(self,filename):
        self.feature = pd.DataFrame(self.feature)
        self.feature.columns = self.current_feature["columns"]
        self.feature.to_csv("preprocessing/"+self.current_feature["path"]+"output/"+filename)


'''
def originalFeatures(df, node_information_df, verbose=False):

    :param df: The train_df or test_df with fields "source" and "target"
    :param node_information_df: The information on which to build the features
    :return: nd-array of shape (df.shape[0], n_features)

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
        if verbose and (counter % 1000 == 1):
            print(counter, " examples processed")

    return np.array([overlap_title, temp_diff, comm_auth]).T

def inOutDegreeFeatures(df, node_information_df, node_degree_df,verbose=False):
    counter = 0
    indegree = []
    outdegree = []
    id_to_index = dict(zip(node_information_df.index.values, range(node_information_df.index.size)))

    for source, target in zip(df["source"], df["target"]):
        indegree.append(node_degree_df[id_to_index[source],0])
        outdegree.append(node_degree_df[id_to_index[source],1])
        counter += 1
        if verbose and (counter % 1000 == 1):
            print(counter, " examples processed")

    return np.array([indegree,outdegree]).T

def graphMetricFeatures(metric ="degrees", percentile = 95): #node_information_df
    loader = np.load("abstractToGraphFeatures/metrics/"+metric+".npz")
    features = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                          shape=loader['shape'])
    E_M2 = np.array((features).multiply(features).mean(axis=0))
    E_M_2 = np.array((features).mean(axis=0))**2
    var = (E_M2-E_M_2)[0]

    # plt.plot(range(100), [np.percentile(var, i) for i in range(100)])
    features_to_keep = (var >= np.percentile(var, percentile))
    features = features[:, features_to_keep]
    return features.toarray()

def similarityFeatures(df, node_information_df, verbose=False):
    counter = 0
    similarity = []
    features = graphMetricFeatures()
    id_to_index = dict(zip(node_information_df.index.values, range(node_information_df.index.size)))

    for source, target in zip(df["source"], df["target"]):
        similarity.append(cosine_similarity(features[id_to_index[source], :].reshape(1, -1),
                                            features[id_to_index[target], :].reshape(1, -1))[0, 0])
        counter += 1
        if verbose and (counter % 1000 == 1):
            print(counter, " examples processed")

    return np.array([similarity]).T

'''