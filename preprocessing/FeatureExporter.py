import pandas as pd
from preprocessing.abstractToGraphFeatures.SimilarityFeatureExtractor import SimilarityFeatureExtractor
from preprocessing.originalFeatures.OriginalFeatureExtractor import OriginalFeatureExtractor
from preprocessing.inOutFeatures.InOutFeatureExtractor import InOutFeatureExtractor
from preprocessing.commonNeighboursFeatures.commonNeighboursExtractor import CommonNeighboursFeatureExtractor
from preprocessing.intersectionFeatures.IntersectionFeatureExtractor import IntersectionFeatureExtractor

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
        },
        "intersection":{
            "columns":["intersection"],
            "path":"intersectionFeatures/",
            "extractor": IntersectionFeatureExtractor,
            "default_args": {}
        },
        "commonNeighbours":{
            "columns":["commonNeighbours"],
            "path":"commonNeighboursFeatures/",
            "extractor": CommonNeighboursFeatureExtractor,
            "default_args": {}
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
