import pandas as pd
from sklearn import svm
from tools import random_sample
from sklearn import metrics
from preprocessing.FeatureExporter import FeatureExporter
from preprocessing.FeatureImporter import FeatureImporter

train_df = pd.read_csv("data/training_set.txt", sep=" ", header=None)
train_df.columns = ["source","target","label"]

test_df = pd.read_csv("data/testing_set.txt", sep=" ", header=None)
test_df.columns = ["source","target"]

node_information_df = pd.read_csv("data/node_information.csv", sep=",", header=None)
node_information_df.columns = ["ID","year","title","authors","journalName","abstract"]
node_information_df = node_information_df.reset_index().set_index("ID")
node_information_df["authors"].fillna("", inplace=True)

df_dict = dict()

df_dict["train"] = {
    "filename": 'training_set.txt',
    "df": random_sample(train_df,p = 0.05)
}
node_degree_df = pd.read_csv("preprocessing/in-out/node_degree.csv", sep=",",header=None)
node_degree_df.columns = ["ID","target_indegree","target_outdegree"]
node_degree_df = node_degree_df.reset_index().set_index("ID")
features = ["inOutDegree","original","similarity"]

testing_on_train = True

if testing_on_train:
    df_dict["test"] = {
        "filename": 'testing_training_set.txt',
        "df": random_sample(train_df,p = 0.05,seed=43)
    }
else:
    df_dict["test"] ={
        "filename": 'testing_set.txt',
        "df": test_df
    }

for key,value in df_dict.items():
    if not FeatureImporter.check(value["filename"],features=features):
        exporter = FeatureExporter(True)
        for feature in features:
            print("Exporting for "+key+", the feature "+feature)
            if not FeatureImporter.check(value["filename"],features=[feature]):
                exporter.computeFeature(value["df"],node_information_df,feature)
                exporter.exportTo(value["filename"])

training_features = FeatureImporter.importFromFile(df_dict["train"]["filename"], features=features)
testing_features = FeatureImporter.importFromFile(df_dict["test"]["filename"], features=features)

labels = df_dict["train"]["df"]["label"].values

classifier = svm.LinearSVC()
classifier.fit(training_features, labels)
labels_pred = classifier.predict(testing_features)

if(testing_on_train):
    labels_true = df_dict["test"]["df"]["label"].values
    print("The Area Under Curve (AUC) is ",metrics.roc_auc_score(labels_true,labels_pred))
else:
    prediction_df = pd.DataFrame(columns=["id","category"],dtype=int)
    prediction_df["id"] = range(len(labels_pred))
    prediction_df["category"] = labels_pred
    prediction_df.to_csv("improved_predictions.csv",index=None)
