import pandas as pd
from sklearn import svm
from tools import random_sample
from sklearn import metrics
from preprocessing.FeatureExporter import FeatureExporter
from preprocessing.FeatureImporter import FeatureImporter
from sklearn.linear_model.logistic import LogisticRegression

#time_sub =

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

testing_on_train = True
features = ["commonNeighbours","original","inOutDegree","similarity"]


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
    # exporter.computeFeature(value["df"], node_information_df, "similarity", percentile=0.97)
    # exporter.exportTo(value["filename"])
    if not FeatureImporter.check(value["filename"],features=features):
        for feature in features:
            exporter = FeatureExporter(True)
            print("Exporting for "+key+", the feature "+feature)
            if not FeatureImporter.check(value["filename"],features=[feature]):
                exporter.computeFeature(value["df"],node_information_df,feature)
                exporter.exportTo(value["filename"])

training_features = FeatureImporter.importFromFile(df_dict["train"]["filename"], features=features)
testing_features = FeatureImporter.importFromFile(df_dict["test"]["filename"], features=features)

labels = df_dict["train"]["df"]["label"].values

classifier = svm.LinearSVC()
classifier = LogisticRegression()
classifier.fit(training_features, labels)
labels_pred = classifier.predict(testing_features)

if(testing_on_train):
    labels_true = df_dict["test"]["df"]["label"].values
    print("The Area Under Curve (AUC) is ",metrics.roc_auc_score(labels_true,labels_pred))
    print("The f1 score is ",metrics.f1_score(labels_true,labels_pred))
else:
    prediction_df = pd.DataFrame(columns=["id","category"],dtype=int)
    prediction_df["id"] = range(len(labels_pred))
    prediction_df["category"] = labels_pred
    prediction_df.to_csv("improved_predictions.csv",index=None)
