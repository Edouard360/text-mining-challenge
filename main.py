from time import gmtime, strftime
import pandas as pd
from classifier import Classifier
from sklearn import svm
from tools import random_sample
from sklearn import metrics

from preprocessing.FeatureExporter import FeatureExporter
from preprocessing.FeatureImporter import FeatureImporter
from sklearn.linear_model.logistic import LogisticRegression

time_sub = strftime("%Y-%m-%d %H:%M:%S", gmtime()).replace(' ','__')

train_df = pd.read_csv("data/training_set.txt", sep=" ", header=None)
train_df.columns = ["source","target","label"]

test_df = pd.read_csv("data/testing_set.txt", sep=" ", header=None)
test_df.columns = ["source","target"]

node_information_df = pd.read_csv("data/node_information.csv", sep=",", header=None)
node_information_df.columns = ["ID","year","title","authors","journalName","abstract"]
node_information_df = node_information_df.reset_index().set_index("ID")
node_information_df["authors"].fillna("", inplace=True)

df_dict = dict()
training_set_percentage = 0.05

df_dict["train"] = {
    "filename": 'training_set.txt',
    "df": random_sample(train_df, p=training_set_percentage)
}

testing_on_train = True
# features = ["commonNeighbours","original","inOutDegree","similarity"]
# features = ["commonNeighbours","tfidf","original","inOutDegree","similarity"]
# features = ["original","inOutDegree","similarity"]
features = ["tfidf","original","inOutDegree","similarity"]
# By uncommenting you can tune in the parameters
parameters = {}
# parameters = {"percentile":95,"metric":"w_degrees"}

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
    if not FeatureImporter.check(value["filename"],features=features,**parameters):
        for feature in features:
            exporter = FeatureExporter(True)
            if not FeatureImporter.check(value["filename"],features=[feature],**parameters):
                print("Exporting for " + key + " the feature " + feature)
                exporter.computeFeature(value["df"],node_information_df,feature,**parameters)
                exporter.exportTo(value["filename"],feature,**parameters)

training_features = FeatureImporter.importFromFile(df_dict["train"]["filename"], features=features,**parameters)
testing_features = FeatureImporter.importFromFile(df_dict["test"]["filename"], features=features,**parameters)

labels = df_dict["train"]["df"]["label"].values

classifier = Classifier()
# classifier = LogisticRegression()
classifier.fit(training_features, labels)
labels_pred = classifier.predict(testing_features)

if(testing_on_train):
    labels_true = df_dict["test"]["df"]["label"].values
    print ("Features : ", features)
    if hasattr(classifier, 'name'):
        print ("Classifier : ", classifier.name)
    else:
        print ("Classifier : ", str(classifier))
    print("AUC is %f | %.2f  of training set" % (metrics.roc_auc_score(labels_true,labels_pred), training_set_percentage))
    print("f1 score is %f | %.2f  of training set" % (metrics.f1_score(labels_true,labels_pred), training_set_percentage))
else:
    prediction_df = pd.DataFrame(columns=["id","category"],dtype=int)
    prediction_df["id"] = range(len(labels_pred))
    prediction_df["category"] = labels_pred
    prediction_df.to_csv("submissions/improved_predictions_of_"+time_sub+".csv",index=None)
