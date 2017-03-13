from time import localtime, strftime
import pandas as pd
from classifier import Classifier
from tools import random_sample
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from preprocessing.FeatureExporter import FeatureExporter
from preprocessing.FeatureImporter import FeatureImporter
from sklearn.linear_model.logistic import LogisticRegression

time_sub = strftime("%Y-%m-%d %H:%M:%S", localtime()).replace(' ', '__')

train_df = pd.read_csv("data/training_set.txt", sep=" ", header=None)
train_df.columns = ["source", "target", "label"]

test_df = pd.read_csv("data/testing_set.txt", sep=" ", header=None)
test_df.columns = ["source", "target"]

node_information_df = pd.read_csv("data/node_information.csv", sep=",", header=None)
node_information_df.columns = ["ID", "year", "title", "authors", "journalName", "abstract"]
node_information_df = node_information_df.reset_index().set_index("ID")
node_information_df["authors"].fillna("", inplace=True)
node_information_df["journalName"].fillna("", inplace=True)

df_dict = dict()
training_set_percentage = 0.05

df_dict["train"] = {
    "filename": 'training_set.txt',
    "df": random_sample(train_df, p=training_set_percentage)
}

testing_on_train = True
early_stopping = False
# features = ["authors", "commonNeighbours", 'original', "inOutDegree", "similarity", "authors"]
features = ["original"]
verbose = True
freq = 5000

# By uncommenting you can tune in the parameters
parameters = {}
# parameters = {"percentile":95,"metric":"degrees"}

if testing_on_train:
    df_dict["test"] = {
        "filename": 'testing_training_set.txt',
        "df": random_sample(train_df, p=0.05, seed=43)
    }
else:
    df_dict["test"] = {
        "filename": 'testing_set.txt',
        "df": test_df
    }

exporter = FeatureExporter(verbose=verbose, freq=freq)
for key, value in df_dict.items():
    if not FeatureImporter.check(value["filename"], features=features, **parameters):
        for feature in features:
            if not FeatureImporter.check(value["filename"], features=[feature], **parameters):
                print("Exporting for " + key + " the feature " + feature)
                exporter.computeFeature(value["df"], node_information_df, feature, **parameters)
                exporter.exportTo(value["filename"], feature, **parameters)

training_features = FeatureImporter.importFromFile(df_dict["train"]["filename"], features=features, **parameters)
# training_features_1 = training_features[:,0:2].reshape(-1,2)
# training_features_2 = training_features[:,3:]
# training_features = np.concatenate((training_features_1,training_features_2),axis = 1)

testing_features = FeatureImporter.importFromFile(df_dict["test"]["filename"], features=features, **parameters)
# testing_features_1 = testing_features[:,0:2].reshape(-1,2)
# testing_features_2 = testing_features[:,3:]
# testing_features = np.concatenate((testing_features_1,testing_features_2),axis = 1)

labels = df_dict["train"]["df"]["label"].values

classifier = Classifier()
# classifier = LogisticRegression()
# classifier = RandomForestClassifier(n_estimators=100)

if testing_on_train:
    labels_true = df_dict["test"]["df"]["label"].values
    if not early_stopping:
        classifier.fit(training_features, labels)
        labels_pred = classifier.predict(testing_features)
        print("Features : ", features)
        if hasattr(classifier, 'name'):
            print("Classifier : ", classifier.name)
        else:
            print("Classifier : ", str(classifier))
        print("f1 score is %f | %.2f  of training set" % (
            metrics.f1_score(labels_true, labels_pred), training_set_percentage))
    else:
        plot_curves = False
        eval_set = [(training_features, labels),
                    (testing_features, labels_true)]
        if plot_curves:
            classifier.plotlearningcurves(eval_set)
        else:
            classifier.early_stop(eval_set)
else:
    classifier.fit(training_features, labels)
    labels_pred = classifier.predict(testing_features)
    prediction_df = pd.DataFrame(columns=["id", "category"], dtype=int)
    prediction_df["id"] = range(len(labels_pred))
    prediction_df["category"] = labels_pred
    prediction_df.to_csv("submissions/improved_predictions_of_" + time_sub + ".csv", index=None)
