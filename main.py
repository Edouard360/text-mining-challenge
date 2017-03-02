import pandas as pd
from sklearn import svm
from feature_extractor import featuresFromDataFrame
from tools import random_sample

train_df = pd.read_csv("training_set.txt", sep=" ",header=None)
train_df.columns = ["source","target","label"]
train_df = random_sample(train_df)

test_df = pd.read_csv("testing_set.txt", sep=" ",header=None)
test_df.columns = ["source","target"]
#test_df = random_sample(test_df,prop = 1)

node_information_df = pd.read_csv("node_information.csv", sep=",",header=None)
node_information_df.columns = ["ID","year","title","authors","journalName","abstract"]
node_information_df = node_information_df.reset_index().set_index("ID")
node_information_df["authors"].fillna("",inplace=True)

training_features = featuresFromDataFrame(train_df,node_information_df)
testing_features = featuresFromDataFrame(test_df,node_information_df)
labels = train_df["label"].values

classifier = svm.LinearSVC()
classifier.fit(training_features, labels)
labels_pred = classifier.predict(testing_features)

prediction_df = pd.DataFrame(columns=["id","category"],dtype=int)
prediction_df["id"] = range(len(labels_pred))
prediction_df["category"] = labels_pred

prediction_df.to_csv("improved_predictions.csv",index=None)