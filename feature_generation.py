
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from sklearn import svm
from feature_extractor import featuresFromDataFrame
from tools import random_sample
import igraph


train_df = pd.read_csv("training_set.txt", sep=" ",header=None)
train_df.columns = ["source","target","label"]
train_df = random_sample(train_df)

test_df = pd.read_csv("testing_set.txt", sep=" ",header=None)
test_df.columns = ["source","target"]
#test_df = random_sample(test_df,prop = 1)

node_information_df = pd.read_csv("node_information.csv", sep=",", header=None)
node_information_df.columns = ["ID","year","title","authors","journalName","abstract"]
node_information_df = node_information_df.reset_index().set_index("ID")
node_information_df["authors"].fillna("", inplace=True)


# In[2]:

train_df.shape
train_df.head()
test_df.head()
node_information_df.head()


# In[6]:

training_features = featuresFromDataFrame(train_df,node_information_df)
testing_features = featuresFromDataFrame(test_df,node_information_df)
labels = train_df["label"].values


# In[2]:

from tools import build_graph
g = build_graph()


# In[3]:

from feature_extractor import InOutDegree
train_df = InOutDegree(train_df, g)
test_df = InOutDegree(test_df, g)


# In[7]:

train_df.to_csv('tmp/train_df')
test_df.to_csv('tmp/test_df')


# In[12]:

training_features = np.concatenate((training_features, train_df[['target_in_degree', 'target_out_degree']].values), 
                                   axis=1)
testing_features = np.concatenate((testing_features, test_df[['target_in_degree', 'target_out_degree']].values), 
                                   axis=1)


# In[19]:

classifier = svm.LinearSVC()
classifier.fit(training_features, labels)
labels_pred = classifier.predict(testing_features)

prediction_df = pd.DataFrame(columns=["id","category"], dtype=int)
prediction_df["id"] = range(len(labels_pred))
prediction_df["category"] = labels_pred

prediction_df.to_csv("submissions/improved_predictionsinoutdegree.csv", index=None)

