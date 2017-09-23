[![Build Status](https://travis-ci.org/Edouard360/text-mining-challenge.svg?branch=master)](https://travis-ci.org/Edouard360/text-mining-challenge)
[![Coverage Status](https://coveralls.io/repos/github/Edouard360/text-mining-challenge/badge.svg?branch=master)](https://coveralls.io/github/Edouard360/text-mining-challenge?branch=master)

# text-mining-challenge
A text mining challenge from course INF582.  
Please check out the [github version of our repository][link] for a tidier display of this README.md file !

[link]: <https://github.com/Edouard360/text-mining-challenge>

## Code structure 

#### Directories
The structure of the directories should be kept as it is in our project:

 - data/
 	- *.csv (put the node_information.csv file here)
 	- *.txt (testing_set.txt and training_set.txt)
 - featureEngineering/
 	- abstractFeatures/
 	- graphArticleFeatures/
 	- graphAuthorsFeatures/
 	- journalFeatures/
 	- lsaFeatures/
 	- originalFeatures/
 - submissions/
    - *.csv (all the submission.csv files will be exported here)
 - report/
 
Note that in each folder Features/ a folder output/ should be included.

## Feature Engineering

We compared each features individually on the same cross validation training-testing-set, 
using the same regressor: RandomForestRegressor with random_state set to 42. 

Bear in mind that these are **feature sets**, and not a single feature. For instance, the graphAuthors feature set is composed of
7 features ("meanACiteB_col", "maxACiteB_col","AOut_col", "BIn_col","ACiteAMean_col", "ACiteASum_col","BOut_col")


|Feature Set|Individual F1 score|
|---|:---:|
|'lsa' |0.576185|
|'original' |0.811745|
|'graphAuthors' |0.879161|
|'graphArticles' |0.992431|
|'journal' |0.611100|
|'similarity' |0.778112|

## Model tuning and comparison 
 
We compared these classifiers, and obtained the respective performance

|Algorithm|F1 score|
|---|:---:|
|Gradient Boosting|0.9|
|Random Forest Regressor|0.8|
|Logistic Regression|0.9|

To prevent overfitting, we did cross validation on a sample of the training set of approximately the same size
For each classifier, explain the procedure that was followed to tackle parameter tuning and prevent overfitting.
