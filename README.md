# text-mining-challenge
A text mining challenge from course INF582.

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

## Model tuning and comparison 
 
We compared these classifiers, and obtained the respective performance

- Gradient Boosting
- Random Forest Regressor
- Logistic Regression
|Algorithm|F1 score|
|---|:---:|
|Gradient Boosting|0.9|
|Random Forest Regressor|0.8|
|Logistic Regression|0.9|

To prevent overfitting, we did cross validation on a sample of the training set of approximately the same size
For each classifier, explain the procedure that was followed to tackle parameter tuning and prevent overfitting.