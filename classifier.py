from sklearn.base import BaseEstimator
import xgboost as xgb
import numpy as np

class Classifier(BaseEstimator):
    def __init__(self):
        self.name = "XGBClassifier"
        self.n_estimators = 100
        self.max_depth = 10
        self.clf = xgb.XGBClassifier(n_estimators=self.n_estimators,
                                    max_depth=self.max_depth,
                                    learning_rate=0.1,
                                    silent=1,
                                    objective='binary:logistic',
                                    nthread=1,
                                    gamma=0.001,
                                    min_child_weight=1,
                                    max_delta_step=0,
                                    subsample=1,
                                    colsample_bytree=1,
                                    colsample_bylevel=1,
                                    reg_alpha=0,
                                    reg_lambda=1,
                                    scale_pos_weight=1,
                                    base_score=0.508,
                                    seed=0,
                                    missing=None)
    def fit(self, X, y):
        self.clf.fit(X, y)
    def predict(self, X):
        return self.clf.predict(X)
    def predict_proba(self, X):
        return self.clf.predict_proba(X)
