from sklearn.base import BaseEstimator
import xgboost as xgb
from tools import xgb_f1
from matplotlib import pyplot


class Classifier(BaseEstimator):
    def __init__(self):
        self.name = "XGBClassifier"
        self.n_estimators = 792
        self.max_depth = 10
        self.clf = xgb.XGBClassifier(n_estimators=self.n_estimators,
                                     max_depth=self.max_depth,
                                     learning_rate=0.1,
                                     silent=1,
                                     objective='binary:logistic',
                                     nthread=-1,
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

    def plotlearningcurves(self, eval_set):
        self.clf.fit(eval_set[0][0], eval_set[0][1], eval_metric=["logloss", "error"],
                     eval_set=eval_set, verbose=False)
        labels_true = df_dict["test"]["df"]["label"].values

        labels_pred = classifier.predict(testing_features)
        results = classifier.evals_result()
        epochs = len(results['validation_0']['error'])
        x_axis = range(0, epochs)
        # plot log loss
        fig, ax = pyplot.subplots()
        ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
        ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
        ax.legend()
        pyplot.ylabel('Log Loss')
        pyplot.title('XGBoost Log Loss')
        pyplot.show()
        # plot classification error
        fig, ax = pyplot.subplots()
        ax.plot(x_axis, results['validation_0']['error'], label='Train')
        ax.plot(x_axis, results['validation_1']['error'], label='Test')
        ax.legend()
        pyplot.ylabel('Classification Error')
        pyplot.title('XGBoost Classification Error')
        pyplot.show()
        # # plot f1 score
        # fig, ax = pyplot.subplots()
        # ax.plot(x_axis, results['validation_0']['f1'], label='Train')
        # ax.plot(x_axis, results['validation_1']['f1'], label='Test')
        # ax.legend()
        # pyplot.ylabel('F1 score')
        # pyplot.title('XGBoost F1Error')
        # pyplot.show()

    def early_stop(self, eval_set):
        self.clf.fit(eval_set[0][0], eval_set[0][1],
                     early_stopping_rounds=10, eval_metric="logloss",
                     eval_set=eval_set, verbose=True)

    def fit_perso(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)
