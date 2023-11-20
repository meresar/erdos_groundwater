import numpy as np
from sklearn.base import BaseEstimator

class gw_mean_predict(BaseEstimator):
    
    def __init__(self):
        self.pred_mean = 0
        self.pred_len = 0

    def fit(self, X, y):
        self.pred_mean = y.mean()

    def predict(self, X, y):
        self.pred_len = X.shape[0]
        return np.ones(self.pred_len)*self.pred_mean

    def score(self, X, y):
        pred = self.predict(X,y)
        score = np.mean((pred-y)**2)
        return -score
