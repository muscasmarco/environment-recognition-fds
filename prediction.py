from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegressionCV

from sklearn.linear_model import RidgeCV
import numpy as np

def onehot_encode(labels):
    
    classes = np.sort(np.unique(labels))
    return np.array([(label == classes) + 0 for label in labels])



class Predictor:
    
    def __init__(self, params):
        
        self.available_models = {
            "log-regr": LogisticRegression(max_iter = 2000),
            "ridge": RidgeClassifier(),
            "svm": SVC(),
            
            "cv-log-reg": LogisticRegressionCV(max_iter = 2000),
            "cv-ridge": RidgeCV()
        }
        
        self.method = params['predict_method']
        self.model = self.available_models[self.method]
        

    def fit(self, X, y, verbose = True):
        
        if verbose:
            print("Training the predictor...", end = "")
        
        
        if 'ridge' in self.method:
            y = onehot_encode(y)
        
        self.model.fit(X, y)  # Fit the training data
        
        if verbose:
            print("Done.")
        
        return 
    
    def predict(self, X):
        
        return self.model.predict(X)