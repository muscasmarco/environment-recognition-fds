from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegressionCV

from sklearn.linear_model import RidgeCV

class Predictor:
    def __init__(self, params):
        
        self.available_models = {
            "log-regr": LogisticRegression(),
            "ridge": RidgeClassifier(),
            "svm": SVC(),
            
            "cv-log-reg": LogisticRegressionCV(),
            "cv-ridge": RidgeCV()
        }
        
        self.method = params['predict_method']
        self.model = self.available_models[self.method]
        

    def fit(self, X, y, verbose = True):
        
        if verbose:
            print("Training the predictor...", end = "")
        
        self.model.fit(X, y)  # Fit the training data
        
        if verbose:
            print("Done.")
        
        return 
    
    def predict(self, X):
        
        return self.model.predict(X)