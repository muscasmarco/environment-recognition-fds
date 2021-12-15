from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, RidgeClassifier
from sklearn.svm import SVC

class Predictor:
    def __init__(self, training_size, max_iterations=2000):
        self.training_size = training_size
        self.split = None

        self.available_models = {
            "log-regr": LogisticRegression(max_iter=max_iterations),
            # "lin-regr": LinearRegression(),
            "ridge": RidgeClassifier(),
            "svm": SVC(),
        }
        
        self.trained_models = {}


    def fit(self, X_BoVW, target, method):
        model = self.available_models[method]
        X_train, X_test, y_train, y_test = train_test_split(
            X_BoVW,
            target,
            train_size=self.training_size,
            stratify=target
        )
        model.fit(X_train, y_train)  # Fit the training data
        
        self.trained_models[method] = model
        
        return model.predict(X_test), y_test  # Make predictions
    
    
    def predict(self, X, method):
        
        model = self.trained_models[method]
        
        return model.predict(X)