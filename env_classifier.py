#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from feature_extraction import FeatureExtractor
from feature_mapping import FeatureMapper
from prediction import Predictor
from sklearn.metrics import accuracy_score
from prediction import onehot_encode
import numpy as np

class EnvironmentClassifier:
    
    __feature_extractor = None
    __feature_mapper = None
    __predictor = None
    
    
    def __init__(self, params):
        self.params = params
        
        
    def fit(self, X, y):
        
        # Declaring all elements of the pipeline
        self.__feature_extractor = FeatureExtractor(self.params)
        self.__feature_mapper = FeatureMapper(self.params)
        self.__predictor = Predictor(self.params)
        
        
        # Feature Extraction
        X_descriptors, X_all_descriptors = self.__feature_extractor.extract(X)
        
        
        # Mapping the feature to BoVW
        self.__feature_mapper.fit(X_all_descriptors)
        X_BoVW = self.__feature_mapper.predict(X_descriptors, training = True)
        
        
        # Fit of the classification model
        self.__predictor.fit(X_BoVW, y)
        
        return 
        
        
    
    def predict(self, X, verbose = True):
        
        X_descriptors , _  = self.__feature_extractor.extract(X, verbose = verbose)
        X_BoVW = self.__feature_mapper.predict(X_descriptors, training = False) # Deactivate any dropout
        return self.__predictor.predict(X_BoVW)
    
    
    
    def evaluate(self, X, y_true, verbose = True):
        
        y_pred = self.predict(X, verbose)

        if 'ridge' in self.__predictor.method:
            y_true = np.argmax(onehot_encode(y_true), axis = 1)
            y_pred = np.argmax(y_pred, axis = 1)
        
        return {'acc-score': accuracy_score(y_pred, y_true), 
                'predictions': y_pred}
        
        