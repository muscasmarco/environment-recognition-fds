#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from sklearn.cluster import MiniBatchKMeans, KMeans
import numpy as np

class FeatureMapper:
    
    supported_methods = ['kmeans', 'minibatch_kmeans']
    
    def __init__(self, params):

        self.num_features = None
        self.method = None
        self.mapper = None
        self.max_iter = 100 # Default by documentation
        self.batch_size = None
        self.dropout = False
        self.dropout_rate = 0.3
        
        if 'mapping_num_features' in params.keys():
            self.num_features = params['mapping_num_features']
        if 'mapping_method' in params.keys():
            self.method = params['mapping_method']
        if 'mapping_batch_size' in params.keys():
            self.batch_size = params['mapping_batch_size']
        if 'mapping_dropout' in params.keys():
            self.dropout = params['mapping_dropout']
        if 'mapping_dropout_rate' in params.keys():
            self.dropout = params['mapping_dropout_rate']
        

    def fit(self, X, verbose = True):
        
        if verbose:
            print("Mapping the descriptors to feature space...", end = '')

        self.mapper = None
        
        if self.method == 'kmeans':
            self.mapper = KMeans(self.num_features, max_iter= self.max_iter)

        if self.method == 'minibatch_kmeans':
            self.mapper = MiniBatchKMeans(n_clusters = self.num_features,
                                          batch_size = self.batch_size, 
                                          max_iter= self.max_iter)


        self.mapper.fit(X)
        
        if verbose:
            print("Done.")
        
        
    def predict(self, X, training = True): # Cumulative = False means onehot encoding of features, otherwise they get summed up
        X_BoVW = np.zeros(shape = (len(X), self.num_features))
                
        for i in range(len(X)):
            prediction = self.mapper.predict(X[i])

            if self.dropout and training: # Drop features by some probability function
                X_BoVW[i][prediction] += 1
                
                # Values are removed/kep 
                
                #trials = 10
                # Probability of dropout
                #dropout_fun = lambda vec : ( np.random.binomial(p = (vec / (np.sum(vec))), n = trials) > 0 ) * 1
                #X_BoVW[i] = np.random.binomial(p = dropout_fun(X_BoVW[i]), n = 1)
                
                dropout_vector = np.random.binomial(p = (1 - np.ones(self.num_features) * self.dropout_rate), n = 1)
                X_BoVW[i] = X_BoVW[i] * dropout_vector # Elementwise product. Keep the feature or not?
                X_BoVW[i] = ( X_BoVW[i] > 0 ) * 1 # Result is still a onehot
                
                #X_BoVW[i] = X_BoVW[i] / np.sum(X_BoVW[i])
                
            else:
                X_BoVW[i][prediction] = 1
        
        return X_BoVW
