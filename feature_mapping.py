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
        self.cumulative_bovw = False
        
        if 'mapping_num_features' in params.keys():
            self.num_features = params['mapping_num_features']
        if 'mapping_method' in params.keys():
            self.method = params['mapping_method']
        if 'mapping_batch_size' in params.keys():
            self.batch_size = params['mapping_batch_size']
        if 'mapping_cumulative_bovw' in params.keys():
            self.cumulative_bovw = params['mapping_cumulative_bovw']
        

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
        
        
    def predict(self, X): # Cumulative = False means onehot encoding of features, otherwise they get summed up
        X_BoVW = np.zeros(shape = (len(X), self.num_features))
                
        for i in range(len(X)):
            prediction = self.mapper.predict(X[i])

            if self.cumulative_bovw: # Another version, for now it's in testing
                X_BoVW[i][prediction] += 1
            else:
                X_BoVW[i][prediction] = 1
        
        return X_BoVW
