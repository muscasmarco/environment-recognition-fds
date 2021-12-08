#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from sklearn.cluster import MiniBatchKMeans, KMeans
import numpy as np

class FeatureMapper:
    
    supported_methods = ['kmeans', 'minibatch_kmeans']
    
    def __init__(self, num_features, method = 'minibatch_kmeans', load_from_disk = False, batch_size = 128):
        self.num_features = num_features
        
        if method not in self.supported_methods:
            raise Exception("Feature mapping method not supported. Try one in: ", self.supported_methods)
        
        self.method = method
        self.batch_size = batch_size
        self.load_from_disk = load_from_disk
        
    def fit(self, X, verbose = True):
        
        
        if verbose:
            print("Mapping the descriptors to feature space...", end = '')

        
        self.mapper = None
        
        if self.method == 'kmeans':
            self.mapper = KMeans(self.num_features)
            
        if self.method == 'minibatch_kmeans':
            self.mapper = MiniBatchKMeans(n_clusters = self.num_features, batch_size = self.batch_size)
        
        
        self.mapper.fit(X)
        
        if verbose:
            print("Done.")
        
        
    def to_bag_of_visual_words(self, X, cumulative = False): # Cumulative = False means onehot encoding of features, otherwise they get summed up
        
    
        X_BoVW = np.zeros(shape = (len(X), self.num_features))

        for i in range(len(X)):
            
            prediction = self.mapper.predict(X[i])
            
            if cumulative:
                X_BoVW[i][prediction] += 1
            else:
                X_BoVW[i][prediction] = 1
            
        return X_BoVW
                
        
'''
n_clusters = 1000
batch_size = 256

mb_kmeans = MiniBatchKMeans(n_clusters = n_clusters, batch_size = batch_size)

mb_kmeans.fit(X_all_descriptors)

X_BoVW = np.zeros(shape = (len(X), n_clusters))

for i in range(len(X)):
    
    prediction = mb_kmeans.predict(X[i])
    
    X_BoVW[i][prediction] = 1
'''
