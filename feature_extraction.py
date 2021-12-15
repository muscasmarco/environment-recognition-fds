#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pickle

import cv2 as cv
import numpy as np

class FeatureExtractor:
    __supported_methods = ['orb', 'sift']

    def __init__(self, image_paths): # Methods are 'orb', 'sift'
        self.image_paths = image_paths
        self.results = {}

        try:
            with open("temp/feature_extr.pickle", "rb") as input_file:
                self.results = pickle.load(input_file)
            print("Stored image keypoints loaded")
        except: pass
        
    
    def extract(self, method, verbose = True):

        if method not in self.__supported_methods:
            raise Exception("Feature extraction method not supported. We support ", self.__supported_methods)

        if self.results.get(method, None) is not None:
            return self.results[method]

        result = None

        if verbose:        
            print("Extracting the image descriptors...", end = '')

        if method == 'orb':
            result = self._orb_extract(self.image_paths)
            
        if method == 'sift':
            result = self._sift_extract(self.image_paths)

        self.results[method] = result

        # try:
        os.makedirs("./temp/", exist_ok=True)
        with open("temp/feature_extr.pickle", "wb") as output_file:
            pickle.dump(self.results, output_file , pickle.HIGHEST_PROTOCOL)
        print("Stored features updated")
        # except: pass
        
        if verbose:
            print("Done.")
        
        return result
        
    def _orb_extract(self, image_paths):
        pass
        X = [] # List of image descriptors [[img1_d1, img1_d2, ...], [img2_d1, img2_d2, ...]] # For feature mapping of individual images
        X_all_descriptors = [] # [img1_d1, img1_d2, ..., img2_d1, img2_d2, ...] # For clustering of features
    
        orb = cv.ORB_create() # Declaration of the keypoint detector
    
        for image_path in image_paths:
            
            image = cv.imread(image_path) # Read the image from disk 
            greyscale = cv.cvtColor(image, code = cv.COLOR_BGR2GRAY) # Needs greyscale
            
            _ , descriptors = orb.detectAndCompute(greyscale, None) # The keypoints are not useful for prediction, just get the descriptors
            
            X.append(descriptors)
            X_all_descriptors.extend(descriptors)
    
        X_all_descriptors = np.array(X_all_descriptors)
        return X, X_all_descriptors
    
    
    
    def _sift_extract(self, image_paths):
        pass
    
        X = [] # List of image descriptors [[img1_d1, img1_d2, ...], [img2_d1, img2_d2, ...]] # For feature mapping of individual images
        X_all_descriptors = [] # [img1_d1, img1_d2, ..., img2_d1, img2_d2, ...] # For clustering of features
    
        sift = cv.SIFT_create()

        for image_path in image_paths:
            image = cv.imread(image_path)
            greyscale = cv.cvtColor(image, code = cv.COLOR_BGR2GRAY)
            
            _ , descriptors = sift.detectAndCompute(greyscale, None)
            
            X.append(descriptors)
            X_all_descriptors.extend(descriptors)

        X_all_descriptors = np.array(X_all_descriptors)
        return X, X_all_descriptors
    