#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pickle

import cv2 as cv
import numpy as np

class FeatureExtractor:
    __supported_methods = ['orb', 'sift', 'rgb', 'hsv']

    def __init__(self, image_paths, load_from_disk = False): # Methods are 'orb', 'sift'
        self.image_paths = image_paths
        self.results = {}
        
        
        if load_from_disk:
            try:
                with open("temp/feature_extr.pickle", "rb") as input_file:
                    self.results = pickle.load(input_file)
                print("Stored image keypoints loaded")
            except: pass
            
    
    def extract(self, image_paths, method, verbose = True):

        if method not in self.__supported_methods:
            raise Exception("Feature extraction method not supported. We support ", self.__supported_methods)

        #if self.results.get(method, None) is not None:
        #    return self.results[method]

        result = None

        if verbose:        
            print("Extracting the image descriptors...", end = '')

        if method == 'orb':
            result = self._orb_extract(image_paths)
            
        if method == 'sift':
            result = self._sift_extract(image_paths)

        if method == 'rgb':
            result = self._rgb_hists_extract(image_paths)

        if method == 'hsv':
            result = self._hsv_hists_extract(image_paths)

        self.results[method] = result

        try:
            os.makedirs("./temp/", exist_ok=True)
            with open("temp/feature_extr.pickle", "wb") as output_file:
                pickle.dump(self.results, output_file , pickle.HIGHEST_PROTOCOL)
                
            print(" stored features updated... ", end = '')
        except: pass
        
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
    
    
    
    
    
    def _rgb_hists_extract(self, image_paths, num_bins = 128, normalize = True):
        pass
            
        X = []
        X_all_hists = [] 
        
        
        for image_path in image_paths:
            
            image = cv.imread(image_path) # Load image from path here as a np.array (G,B,R due to opencv imread)
            bgr_planes = cv.split(image)
            
            # Thanks to opencv documentation we know how to build the histograms for each channel.
            b_hist = cv.calcHist(bgr_planes, [0], None, [num_bins], (0, 256), accumulate = False)
            g_hist = cv.calcHist(bgr_planes, [1], None, [num_bins], (0, 256), accumulate = False)
            r_hist = cv.calcHist(bgr_planes, [2], None, [num_bins], (0, 256), accumulate = False)
            
            hist = np.concatenate((r_hist, g_hist, b_hist))
            
            if normalize: # Normalize the hists
                hist = hist / hist.max()
            
            hist = hist.reshape(hist.size)
            
            X.append(hist)
            X_all_hists.append(hist)
            
    
        X_all_hists = np.array(X_all_hists)
        return X, X_all_hists
    
    
    def _hsv_hists_extract(self, image_paths, num_bins = 128, normalize = True):
        pass
    
        X = []
        X_all_hists = [] 
    
        for image_path in image_paths:
            
            image = cv.imread(image_path) # Load image from path here as a np.array (G,B,R due to opencv imread)
            image = cv.cvtColor(image, cv.COLOR_BGR2HSV) # Conversion to HSV image
            hsv_planes = cv.split(image) # Splitting into HSV channels
            
            h_hist = cv.calcHist(hsv_planes, [0], None, [num_bins], (0, 256), accumulate = False)
            s_hist = cv.calcHist(hsv_planes, [1], None, [num_bins], (0, 256), accumulate = False)
            v_hist = cv.calcHist(hsv_planes, [2], None, [num_bins], (0, 256), accumulate = False)
            
            hist = np.concatenate((h_hist, s_hist, v_hist))
            
            if normalize: # Normalize the hists
                hist = hist / hist.max()
            
            hist = hist.reshape(hist.size)

            
            X.append(hist)
            X_all_hists.append(hist)
            
    
        X_all_hists = np.array(X_all_hists)
        return X, X_all_hists
    