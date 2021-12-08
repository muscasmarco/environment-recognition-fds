#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from glob import glob    


class DatasetGetter:
    
    def __init__(self, images_folder = "./dataset/images/", image_format = "jpg"):
        self.images_folder = images_folder
        self.image_format = image_format
        
    def get_dataframe(self, load_from_disk = True, save_to_disk = False, dataset_path = "./dataset/oliva_torralba_2001.csv", verbose = True):
        
        dataframe = None
        
        if load_from_disk == False:
            
            # Pattern matching to get images paths (with roots)
            images_paths = glob(self.images_folder + "*." + self.image_format)
            
            # Splitting to get a list of (img_path, label) pairs
            img_lab_splitter = lambda path : (path, path.split("_")[0].split("/")[-1])
            img_lab_pairs = [img_lab_splitter(path) for path in images_paths]
            
            # Get dataframe
            cols = ['image_path', "label"]
            dataframe = pd.DataFrame(data = img_lab_pairs, columns = cols)
            
            if save_to_disk:
                dataframe.to_csv(dataset_path, index = False)
            
        else:
            dataframe = pd.read_csv(dataset_path)
            
        
        return dataframe