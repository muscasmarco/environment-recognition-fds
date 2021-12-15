#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from glob import glob    
import requests
import os
import zipfile

class DatasetGetter:
    
    def __init__(self, images_folder = "./dataset/images/", image_format = "jpg", download = False):
        self.images_folder = images_folder
        self.image_format = image_format
        
        if download:
            dataset_link = "https://people.csail.mit.edu/torralba/code/spatialenvelope/spatial_envelope_256x256_static_8outdoorcategories.zip"
            response = requests.get(dataset_link, stream=True)
            
            
            try:
                os.makedirs("./dataset/")
            except FileExistsError:
                print(images_folder + " already exists, skipping creation.")
            
            print("Downloading dataset...", end = '')
            with open("./dataset/dataset.zip", "wb") as handle:
                for data in response.iter_content():
                    handle.write(data)
            print("Done.")
            
            print("Extracting zip...", end = '')
            with zipfile.ZipFile("./dataset/dataset.zip", 'r') as zip_ref:
                zip_ref.extractall("./dataset/")
            
            os.rename("./dataset/spatial_envelope_256x256_static_8outdoorcategories/", "./dataset/images/")
            
            print("Done.")
            
        
        
    def get_dataframe(self, balanced = True, shuffle = True, load_from_disk = False, save_to_disk = True, dataset_path = "./dataset/oliva_torralba_2001.csv", verbose = True):
        
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
        
        if balanced:
            # Thanks to Samuel Nde, StackOverflow for providing a balancing by label method
            groups = dataframe.groupby('label')
            dataframe = pd.DataFrame(groups.apply(lambda x: x.sample(groups.size().min()).reset_index(drop=True)))

        if shuffle:
            dataframe = dataframe.sample(frac = 1).reset_index(drop = True)
        
        return dataframe
