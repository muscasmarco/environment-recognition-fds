#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 20:15:25 2021

@author: marco
"""
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def confusion_matrix_plot(y_true, y_pred):
    
    acc_score = accuracy_score(y_pred,  y_true) # Calculate the accuracy 
    conf_matrix = confusion_matrix(y_true, y_pred) # Create a confusion matrix
    
    # Credit to blog.finxter.com for setting up a nice looking and interpretable confusion matrix plot
    plt.figure(0, figsize = (15, 15))
    
    ax = sns.heatmap(conf_matrix, annot=True, fmt='g')
    
    conf_matrix_title = str("Confusion matrix | Accuracy %.2f" % acc_score)
    ax.set_title(conf_matrix_title)
    
    ax.set_xlabel("Predicted Environment")
    ax.set_ylabel("True environment")
    
    labels = pd.DataFrame(data = pd.unique(y_true), columns = ['label']).sort_values('label').label.values
    
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
