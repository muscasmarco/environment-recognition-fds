#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from dataset_getter import DatasetGetter
from feature_extraction import FeatureExtractor
from feature_mapping import FeatureMapper


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

''' GET DATASET '''

download_dataset = False # The repository does not contain the dataset, make sure to download it! (then set this to false)
dataset = DatasetGetter(download = download_dataset).get_dataframe(load_from_disk = False)



''' FEATURE EXTRACTION '''
fe = FeatureExtractor(method = 'sift')
X, descriptors = fe.extract(dataset.image_path.values)
y = dataset.label.values




''' FEATURE MAPPING '''
fm = FeatureMapper(num_features = 2000, method = 'minibatch_kmeans', batch_size = 512)
fm.fit(descriptors) # Make the clusters to later build the BoVW
X_BoVW = fm.to_bag_of_visual_words(X) # And here we build the feature maps through clustering



''' PREDICTION '''
train_size = 0.8 # Size in percentage of the training split
X_train, X_test, y_train, y_test = train_test_split(X_BoVW,y, train_size = train_size, stratify = y)

logistic_regression = LogisticRegression(max_iter = 1000) # Declare the model
logistic_regression.fit(X_train, y_train) # Fit the training data
y_test_predictions = logistic_regression.predict(X_test) # Make predictions




''' EVALUATION '''

acc_score = accuracy_score(y_test_predictions,  y_test) # Calculate the accuracy 
conf_matrix = confusion_matrix(y_test, y_test_predictions) # Create a confusion matrix

# Credit to blog.finxter.com for setting up a nice looking and interpretable confusion matrix plot
plt.figure(0, figsize = (15, 15))

ax = sns.heatmap(conf_matrix, annot=True, fmt='g');

conf_matrix_title = str("Confusion matrix | Accuracy %.2f" % acc_score)
ax.set_title(conf_matrix_title)

ax.set_xlabel("Predicted Environment")
ax.set_ylabel("True environment")

labels = pd.DataFrame(data = pd.unique(y_test), columns = ['label']).sort_values('label').label.values

ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels)



























