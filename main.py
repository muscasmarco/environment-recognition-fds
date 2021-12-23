#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from dataset_getter import DatasetGetter
from env_classifier import EnvironmentClassifier
from visualization import confusion_matrix_plot

from sklearn.model_selection import train_test_split


''' GET DATASET '''

download_dataset = False # The repository does not contain the dataset, make sure to download it! (then set this to false)
dataset = DatasetGetter(download = download_dataset).get_dataframe(load_from_disk = False, balanced = True, shuffle = True)


''' SPLITTING INTO TRAIN,TEST '''
train_dataset, test_dataset = train_test_split(dataset, train_size = 0.8, stratify = dataset.label.values)



''' SETTING UP THE MODEL '''

parameters = {
    'extract_method': 'sift',
    
    'mapping_method': 'minibatch_kmeans',
    'mapping_num_features': 1024,
    'mapping_batch_size': 256,
    'mapping_max_iter': 200,
    'mapping_cumulative_bovw': False,
    
    'predict_method': 'svm'    
    }



env_classifier = EnvironmentClassifier(parameters)

''' TRAINING '''
env_classifier.fit(train_dataset.image_path, 
                        train_dataset.label)


''' EVALUATION on test set '''

results = env_classifier.evaluate(test_dataset.image_path.values, test_dataset.label.values, verbose = False)

test_accuracy = results['acc-score']
y_test_predictions = results['predictions']

confusion_matrix_plot(test_dataset.label.values, y_test_predictions)
print("(%d samples) | Accuracy: %.2f" % (len(y_test_predictions), test_accuracy))

























