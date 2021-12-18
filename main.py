#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from dataset_getter import DatasetGetter
from hypsearch import Executor
from visualization import confusion_matrix_plot

from sklearn.model_selection import train_test_split


''' GET DATASET '''

download_dataset = False # The repository does not contain the dataset, make sure to download it! (then set this to false)
dataset = DatasetGetter(download = download_dataset).get_dataframe(load_from_disk = False, balanced = True, shuffle = True)





''' SPLITTING INTO TRAIN,TEST '''
train_dataset, test_dataset = train_test_split(dataset, train_size = 0.75, stratify = dataset.label.values)



''' SETTING UP THE MODEL '''
executor = Executor(train_dataset)

parameters = {
    'extract_method': 'sift',
    
    'mapping_method': 'minibatch_kmeans',
    'mapping_feature_size': 2000,
    'mapping_batch_size': 512,
    
    'predict_method': 'svm',
    }

''' TRAINING '''
res_dict = executor.run(pars = parameters)
print("Validation accuracy: %.2f" % res_dict['valid-acc'])


''' EVALUATION on test set '''

y_test_predictions = executor.predict(test_dataset.image_path.values, parameters)
confusion_matrix_plot(test_dataset.label.values, y_test_predictions)


























