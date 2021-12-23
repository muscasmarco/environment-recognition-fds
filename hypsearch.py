import pickle
import time
import traceback
import sys, os
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

from dataset_getter import DatasetGetter
from grid import Grid
from env_classifier import EnvironmentClassifier

        
def onehot_encode(labels):
    classes = np.sort(np.unique(labels))
    return np.array([np.where(classes == label)[0][0] for label in labels])

def parameter_search():
    print("*** Parameter Search ***")
    start_time = time.time()
 
    columns = ['params', 'acc']
    
    try:
        results_df = pd.read_csv("raw-results.csv")
    except:
        print("Dataframe not found, creating one now.")
        results_df = pd.DataFrame(columns = columns)

    # parse command line arguments
    parser = argparse.ArgumentParser(description="Parser for Parameter Search arguments")
    parser.add_argument("--folder", type=str, default="./", help="The folder in which the results are stored")
    args = parser.parse_args()
    folder = args.folder
    
    pars = {
        "extract_method": "sift",
        "mapping_method": "minibatch_kmeans",
        # "mapping_batch_size": 512,
        # "mapping_feature_size": 200,
        # "predict_method": "log-regr",
    }

    # Setup Grid
    grid = Grid()
    grid.add("extract_method", ["sift", "orb"])
    
    grid.add("mapping_batch_size", [256, 512])
    grid.add("mapping_num_features", [512, 1024, 2048])
    grid.add("mapping_cumulative_bovw", [False]) #[True, False])
    
    grid.add("predict_method", ["log-regr", "svm", "ridge"]) # lin-regr",
    
    print(grid)
    
    print("Number of combinations: " + str(len(grid)))
    
    

    print("-- Loading Data --")
    
    # Dataset is kept the same at each iteration, as to avoid a "lucky" run
    dataset = DatasetGetter(download = False).get_dataframe(load_from_disk = False, balanced = True, shuffle = True)
    train_dataset, test_dataset = train_test_split(dataset, train_size = 0.8, stratify = dataset.label.values, random_state = 0)

    runnr = 0

    # create directory structure
    log_dir = folder + "log/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    model_dir = folder + "model/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    error_log = log_dir + '/error.txt'


    results_list = []

    with open(log_dir + "grid.txt", "w") as info:
        info.write(str(grid))

    for pack in grid:
        runnr += 1
        pars = {**pars, **pack}

        print("Using parameter pack: ", pars)
           
        filename = Grid.generate_filename(pack)

        try:
            print("Iteration code: " + filename)

            
    
            
            env_classifier = EnvironmentClassifier(pars)
            
            env_classifier.fit(train_dataset.image_path, 
                                    train_dataset.label)
            
            results = env_classifier.evaluate(test_dataset.image_path.values,
                                              test_dataset.label.values,
                                              verbose = False)
            


            
            
            test_accuracy = results['acc-score']
            
            print("\n Validation accuracy: %.2f \n" % test_accuracy)

            new_results_df_row = pd.DataFrame(data = [[pars, test_accuracy]], 
                                              columns = columns)
            results_df = results_df.append(new_results_df_row, ignore_index = True)

            results_df.to_csv("raw-results.csv", index = False, na_rep = "nan")
            
            
            
            
            results_list.append(((test_accuracy, pack), filename))
            

            with open(model_dir + "/all_results.pickle", "wb") as f:
                pickle.dump(results_list, f, pickle.HIGHEST_PROTOCOL)

            with open("log/" + folder + "/all.txt", "a") as info:
                info.write("ACC: {}\t{}\n{}\n\n".format(test_accuracy, filename, pack))
                
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            print(sys.exc_info()[0])
            with open(error_log, "a") as error:
                error.write(traceback.format_exc())
                
                
    with open("log/" + folder + "/info.txt", "a") as info:
        info.write("Run finished\n")

    print("Finished parameter search in {:2f} minutes".format((time.time() - start_time) / 60))
    
    return results_df

    
if __name__ == "__main__":
    df = parameter_search()
    
    
    params = [d for d in df.params.values]
    accuracies = df.acc.values
    
    test = pd.DataFrame.from_dict(params)
    test['valid-acc'] = accuracies
    
    test.to_csv("final-results.csv")
    