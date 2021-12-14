import time
import traceback
import sys, os
import argparse

import ast

from dataset_getter import DatasetGetter
from feature_extraction import FeatureExtractor
from feature_mapping import FeatureMapper

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class Grid:
    def __init__(self):
        self._grid = {}
        self._position = {}
        self._finished = False
        self._started = False
        self._len = 1

    def __iter__(self):
        if not self._grid:
            raise ValueError("Cannot iterate over empty grid")

        self._started = True
        return self

    def add(self, var_name, values):
        if self._started:
            raise ValueError("Cannot modify Grid after start of iteration")

        self._grid[var_name] = values
        self._position[var_name] = 0
        self._len *= len(values)

    def __next__(self):
        if self._finished:
            raise StopIteration

        pack = {}
        keys = self._grid.keys()
        for k in keys:
            values = self._grid[k]
            pack[k] = values[self._position[k]]

        for i, k in enumerate(keys):
            self._position[k] += 1
            if self._position[k] < len(self._grid[k]):
                break
            else:
                if i == len(self._grid) - 1:
                    self._finished = True
                self._position[k] = 0

        return pack

    def reset(self):
        self._finished = False

    def __len__(self):
        return self._len

    def __str__(self):
        if not self._grid:
            return ""
        s = ""
        for key, values in self._grid.items():
            s += str(key) + ":\t" + str(values) + "\n"
        return s[:-1]

    def dimension(self):
        return len(self._grid)

    def contains(self, name):
        return name in self._grid.keys()

    def values(self):
        return self._grid.values()

    def var_names(self):
        return self._grid.keys()


    @staticmethod
    def fromstring(string):
        string = string.strip()
        grid = Grid()
        lines = string.split("\n")
        for line in lines:
            name, lst = line.split(":")
            values = ast.literal_eval(lst[lst.find("["):])
            grid.add(name, values)
        return grid

    @staticmethod
    def generate_filename(pack):
        filename = ""
        for key in sorted(pack.keys()):
            filename += str(key)[:4] + "_" + str(pack[key]) + "_"
        return "run_" + str(abs(hash(filename)))[:6]

class Executor:
    def __init__(self, dataset):
        self.dataset = dataset
        self.X = None
        self.y = None
        self.descriptors = None
        self.X_BoVW = None
        self.y_test = None
        self.y_test_predictions = None
        self.acc_score = None
        self.pars = None

    def _extract_features(self, extract_method):
        fe = FeatureExtractor(method=extract_method)
        self.X, self.descriptors = fe.extract(self.dataset.image_path.values)
        self.y = self.dataset.label.values

    def _map_features(self, mapping_method='minibatch_kmeans', feature_size=2000, batch_size=512):
        fm = FeatureMapper(num_features=feature_size, method=mapping_method, batch_size=batch_size)
        fm.fit(self.descriptors)  # Make the clusters to later build the BoVW
        self.X_BoVW = fm.to_bag_of_visual_words(self.X)  # And here we build the feature maps through clustering

    def _predict(self, max_iter=1000):
        train_size = 0.8  # Size in percentage of the training split
        X_train, X_test, y_train, self.y_test = train_test_split(self.X_BoVW, self.y, train_size=train_size, stratify=self.y)
        logistic_regression = LogisticRegression(max_iter=max_iter)  # Declare the model
        logistic_regression.fit(X_train, y_train)  # Fit the training data
        self.y_test_predictions = logistic_regression.predict(X_test)  # Make predictions

    def _evaluate(self):
        self.acc_score = accuracy_score(self.y_test_predictions, self.y_test)  # Calculate the accuracy

    def run(self, pars):

        self._extract_features(pars["extract_method"])
        self._map_features(pars["mapping_method"], pars["mapping_feature_size"], pars["mapping_batch_size"])
        self._predict(pars["predict_maxiter"])
        self._evaluate()

        return {
            "acc": self.acc_score
        }

def parameter_search():
    print("*** Parameter Search ***")
    start_time = time.time()

    # parse command line arguments
    parser = argparse.ArgumentParser(description="Parser for Parameter Search arguments")
    parser.add_argument("--folder", type=str, default="parsearch", help="The folder in which the results are stored")
    parser.add_argument("--dataset", type=str, default='QSAR', help="The used dataset. 'QSAR' or 'MNIST'")
    parser.add_argument("--method", type=str, default='tempens', help="The used method. 'pimodel' or 'tempens'")
    args = parser.parse_args()
    folder = args.folder

    pars = {
        "extract_method": ["sift"],
        "mapping_method": ["minibatch_kmeans"],
        "mapping_batch_size": [512],
        "mapping_feature_size": [200],
        "predict_maxiter": [2000]
    }

    # Setup Grid
    grid = Grid()
    grid.add("extract_method", ["orb", "sift"])
    grid.add("mapping_method", ["kmeans", "minibatch_kmeans"])
    grid.add("mapping_batch_size", [512])
    grid.add("mapping_feature_size", [200])
    grid.add("predict_method", ["log-regr"])
    grid.add("predict_maxiter", [2000])
    print(grid)
    print("Size: " + str(len(grid)))


    print("-- Loading Data --")

    download_dataset = False  # The repository does not contain the dataset, make sure to download it! (then set this to false)
    dataset = DatasetGetter(download=download_dataset).get_dataframe(load_from_disk=False)
    executor = Executor(dataset)

    runnr = 0

    # create directory structure
    if not os.path.exists('log'):
        os.makedirs('log')
    if not os.path.exists('log/' + folder):
        os.makedirs('log/' + folder)
    log_dir = 'log/' + folder + '/log'
    model_dir = 'log/' + folder + '/models'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    error_log = 'log/' + folder + '/error.txt'

    results_list = []

    with open("log/" + folder + "/grid.txt", "w") as info:
        info.write(str(grid))

    for pack in grid:
        runnr += 1
        pars = {**pars, **pack}

        filename = Grid.generate_filename(pack)

        log_file = log_dir + '/log_' + filename + ".txt"
        pars['log_file'] = log_file
        results_file = 'log/' + folder + '/' + filename + ".txt"
        pars['results_file'] = results_file
        model_file = model_dir + '/model_' + filename + ".pt"
        pars['model_file'] = model_file

        for i in range(1):
            try:
                print("start with " + filename)

                last_answer = executor.run(pars)

                results_list.append((last_answer.get('acc'), filename))

                with open("log/" + folder + "/info.txt", "w") as info:
                    for acc, name in sorted(results_list, reverse=True):
                        info.write("ACC: {}\t{}\n{}\n\n".format(acc, name, pack))

                break
            except Exception as e:
                print(e)
                print(traceback.format_exc())
                print(sys.exc_info()[0])
                with open(error_log, "a") as error:
                    error.write(traceback.format_exc())
                print('retry after exception ({})'.format(i+1))

    with open("log/" + folder + "/info.txt", "a") as info:
        info.write("Run finished\n")

    print("Finished parameter search in {:2f} minutes".format((time.time() - start_time) / 60))

if __name__ == "__main__":
    parameter_search()
