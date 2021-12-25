# Environment Recognition

This is the repository of our final project for Fundamentals of Data Science - Winter 2021. Our group is composed by: 

- Marco Muscas (1883544)
- Robin Münk (1991774)
- Aizirek Doronbekova  (1985696)
- Sowmya Kommineni Lokanathbabu (1874225)

---

The dataset was directly downloaded from [Modeling the shape of the scene: a holistic representation of the spatial envelope](https://people.csail.mit.edu/torralba/code/spatialenvelope/).

The DatasetGetter class has also a downloader for the dataset that we suggest using, as it organizes the files automatically.

**IMPORTANT** : Since the downloader is actually kind of slow, here is the link for the .zip, already organized. Download it, 
extract it in the project folder and you are ready for testing. Here is the link: [GDrive Dataset Link](https://drive.google.com/file/d/11H2phzOQY5KgWGtktwE_t2RqkLOI2hNM/view?usp=sharing)

---

A bit of explaining could be helpful in recognizing what the various files do: 

- *main.py* - This is the demonstration file that can be run to test the whole pipeline, already provided with the best hyperparameters we found.
- *env_classifier.py* - The image classifier. Give it a dataset of images and labels and it will do the classification automatically.
- *feature_extraction.py* - This one contains the extractor for SIFT, ORB, HSV and RGB histograms.
- *feature_mapping.py* - Here we have the class for making the Bag of Visual Words.
- *prediction.py* - Finally, the predictor class where by setting the parameters, you can choose and easily train a classifier.

- *hypersearch.py* - In this file we set the hyperparameter search using the classes defined in *grid.py* and *env_classifier.py*. 
- *visualization.py* - A utils file for visualization of the confusion matrix, code was kindly provided (and edited to suit our needs) from [this site](https://blog.finxter.com/).

There are also the .csv files that you can take a look at to see what we discovered during our many trials. We suggest taking a look at **final-results.csv**.



