# Supervised Decision Boundary Maps (sdbm)

This repo contains the code to produce the results presented at the paper **SDBM: Supervised Decision Boundary Maps for Machine Learning
Classifiers**. All the commands must be executed from inside the folder *code*.

# Python Setup

- Install Anaconda Python
- Install tensorflow, MulticoreTSNE and UMAP

```
python -m pip install -r requirements.txt
```

## Collecting the datasets

We use synthetic datasets, and real datasets. The synthetic data are generated with the scikit-learn function `make_blobs`. There are 4 synthetic datasets, each one is a combination of a number of classes (2 or 10) and a number of dimensions for each observation (100 or 700). Each synthetic dataset has 5000 observations and all of them may be generated running the script:

```
python make_blobs.py
```

The real datasets are the MNIST, Fashion-MNIST, UCI HAR Dataset and the Reuters newswire classification dataset. All of them are downloaded and formatted using the script:

```
python get_data.py
```

## Running the experiments

For every experiment script there is more details written as comments directly in the experiment script file (i.e. **.py**).

The first experiment will generate boundary maps for synthetic data (e.g. blobs). The generated images and files will be stored at the folder 'data/results_blobs'.To run this experiment execute the script:

```
python experiment_blobs.py
```

The second experiment will generate boundary maps, for different classifiers over the real datasets described above. It can be executed by running the script:

```
python experiment_real.py
```

The third experiment is designed to compare the performance of SSNP and UMAP+iLAMP while generating
decision boundary maps of different sizes, with different amounts of syntetic data, with 
different number of classes and dimensions. In this experiment it is used only the Logistic Regression
classifier. This experiment can be executed with the script:

```
python experiment_scalability.py
```
