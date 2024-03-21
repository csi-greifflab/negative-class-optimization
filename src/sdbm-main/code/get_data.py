#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Code adapted from: https://github.com/mespadoto/ssnp/blob/main/code/get_data.py

import argparse
import os
import shutil
import ssl
import tempfile
import zipfile

import numpy as np
import pandas as pd
import scipy.io as sio
import tensorflow as tf
import wget
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras import datasets as kdatasets

def download_file(urls, base_dir, name):
    dir_name = os.path.join(base_dir, name)
    ssl._create_default_https_context = ssl._create_unverified_context

    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

        for url in urls:
            wget.download(url, out=dir_name)

def save_dataset(name, X, y):
    print(name, X.shape)

    lenc = LabelEncoder()
    y = lenc.fit_transform(y)

    for l in np.unique(y):
        print('-->', l, np.count_nonzero(y == l))

    dir_name = os.path.join(base_dir, name)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X.astype('float32'))


    np.save(os.path.join(dir_name, 'X.npy'), X)
    np.save(os.path.join(dir_name, 'y.npy'), y)

    np.savetxt(os.path.join(dir_name, 'X.csv.gz'), X, delimiter=',')
    np.savetxt(os.path.join(dir_name, 'y.csv.gz'), y, delimiter=',')


def remove_all_datasets(base_dir):
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)

def process_reuters():
    (x_train, y_train), (_, _) = kdatasets.reuters.load_data(skip_top=0, test_split=0.0, seed=420)
    word_index = kdatasets.reuters.get_word_index()

    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    sentences = {}

    classes = [3, 4, 11, 16, 19, 20]

    for c in classes:
        print(c, np.sum(y_train == c))
        x_sentences = x_train[np.where(y_train == c)]

        sentences[c] = []
        
        for x in x_sentences:
            decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in x])
            sentences[c].append(decoded_newswire)

    reuters_sentences = []
    reuters_labels = []

    for c in classes:
        reuters_sentences += sentences[c]
        reuters_labels += list(np.repeat(c, len(sentences[c])))

    tfidf = TfidfVectorizer(max_features=5000)
    lenc = LabelEncoder()

    X_reuters = tfidf.fit_transform(reuters_sentences)
    X_reuters = X_reuters.todense()
    y_reuters = lenc.fit_transform(reuters_labels)

    save_dataset('reuters', X_reuters, y_reuters)

def process_fashionmnist():
    (X, y), (_, _) = kdatasets.fashion_mnist.load_data()
    save_dataset('fashionmnist', X.reshape((-1, 28 * 28)), y.squeeze())

def process_mnist():
    (X, y), (_, _) = kdatasets.mnist.load_data()
    save_dataset('mnist', X.reshape((-1, 28 * 28)), y.squeeze())

def process_har():
    har = zipfile.ZipFile('../data/har/UCI HAR Dataset.zip')
    tmp_dir = tempfile.TemporaryDirectory()
    har.extractall(tmp_dir.name)

    df = pd.read_csv(os.path.join(tmp_dir.name, 'UCI HAR Dataset', 'train',
                                  'X_train.txt'), header=None, delim_whitespace=True)
    labels = pd.read_csv(os.path.join(tmp_dir.name, 'UCI HAR Dataset', 'train',
                                      'y_train.txt'), header=None, delim_whitespace=True)

    y = np.array(labels[0]).astype('uint8')
    X = np.array(df)

    y = y - 1

    save_dataset('har', X, y)

if __name__ == '__main__':
    base_dir = '../data/'

    datasets = dict()

    datasets['har'] = ['http://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI HAR Dataset.zip']

    parser = argparse.ArgumentParser(
        description='Dataset Downloader')

    parser.add_argument('-d', action='store_true', help='delete all datasets')
    parser.add_argument('-s', action='store_true',
                        help='skip download, assume files are in place')
    args, unknown = parser.parse_known_args()

    if args.d:
        print('Removing all datasets')
        remove_all_datasets(base_dir)

    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)

    if not args.s:
        print('Downloading all datasets')
        for name, url in datasets.items():
            print('')
            print(name)
            download_file(url, base_dir, name)

    print('')
    print('Processing all datasets')

    for func in sorted([f for f in dir() if f[:8] == 'process_']):
        print(str(func))
        globals()[func]()

