"""
In this experiment we create the DBMs using real datasets collected and
formatted through the script `get_data.py`.
The DBMs, for each classifier used, are created with the UMAP+iLAMP
and with the SSNP techniques.

The datasets collected are:
- Fashion-MNIST
- MNIST
- HAR
- REUTERS
"""

import tensorflow as tf

import ssnp

from make_blobs import blob_filename

import numpy as np
import os
import pickle

from time import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.extmath import cartesian

from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from skimage.color import rgb2hsv, hsv2rgb

from tqdm import tqdm

import matplotlib.cm as cm
from PIL import Image

#classifier_name = "lr"
#classifier = linear_model.LogisticRegression()

#classifier_name = "svm"
#classifier = svm.SVC(probability=True)

classifier_names = ["lr", "svm", "rf", "mlp"]
classifiers = [
    linear_model.LogisticRegression(),
    SVC(kernel="rbf", probability=True),
    RandomForestClassifier(n_estimators=200),
    MLPClassifier(hidden_layer_sizes=(200,)*3),
    ]

output_dir = '../data/results_real'

def results_to_png(np_matrix, prob_matrix, grid_size, n_classes,
        dataset_name, classifier_name, real_points=None,
        max_value_hsv=None,
        suffix=None):
    global output_dir

    if suffix is not None:
        suffix = f"_{suffix}"
    else:
        suffix = ""
    data = cm.tab20(np_matrix/n_classes)
    data_vanilla = data[:,:,:3].copy()

    if max_value_hsv is not None:
        data_vanilla = rgb2hsv(data_vanilla)
        data_vanilla[:, :, 2] = max_value_hsv
        data_vanilla = hsv2rgb(data_vanilla)
    
    if real_points is not None:
        data_vanilla = rgb2hsv(data_vanilla)
        data_vanilla[real_points[:, 0], real_points[:, 1], 2] = 1
        data_vanilla = hsv2rgb(data_vanilla)
    
    data_alpha = data.copy()

    data_hsv = data[:,:,:3].copy()
    data_alpha[:,:,3] = prob_matrix

    data_hsv = rgb2hsv(data_hsv)
    data_hsv[:,:,2] = prob_matrix
    data_hsv = hsv2rgb(data_hsv)

    rescaled_vanilla = (data_vanilla*255.0).astype(np.uint8)
    im = Image.fromarray(rescaled_vanilla)
    print(f"Saving vanilla. {grid_size}x{grid_size} - {dataset_name} - {classifier_name}")
    im.save(os.path.join(output_dir,f"{classifier_name}_{grid_size}x{grid_size}_{dataset_name}_vanilla{suffix}.png"))

    rescaled_alpha = (255.0*data_alpha).astype(np.uint8)
    im = Image.fromarray(rescaled_alpha)
    print(f"Saving alpha. {grid_size}x{grid_size} - {dataset_name} - {classifier_name}")
    im.save(os.path.join(output_dir,f"{classifier_name}_{grid_size}x{grid_size}_{dataset_name}_alpha{suffix}.png"))

    rescaled_hsv = (255.0*data_hsv).astype(np.uint8)
    im = Image.fromarray(rescaled_hsv)
    print(f"Saving hsv. {grid_size}x{grid_size} - {dataset_name} - {classifier_name}")
    im.save(os.path.join(output_dir,f"{classifier_name}_{grid_size}x{grid_size}_{dataset_name}_hsv{suffix}.png"))

if __name__ == "__main__":
    patience = 5
    epochs = 200
    
    min_delta = 0.05

    verbose = False
    results = []

    os.makedirs(output_dir, exist_ok=True)

    data_dir = '../data'
    data_dirs = ['mnist', 'fashionmnist', 'har', 'reuters']

    epochs_dataset = {}
    epochs_dataset['fashionmnist'] = 10
    epochs_dataset['mnist'] = 10
    epochs_dataset['har'] = 10
    epochs_dataset['reuters'] = 10

    classes_mult = {}
    classes_mult['fashionmnist'] = 2
    classes_mult['mnist'] = 2
    classes_mult['har'] = 2
    classes_mult['reuters'] = 1

    grid_size = 300

    
    for clf_name, clf in zip(classifier_names, classifiers):
        for d in tqdm(data_dirs):
            dataset_name = d

            X = np.load(os.path.join(data_dir, d,  'X.npy'))
            y = np.load(os.path.join(data_dir, d, 'y.npy'))
            n_samples = X.shape[0]
            n_classes = len(np.unique(y))

            train_size = min(int(n_samples*0.9), 5000)
            test_size = 1000 # inverse

            X_train, X_test, y_train, y_test =\
                train_test_split(X, y, train_size=train_size, random_state=420, stratify=y)

            out_name = f"{clf_name}_{grid_size}x{grid_size}_{dataset_name}"
            out_file = os.path.join(output_dir, out_name + ".npy")
            if os.path.exists(out_file):
                img_grid_ssnp = np.load(os.path.join(output_dir, out_name + ".npy"))
                prob_grid_ssnp = np.load(os.path.join(output_dir, out_name+ "_prob" + ".npy"))
                prob_grid_ssnp = prob_grid_ssnp.clip(max=0.8)

                X_ssnpgt_proj_file = f'X_SSNP_{dataset_name}.npy'
                X_ssnpgt = np.load(os.path.join(output_dir,X_ssnpgt_proj_file))


                # Background mode
                # normalized = None
                # suffix="ssnp_background"

                # With real data mode
                scaler = MinMaxScaler()
                scaler.fit(X_ssnpgt)
                normalized = scaler.transform(X_ssnpgt)
                normalized = normalized.astype('float32')

                normalized *= (grid_size-1)
                normalized = normalized.astype(int)
                
                img_grid_ssnp[normalized[:, 0], normalized[:, 1]] = y_train
                prob_grid_ssnp[normalized[:, 0], normalized[:, 1]] = 1.0
                suffix="ssnp_w_real"


                results_to_png(
                    np_matrix=img_grid_ssnp,
                    prob_matrix=prob_grid_ssnp,
                    grid_size=grid_size,
                    n_classes=n_classes,
                    real_points=normalized,
                    max_value_hsv=0.8,
                    dataset_name=dataset_name,
                    classifier_name=clf_name,
                    suffix=suffix)
                continue

            print('------------------------------------------------------')
            print('Dataset: {0}'.format(dataset_name))
            print(X.shape)
            print(y.shape)
            print(np.unique(y))

            # Name of the datapoints file projected by SSNP
            X_ssnpgt_proj_file = f'X_SSNP_{dataset_name}.npy'

            # Saved classifier pickle file name
            name = f"{dataset_name}_{clf_name}.pkl"
            # Saved projector pickle file name
            name_projector = f"{dataset_name}_ssnp"

            epochs = epochs_dataset[dataset_name]
            ssnpgt = ssnp.SSNP(epochs=epochs, verbose=verbose, patience=patience, opt='adam', bottleneck_activation='linear')
            # If the projector has already been fitted created than it will be loaded.
            if os.path.exists(os.path.join(output_dir,name_projector)):
                ssnpgt.load_model(os.path.join(output_dir,name_projector))
            else: #otherwise it will be fitted
                ssnpgt.fit(X_train, y_train)
                ssnpgt.save_model(os.path.join(output_dir,name_projector))

            # If the classifier was already fitted than it will be loaded.
            if os.path.exists(os.path.join(output_dir,name)):
                clf = pickle.load(open(os.path.join(output_dir,name), "rb"))
            # otherwise it will be fitted
            else:
                start = time()
                clf.fit(X_train, y_train)
                print("\tAccuracy on test data: ", clf.score(X_test, y_test))
                endtime = time() - start
                print("\tFinished training classifier...", endtime)
                with open(os.path.join(output_dir,f"{dataset_name}_{clf_name}.txt"), "w") as f:
                    f.write(f"Accuracy on test data: {clf.score(X_test, y_test)}\n")
                    f.write(f"Finished training classifier... {endtime}\n")


            if os.path.exists(os.path.join(output_dir,X_ssnpgt_proj_file)):
                print(f"Projetcted points found! {os.path.join(output_dir,X_ssnpgt_proj_file)}")
                X_ssnpgt = np.load(os.path.join(output_dir,X_ssnpgt_proj_file))
            else:
                print("Projected points not found! Transforming...")
                X_ssnpgt = ssnpgt.transform(X_train)
                np.save(os.path.join(output_dir,X_ssnpgt_proj_file), X_ssnpgt)
                print(f"Projected points ({dataset_name}) saved.")

            print("Defining grid around projected 2D points.")
            scaler = MinMaxScaler()
            scaler.fit(X_ssnpgt)
            xmin = np.min(X_ssnpgt[:, 0])
            xmax = np.max(X_ssnpgt[:, 0])
            ymin = np.min(X_ssnpgt[:, 1])
            ymax = np.max(X_ssnpgt[:, 1])

            img_grid = np.zeros((grid_size,grid_size))
            prob_grid = np.zeros((grid_size,grid_size))

            x_intrvls = np.linspace(xmin, xmax, num=grid_size)
            y_intrvls = np.linspace(ymin, ymax, num=grid_size)

            x_grid = np.linspace(0, grid_size-1, num=grid_size)
            y_grid = np.linspace(0, grid_size-1, num=grid_size)

            

            pts = cartesian((x_intrvls, y_intrvls))
            pts_grid = cartesian((x_grid, y_grid))
            pts_grid = pts_grid.astype(int)

            batch_size = 100000

            pbar = tqdm(total = len(pts))

            position = 0
            while True:
                if position >= len(pts):
                    break

                pts_batch = pts[position:position+batch_size]
                image_batch = ssnpgt.inverse_transform(pts_batch)

                #labels = classifier.predict(image_batch)
                probs = clf.predict_proba(image_batch)
                alpha = np.amax(probs, axis=1)
                labels = probs.argmax(axis=1)

                #pts_grid_batch = tuple(pts_grid[position:position+batch_size])
                pts_grid_batch = pts_grid[position:position+batch_size]

                img_grid[
                    pts_grid_batch[:, 0], # Primeira coluna
                    pts_grid_batch[:, 1]  # Segunda coluna
                    ] = labels
                
                position += batch_size

                prob_grid[
                    pts_grid_batch[:, 0], # Primeira coluna
                    pts_grid_batch[:, 1]  # Segunda coluna
                    ] = alpha
                
                pbar.update(batch_size)

            pbar.close()

            np.save(os.path.join(output_dir,f"{out_name}.npy"), img_grid)
            np.save(os.path.join(output_dir,f"{out_name}_prob.npy"), prob_grid)
            results_to_png(
                np_matrix=img_grid,
                prob_matrix=prob_grid,
                grid_size=grid_size,
                dataset_name=dataset_name,
                classifier_name=clf_name,
                n_classes=n_classes)
