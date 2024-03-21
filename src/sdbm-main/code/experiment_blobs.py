"""
In this experiment we create the DBMs using synthetic datasets created with make_blobs.
The DBMs, for each classifier used, are created with the UMAP+iLAMP and with the SSNP techniques.

The observations in the datasets have:
- 100 or 700 dimensions 
and are divided in:
- 2 or 10 classes.

So there are in total 4 datasets, each one with 5000 observations.
"""

import ssnp

from make_blobs import blob_filename

import numpy as np
import os

from pathlib import Path

script_path = Path(__file__).parent
os.chdir(script_path)

import pickle

from time import time

from sklearn.model_selection import train_test_split
from sklearn.utils.extmath import cartesian

from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from skimage.color import rgb2hsv, hsv2rgb

from tqdm import tqdm

import matplotlib.cm as cm
from PIL import Image

from lamp import ilamp
import umap
from sklearn.neighbors import KDTree

classifier_names = ["lr", "svm", "rf", "mlp"]
classifiers = [
    linear_model.LogisticRegression(),
    SVC(kernel="rbf", probability=True),
    RandomForestClassifier(n_estimators=200),
    MLPClassifier(hidden_layer_sizes=(200,) * 3),
]

output_dir = "../data/results_blobs"


def results_to_png(
    np_matrix,
    prob_matrix,
    grid_size,
    n_classes,
    dataset_name,
    classifier_name,
    real_points=None,
    max_value_hsv=None,
    suffix=None,
):
    global output_dir

    if suffix is not None:
        suffix = f"_{suffix}"
    else:
        suffix = ""
    data = cm.tab20(np_matrix / n_classes)
    data_vanilla = data[:, :, :3].copy()

    if max_value_hsv is not None:
        data_vanilla = rgb2hsv(data_vanilla)
        data_vanilla[:, :, 2] = max_value_hsv
        data_vanilla = hsv2rgb(data_vanilla)

    if real_points is not None:
        data_vanilla = rgb2hsv(data_vanilla)
        data_vanilla[real_points[:, 0], real_points[:, 1], 2] = 1
        data_vanilla = hsv2rgb(data_vanilla)

    data_alpha = data.copy()

    data_hsv = data[:, :, :3].copy()
    data_alpha[:, :, 3] = prob_matrix

    data_hsv = rgb2hsv(data_hsv)
    data_hsv[:, :, 2] = prob_matrix
    data_hsv = hsv2rgb(data_hsv)

    rescaled_vanilla = (data_vanilla * 255.0).astype(np.uint8)
    im = Image.fromarray(rescaled_vanilla)
    print(
        f"Saving vanilla. {grid_size}x{grid_size} - {dataset_name} - {classifier_name}"
    )
    im.save(
        os.path.join(
            output_dir,
            f"{classifier_name}_{grid_size}x{grid_size}_{dataset_name}_vanilla{suffix}.png",
        )
    )

    rescaled_alpha = (255.0 * data_alpha).astype(np.uint8)
    im = Image.fromarray(rescaled_alpha)
    print(f"Saving alpha. {grid_size}x{grid_size} - {dataset_name} - {classifier_name}")
    im.save(
        os.path.join(
            output_dir,
            f"{classifier_name}_{grid_size}x{grid_size}_{dataset_name}_alpha{suffix}.png",
        )
    )

    rescaled_hsv = (255.0 * data_hsv).astype(np.uint8)
    im = Image.fromarray(rescaled_hsv)
    print(f"Saving hsv. {grid_size}x{grid_size} - {dataset_name} - {classifier_name}")
    im.save(
        os.path.join(
            output_dir,
            f"{classifier_name}_{grid_size}x{grid_size}_{dataset_name}_hsv{suffix}.png",
        )
    )


if __name__ == "__main__":
    patience = 5
    epochs = 200

    min_delta = 0.05

    verbose = False
    results = []

    os.makedirs(output_dir, exist_ok=True)

    data_dir = "../data"
    # data_dirs = ['mnist', 'fashionmnist', 'har', 'reuters']
    data_dirs = [
        blob_filename(5000, 2, 100),
        blob_filename(5000, 2, 700),
        blob_filename(5000, 10, 100),
        blob_filename(5000, 10, 700),
    ]

    epochs_dataset = {}
    epochs_dataset[blob_filename(5000, 2, 100)] = 10
    epochs_dataset[blob_filename(5000, 2, 700)] = 10
    epochs_dataset[blob_filename(5000, 10, 100)] = 10
    epochs_dataset[blob_filename(5000, 10, 700)] = 10

    grid_size = 300

    for d in tqdm(data_dirs):
        dataset_name = d
        X = np.load(os.path.join(data_dir, d, "X.npy"))
        y = np.load(os.path.join(data_dir, d, "y.npy"))
        n_samples = X.shape[0]
        n_classes = len(np.unique(y))

        # train_size = 10000
        train_size = min(int(n_samples * 0.9), 5000)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, random_state=420, stratify=y
        )
        print("------------------------------------------------------")
        print("Dataset: {0}".format(dataset_name))
        print(X.shape)
        print(y.shape)
        print(np.unique(y))

        # Name of the datapoints file projected by SSNP
        X_ssnpgt_proj_file = f"X_SSNP_{dataset_name}.npy"
        X_umap_proj_file = f"X_UMAP_{dataset_name}.npy"

        # Saved projector pickle file name
        name_projector_ssnp = f"{dataset_name}_ssnp"
        name_projector_umap = f"{dataset_name}_umap"

        epochs = epochs_dataset[dataset_name]
        ssnpgt = ssnp.SSNP(
            epochs=epochs,
            verbose=verbose,
            patience=patience,
            opt="adam",
            bottleneck_activation="linear",
        )

        # If the projector has already been fitted created than it will be loaded.
        if os.path.exists(os.path.join(output_dir, name_projector_ssnp)):
            ssnpgt.load_model(os.path.join(output_dir, name_projector_ssnp))
        else:  # otherwise it will be fitted
            ssnpgt.fit(X_train, y_train)
            ssnpgt.save_model(os.path.join(output_dir, name_projector_ssnp))

        umap_reducer = umap.UMAP()
        # If the projector has already been fitted created than it will be loaded.
        if os.path.exists(os.path.join(output_dir, name_projector_umap)):
            with open(os.path.join(output_dir, name_projector_umap), "rb") as f:
                umap_reducer = pickle.load(f)
        else:  # otherwise it will be fitted
            umap_reducer.fit(X_train, y_train)
            with open(os.path.join(output_dir, name_projector_umap), "wb") as f:
                pickle.dump(umap_reducer, f)

        if os.path.exists(os.path.join(output_dir, X_ssnpgt_proj_file)):
            print(
                f"Projected SSNP points found! {os.path.join(output_dir,X_ssnpgt_proj_file)}"
            )
            X_ssnpgt = np.load(os.path.join(output_dir, X_ssnpgt_proj_file))
        else:
            print("Projected SSNP points not found! Transforming...")
            X_ssnpgt = ssnpgt.transform(X_train)
            np.save(os.path.join(output_dir, X_ssnpgt_proj_file), X_ssnpgt)
            print(f"Projected points ({dataset_name}) saved.")

        if os.path.exists(os.path.join(output_dir, X_umap_proj_file)):
            print(
                f"Projected UMAP points found! {os.path.join(output_dir,X_umap_proj_file)}"
            )
            X_umap = np.load(os.path.join(output_dir, X_umap_proj_file))
        else:
            print("Projected UMAP points not found! Transforming...")
            X_umap = umap_reducer.transform(X_train)
            np.save(os.path.join(output_dir, X_umap_proj_file), X_umap)
            print(f"Projected UMAP points ({dataset_name}) saved.")

        X_kdtree_file = f"X_kdtree_{dataset_name}.pkl"
        if os.path.exists(os.path.join(output_dir, X_kdtree_file)):
            with open(os.path.join(output_dir, X_kdtree_file), "rb") as f:
                X_kdtree = pickle.load(f)
        else:  # otherwise it will be fitted
            X_kdtree = KDTree(X_umap)
            with open(os.path.join(output_dir, X_kdtree_file), "wb") as f:
                pickle.dump(X_kdtree, f)

        for clf_name, clf in zip(classifier_names, classifiers):
            ssnp_done = False
            umap_done = False

            out_name = f"{clf_name}_{grid_size}x{grid_size}_{dataset_name}"
            out_file = os.path.join(output_dir, out_name + "_ssnp.npy")
            if os.path.exists(out_file):
                img_grid_ssnp = np.load(
                    os.path.join(output_dir, out_name + "_ssnp.npy")
                )
                prob_grid_ssnp = np.load(
                    os.path.join(output_dir, out_name + "_ssnp_prob" + ".npy")
                )
                prob_grid_ssnp = prob_grid_ssnp.clip(max=0.8)

                # Background mode
                normalized = None
                suffix = "ssnp_background"

                results_to_png(
                    np_matrix=img_grid_ssnp,
                    prob_matrix=prob_grid_ssnp,
                    grid_size=grid_size,
                    n_classes=n_classes,
                    real_points=normalized,
                    max_value_hsv=0.8,
                    dataset_name=dataset_name,
                    classifier_name=clf_name,
                    suffix=suffix,
                )
                ssnp_done = True

            out_file = os.path.join(output_dir, out_name + "_ilamp.npy")
            if os.path.exists(out_file):
                img_grid_ilamp = np.load(
                    os.path.join(output_dir, out_name + "_ilamp.npy")
                )
                prob_grid_ilamp = np.load(
                    os.path.join(output_dir, out_name + "_ilamp_prob" + ".npy")
                )
                prob_grid_ilamp = prob_grid_ilamp.clip(max=0.8)

                # Background mode
                normalized = None
                suffix = "ilamp_background"

                results_to_png(
                    np_matrix=img_grid_ilamp,
                    prob_matrix=prob_grid_ilamp,
                    grid_size=grid_size,
                    n_classes=n_classes,
                    real_points=normalized,
                    max_value_hsv=0.8,
                    dataset_name=dataset_name,
                    classifier_name=clf_name,
                    suffix=suffix,
                )
                umap_done = True

            if umap_done and ssnp_done:
                continue

            # Saved classifier pickle file name
            name = f"{dataset_name}_{clf_name}.pkl"
            # If the classifier was already fitted than it will be loaded.
            if os.path.exists(os.path.join(output_dir, name)):
                clf = pickle.load(open(os.path.join(output_dir, name), "rb"))
            # otherwise it will be fitted
            else:
                start = time()
                clf.fit(X_train, y_train)
                print("\tAccuracy on test data: ", clf.score(X_test, y_test))
                endtime = time() - start
                print("\tFinished training classifier...", endtime)
                with open(
                    os.path.join(output_dir, f"{dataset_name}_{clf_name}.txt"), "w"
                ) as f:
                    f.write(f"Accuracy on test data: {clf.score(X_test, y_test)}\n")
                    f.write(f"Finished training classifier... {endtime}\n")

            print("Defining grid around projected 2D points.")
            xmin_ssnp = np.min(X_ssnpgt[:, 0])
            xmax_ssnp = np.max(X_ssnpgt[:, 0])
            ymin_ssnp = np.min(X_ssnpgt[:, 1])
            ymax_ssnp = np.max(X_ssnpgt[:, 1])

            xmin_umap = np.min(X_umap[:, 0])
            xmax_umap = np.max(X_umap[:, 0])
            ymin_umap = np.min(X_umap[:, 1])
            ymax_umap = np.max(X_umap[:, 1])

            # Now a we create a grid of points around a 'bounding box'
            img_grid_ssnp = np.zeros((grid_size, grid_size))
            prob_grid_ssnp = np.zeros((grid_size, grid_size))

            img_grid_ilamp = np.zeros((grid_size, grid_size))
            prob_grid_ilamp = np.zeros((grid_size, grid_size))

            x_intrvls_ssnp = np.linspace(xmin_ssnp, xmax_ssnp, num=grid_size)
            y_intrvls_ssnp = np.linspace(ymin_ssnp, ymax_ssnp, num=grid_size)

            x_intrvls_umap = np.linspace(xmin_umap, xmax_umap, num=grid_size)
            y_intrvls_umap = np.linspace(ymin_umap, ymax_umap, num=grid_size)

            x_grid = np.linspace(0, grid_size - 1, num=grid_size)
            y_grid = np.linspace(0, grid_size - 1, num=grid_size)

            pts_ssnp = cartesian((x_intrvls_ssnp, y_intrvls_ssnp))
            pts_umap = cartesian((x_intrvls_umap, y_intrvls_umap))
            pts_grid = cartesian((x_grid, y_grid))
            pts_grid = pts_grid.astype(int)

            batch_size = min(grid_size**2, 10000)

            pbar = tqdm(total=len(pts_ssnp))

            position = 0
            while True:
                if position >= len(pts_ssnp):
                    break

                pts_batch_ssnp = pts_ssnp[position : position + batch_size]
                image_batch_ssnp = ssnpgt.inverse_transform(pts_batch_ssnp)

                pts_batch_umap = pts_umap[position : position + batch_size]
                image_batch_ilamp = ilamp(
                    data=X_train,
                    data_proj=X_umap,
                    p=pts_batch_umap,
                    pre_kdtree=X_kdtree,
                )

                probs_ssnp = clf.predict_proba(image_batch_ssnp)
                alpha_ssnp = np.amax(probs_ssnp, axis=1)
                labels_ssnp = probs_ssnp.argmax(axis=1)

                probs_ilamp = clf.predict_proba(image_batch_ilamp)
                alpha_ilamp = np.amax(probs_ilamp, axis=1)
                labels_ilamp = probs_ilamp.argmax(axis=1)

                pts_grid_batch = pts_grid[position : position + batch_size]

                img_grid_ssnp[
                    pts_grid_batch[:, 0],  # First column
                    pts_grid_batch[:, 1],  # Second column
                ] = labels_ssnp

                img_grid_ilamp[
                    pts_grid_batch[:, 0],  # First column
                    pts_grid_batch[:, 1],  # Second column
                ] = labels_ilamp

                position += batch_size

                prob_grid_ssnp[
                    pts_grid_batch[:, 0],  # First column
                    pts_grid_batch[:, 1],  # Second column
                ] = alpha_ssnp

                prob_grid_ilamp[
                    pts_grid_batch[:, 0],  # First column
                    pts_grid_batch[:, 1],  # Second column
                ] = alpha_ilamp

                pbar.update(batch_size)

            pbar.close()

            np.save(os.path.join(output_dir, f"{out_name}_ssnp.npy"), img_grid_ssnp)
            np.save(
                os.path.join(output_dir, f"{out_name}_ssnp_prob.npy"), prob_grid_ssnp
            )
            np.save(os.path.join(output_dir, f"{out_name}_ilamp.npy"), img_grid_ilamp)
            np.save(
                os.path.join(output_dir, f"{out_name}_ilamp_prob.npy"), prob_grid_ilamp
            )

            out_name = f"{clf_name}_{grid_size}x{grid_size}_{dataset_name}"
            out_file = os.path.join(output_dir, out_name + "_ssnp.npy")
            if os.path.exists(out_file):
                img_grid_ssnp = np.load(
                    os.path.join(output_dir, out_name + "_ssnp.npy")
                )
                prob_grid_ssnp = np.load(
                    os.path.join(output_dir, out_name + "_ssnp_prob" + ".npy")
                )
                prob_grid_ssnp = prob_grid_ssnp.clip(max=0.8)

                # Background mode
                normalized = None
                suffix = "ssnp_background"

                results_to_png(
                    np_matrix=img_grid_ssnp,
                    prob_matrix=prob_grid_ssnp,
                    grid_size=grid_size,
                    n_classes=n_classes,
                    real_points=normalized,
                    max_value_hsv=0.8,
                    dataset_name=dataset_name,
                    classifier_name=clf_name,
                    suffix=suffix,
                )
                ssnp_done = True

            out_file = os.path.join(output_dir, out_name + "_ilamp.npy")
            if os.path.exists(out_file):
                img_grid_ilamp = np.load(
                    os.path.join(output_dir, out_name + "_ilamp.npy")
                )
                prob_grid_ilamp = np.load(
                    os.path.join(output_dir, out_name + "_ilamp_prob" + ".npy")
                )
                prob_grid_ilamp = prob_grid_ilamp.clip(max=0.8)

                # Background mode
                normalized = None
                suffix = "ilamp_background"

                results_to_png(
                    np_matrix=img_grid_ilamp,
                    prob_matrix=prob_grid_ilamp,
                    grid_size=grid_size,
                    n_classes=n_classes,
                    real_points=normalized,
                    max_value_hsv=0.8,
                    dataset_name=dataset_name,
                    classifier_name=clf_name,
                    suffix=suffix,
                )
