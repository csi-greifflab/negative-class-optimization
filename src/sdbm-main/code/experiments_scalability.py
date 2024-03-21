"""
In this experiment the goal here is to compare execution times of ssnp and ilamp+umap.

The datasets used are synthetic and the classifier used is the Logistic Regression.
Only vanilla maps (i.e. without shading/transparency) are generated.

A csv file (time.csv) with the time elapsed and the parameters of each trial is also produced.
"""

import tensorflow as tf

import ssnp

import numpy as np
import os

from pathlib import Path
script_path = Path(__file__).parent
os.chdir(script_path)


from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.extmath import cartesian

from sklearn import linear_model

import pandas as pd

from sklearn.datasets import make_blobs

from tqdm import tqdm

import matplotlib.cm as cm
from PIL import Image

import lamp
import umap
from sklearn.neighbors import KDTree

from time import perf_counter

classifier_names = ["lr"]
classifiers = [
    linear_model.LogisticRegression(),
    ]

output_dir = '../data/results_scalability'


class Projector:
    def __init__(self,
        projector2D, inverse_projector,
        X_train, y_train, clf,
        grid_size,
        dataset_name,
        clf_name):
        self.projector2D = projector2D
        self.inverse_projector = inverse_projector
        self.X_train = X_train
        self.y_train = y_train
        self.clf = clf
        self.grid_size = grid_size
        self.dataset_name = dataset_name
        self.clf_name = clf_name

        self.is_fitted = False
        self.transformed = None

        # Defined at create_2D_grid
        self.img_grid = None
        self.prob_grid = None
        self.x_intrvls = None
        self.y_intrvls = None
        self.pts_grid = None
        pass

    def fit(self):
        self.is_fitted = True
        pass

    def transform(self):
        """
        Creates the 2D projection of the input points.
        It initializes the property 'self.transformed'
        """
        self.transformed = self.projector2D.transform(self.X_train)

    def create_2D_grid(self):
        xmin = np.min(self.transformed[:, 0])
        xmax = np.max(self.transformed[:, 0])
        ymin = np.min(self.transformed[:, 1])
        ymax = np.max(self.transformed[:, 1])

        self.img_grid = np.zeros((self.grid_size,)*2)
        self.prob_grid = np.zeros((self.grid_size,)*2)

        # pts are the points in the projection space (2D real plane)
        self.x_intrvls = np.linspace(xmin, xmax, num=self.grid_size)
        self.y_intrvls = np.linspace(ymin, ymax, num=self.grid_size)
        self.pts = cartesian((self.x_intrvls, self.y_intrvls))

        # pts_grid are the points in the discrete 2D space [0, grid_size-1] X [0, grid_size-1]
        x_grid = np.linspace(0, self.grid_size-1, num=self.grid_size)
        y_grid = np.linspace(0, self.grid_size-1, num=self.grid_size)
        self.pts_grid = cartesian((x_grid, y_grid))
        self.pts_grid = self.pts_grid.astype(int)


    def inverse_transform(self, pts_subset):
        pass

    def generate_map(self, batch_size=10000, suffix=""):
        if not self.is_fitted:
            self.fit()
        if self.transformed is None:
            self.transform()
        if self.pts_grid is None:
            self.create_2D_grid()

        pbar = tqdm(total = len(self.pts))

        position = 0
        while True:
            if position >= len(self.pts):
                break

            pts_batch = self.pts[position:position+batch_size]
            image_batch = self.inverse_transform(pts_batch)

            probs = self.clf.predict_proba(image_batch)
            alpha = np.amax(probs , axis=1)
            labels = probs.argmax(axis=1)

            pts_grid_batch = self.pts_grid[position:position+batch_size]
            self.img_grid[
                    pts_grid_batch[:, 0],
                    pts_grid_batch[:, 1] 
                    ] = labels
            
            self.prob_grid[
                    pts_grid_batch[:, 0],
                    pts_grid_batch[:, 1]
                    ] = alpha

            position += batch_size
            pbar.update(batch_size)
        
        pbar.close()

        results_to_png(
            np_matrix=self.img_grid,
            prob_matrix=self.prob_grid,
            grid_size=self.grid_size,
            dataset_name=self.dataset_name,
            classifier_name=self.clf_name,
            suffix=suffix)          
        
        
        pass

class Projector_SSNP(Projector):
    def __init__(self, X_train, y_train, clf,
        grid_size, dataset_name,
            clf_name):
        epochs = 10
        #verbose = False
        verbose = 2
        patience = 0

        ssnpgt = ssnp.SSNP(epochs=epochs,
            verbose=verbose,
            patience=patience,
            opt='adam',
            bottleneck_activation='linear')

        super().__init__(ssnpgt, ssnpgt, X_train, y_train, clf,
            grid_size,
            dataset_name=dataset_name,
            clf_name=clf_name)


    def fit(self):
        self.projector2D.fit(
            self.X_train,
            self.y_train)
        self.is_fitted = True

    #def transform(self): - Default from the base class

    #def create_2D_grid(self): - Default from the base class

    def inverse_transform(self, pts_subset):
        image_batch = self.inverse_projector.inverse_transform(pts_subset)
        return image_batch
        pass



class Projector_UMAP_ILAMP(Projector):
    def __init__(self, X_train, y_train, clf,
        grid_size, dataset_name,
            clf_name):
        umap_reducer = umap.UMAP()
        ilamp = lamp.ilamp
        self._KDTree = None

        super().__init__(umap_reducer, ilamp, X_train, y_train, clf,
            grid_size, dataset_name=dataset_name,
            clf_name=clf_name)

    def fit(self):
        self.projector2D.fit(
            self.X_train,
            self.y_train)
        self.is_fitted = True

    #def transform(self): - Default from the base class

    #def create_2D_grid(self): - Default from the base class

    def inverse_transform(self, pts_subset):
        if self._KDTree is None:
            self._KDTree = KDTree(self.transformed)
        image_batch = self.inverse_projector(
            data=self.X_train,
            data_proj=self.transformed,
            p=pts_subset,
            pre_kdtree=self._KDTree
        )
        return image_batch




def results_to_png(np_matrix, prob_matrix, grid_size, dataset_name, classifier_name, suffix=""):
    global output_dir
    data = cm.tab20(np_matrix/np.max(np_matrix))
    data_vanilla = data[:,:,:3].copy()
    rescaled_vanilla = (255.0 / data_vanilla.max() * (data_vanilla - data_vanilla.min())).astype(np.uint8)
    im = Image.fromarray(rescaled_vanilla)
    print(f"Saving vanilla. {grid_size}x{grid_size} - {dataset_name} - {classifier_name}")
    if len(suffix) > 0: suffix = f"_{suffix}"
    im.save(os.path.join(output_dir,f"{classifier_name}_{grid_size}x{grid_size}_{dataset_name}{suffix}_vanilla.png"))


if __name__ == "__main__":
    patience = 5
    epochs = 200
    
    min_delta = 0.05

    # Let C = 5 be the number of classes/blobs.
    C = 5

    # Let DIMS = [10, 25, 50, 100, 250, 500] be the dimensions list to be tested.
    DIMS = [10, 25, 50, 100, 250, 500]

    # Let OBS = [100, 250, 500, 1000, 3000, 5000] be the number of observations list.
    OBS = [1000,]

    # Let TS be a list of classes to generate the boundary maps.
    TS = [Projector_SSNP, Projector_UMAP_ILAMP]
    
    # Let TS_NAMES be a list of names for the output files.
    TS_NAMES = ['SSNP', 'ILAMP-UMAP']

    verbose = False
    results = []

    os.makedirs(output_dir, exist_ok=True)

    data_dir ='../data'
    
    grid_size = [25, 50, 100, 200, 300]


    for gsize in tqdm(grid_size):
        print(f"Grid size - {gsize}")
        for dim in tqdm(DIMS):
            print(f"DIMS - {dim}")
            for obs in OBS:
                print(f"OBS - {obs}")
                X, y = make_blobs(n_features=dim, n_samples=obs, centers=C)
                
                scaler = MinMaxScaler()
                scaler.fit(X.copy())
                nX = scaler.transform(X)
                nX = nX.astype('float32')
                outX = os.path.join(output_dir, data_dir, f"blobs_d{dim}_o{obs}_c{C}")
                os.makedirs(outX, exist_ok=True)
                np.save(os.path.join(outX, "X.npy"), X)
                np.save(os.path.join(outX, "normalized_X.npy"), nX)
                np.save(os.path.join(outX, "y.npy"), y)

                Xs = [nX, X]
                for t, t_name, X in zip(TS, TS_NAMES, Xs):
                    clf = linear_model.LogisticRegression()
                    clf.fit(X, y) 
                    print(t_name)
                    t0 = perf_counter()
                    pt = t(X, y, clf, gsize, f"gsize{gsize}-dim{dim}-obs{obs}-centers{C}", "LR")
                    pt.generate_map(suffix=f"gsize{gsize}-dim{dim}-obs{obs}-centers{C}-{t_name}")
                    elapsed_time = perf_counter() - t0
                    results.append( (t_name, dim, obs, gsize, elapsed_time) )

    df = pd.DataFrame(results, columns=['test_name', 'dimensions', 'observations', 'grid size', 'elapsed_time'])
    df.to_csv(os.path.join(output_dir, 'time.csv'), header=True, index=None)
