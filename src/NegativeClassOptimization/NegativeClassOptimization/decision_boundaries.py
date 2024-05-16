"""
Code for the decision boundaries computations.

Main function is the SSNP (Self-supervised neural projector), based on which the 
Self-supervised decision boundary maps (SDBM) was developed.

Resources:
- https://github.com/mespadoto/sdbm 
- Quantitative and Qualitative Comparison of Decision-Map Techniques for Explaining Classification Models 
- SDBM: Supervised Decision Boundary Maps for Machine Learning Classifiers
- iLAMP: Exploring High-Dimensional Spacing through Backward Multidimensional Projection
"""


import os

import matplotlib.cm as cm
import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from skimage.color import hsv2rgb, rgb2hsv
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.extmath import cartesian
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

os.environ['TF_DETERMINISTIC_OPS'] = '1'


class SSNP():
    """
    Self-supervised neural projector:
    - https://github.com/mespadoto/sdbm
    - SDBM: Supervised Decision Boundary Maps for Machine Learning Classifiers 
    """
    
    def __init__(self, init_labels='precomputed', epochs=100,
            input_l1=0.0, input_l2=0.0, bottleneck_l1=0.0,
            bottleneck_l2=0.5, verbose=1, opt='adam',
            bottleneck_activation='tanh', act='relu',
            init='glorot_uniform', bias=0.0001, patience=3,
            min_delta=0.01):
        self.init_labels = init_labels
        self.epochs = epochs
        self.verbose = verbose
        self.opt = opt
        self.act = act
        self.init = init
        self.bias = bias
        self.input_l1 = input_l1
        self.input_l2 = input_l2
        self.bottleneck_l1 = bottleneck_l1
        self.bottleneck_l2 = bottleneck_l2
        self.bottleneck_activation = bottleneck_activation
        self.patience = patience
        self.min_delta = min_delta

        self.label_bin = LabelBinarizer()

        self.model = None
        self.fwd = None
        self.inv = None

        tf.random.set_seed(42)

        self.is_fitted = False
        K.clear_session()

    def save_model(self, saved_model_folder):
        tf.keras.models.save_model(self.model, saved_model_folder)

    def load_model(self, saved_model_folder):
        self.model = tf.keras.models.load_model(saved_model_folder)


        self.model.compile(optimizer=self.opt,
                    loss={'main_output': 'categorical_crossentropy', 'decoder_output': 'binary_crossentropy'},
                    metrics=['accuracy'])
        
        model = self.model

        main_input = model.inputs
        main_output = model.get_layer('main_output')
        encoded = model.get_layer('encoded')


        encoded_input = Input(shape=(2,))
        l = model.get_layer('enc1')(encoded_input)
        l = model.get_layer('enc2')(l)
        l = model.get_layer('enc3')(l)
        decoder_layer = model.get_layer('decoder_output')(l)

        self.inv = Model(encoded_input, decoder_layer)

        self.fwd = Model(inputs=main_input, outputs=encoded.output)
        self.clustering = Model(inputs=main_input, outputs=main_output.output)

        self.is_fitted = True


    def fit(self, X, y=None):
        if y is None and self.init_labels == 'precomputed':
            raise Exception('Must provide labels when using init_labels = precomputed')
        
        if y is None:
            y = self.init_labels.fit_predict(X)

        self.label_bin.fit(y)

        main_input = Input(shape=(X.shape[1],), name='main_input')
        x = Dense(512,  activation=self.act,
                        kernel_initializer=self.init,
                        bias_initializer=Constant(self.bias))(main_input)
        x = Dense(128,  activation=self.act,
                        kernel_initializer=self.init,
                        bias_initializer=Constant(self.bias))(x)
        x = Dense(32, activation=self.act,
                        activity_regularizer=regularizers.l1_l2(l1=self.input_l1, l2=self.input_l2),
                        kernel_initializer=self.init,
                        bias_initializer=Constant(self.bias))(x)
        encoded = Dense(2,
                        activation=self.bottleneck_activation,
                        kernel_regularizer=regularizers.l1_l2(l1=self.bottleneck_l1, l2=self.bottleneck_l2),
                        kernel_initializer=self.init,
                        name='encoded',
                        bias_initializer=Constant(self.bias))(x)

        x = Dense(32, activation=self.act, kernel_initializer=self.init, name='enc1', bias_initializer=Constant(self.bias))(encoded)
        x = Dense(128, activation=self.act, kernel_initializer=self.init, name='enc2', bias_initializer=Constant(self.bias))(x)
        x = Dense(512, activation=self.act, kernel_initializer=self.init, name='enc3', bias_initializer=Constant(self.bias))(x)

        n_classes = len(np.unique(y))
        
        if n_classes == 2:
            n_units = 1
            main_output_activation = 'sigmoid'
            main_loss = 'binary_crossentropy'
        else:
            n_units = n_classes
            main_output_activation = 'softmax'
            main_loss = 'categorical_crossentropy'

        main_output = Dense(n_units,
                            activation=main_output_activation,
                            name='main_output',
                            kernel_initializer=self.init,
                            bias_initializer=Constant(self.bias))(x)

        decoder_output = Dense( X.shape[1],
                                activation='sigmoid',
                                name='decoder_output',
                                kernel_initializer=self.init,
                                bias_initializer=Constant(self.bias))(x)

        model = Model(inputs=main_input, outputs=[main_output, decoder_output])
        self.model = model 

        model.compile(optimizer=self.opt,
                    loss={'main_output': main_loss, 'decoder_output': 'binary_crossentropy'},
                    metrics=['accuracy'])

        if self.patience > 0:
            callbacks = [EarlyStopping(monitor='val_loss', mode='min', min_delta=self.min_delta, patience=self.patience, restore_best_weights=True, verbose=self.verbose)]
        else:
            callbacks = []

        

        hist = model.fit(X,
                    [self.label_bin.transform(y), X],
                    batch_size=32,
                    epochs=self.epochs,
                    shuffle=True,
                    verbose=self.verbose,
                    validation_split=0.05,
                    callbacks=callbacks)

        encoded_input = Input(shape=(2,))
        l = model.get_layer('enc1')(encoded_input)
        l = model.get_layer('enc2')(l)
        l = model.get_layer('enc3')(l)
        decoder_layer = model.get_layer('decoder_output')(l)

        self.inv = Model(encoded_input, decoder_layer)

        self.fwd = Model(inputs=main_input, outputs=encoded)
        self.clustering = Model(inputs=main_input, outputs=main_output)
        self.is_fitted = True

        return hist

    def transform(self, X):
        if self._is_fit():
            return self.fwd.predict(X)
           
    def inverse_transform(self, X_2d):
        if self._is_fit():
            return self.inv.predict(X_2d)

    def predict(self, X):
        if self._is_fit():
            y_pred = self.clustering.predict(X)
            return self.label_bin.inverse_transform(y_pred)

    def _is_fit(self):
        if self.is_fitted:
            return True
        else:
            raise Exception('Model not trained. Call fit() before calling transform()')


def results_to_png(
    np_matrix,
    prob_matrix,
    grid_size,
    n_classes,
    dataset_name,
    classifier_name,
    output_dir,
    real_points=None,
    max_value_hsv=None,
    suffix=None,
):
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

    imgs = []
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
    imgs.append(im)

    rescaled_alpha = (255.0 * data_alpha).astype(np.uint8)
    im = Image.fromarray(rescaled_alpha)
    print(f"Saving alpha. {grid_size}x{grid_size} - {dataset_name} - {classifier_name}")
    im.save(
        os.path.join(
            output_dir,
            f"{classifier_name}_{grid_size}x{grid_size}_{dataset_name}_alpha{suffix}.png",
        )
    )
    imgs.append(im)

    rescaled_hsv = (255.0 * data_hsv).astype(np.uint8)
    im = Image.fromarray(rescaled_hsv)
    print(f"Saving hsv. {grid_size}x{grid_size} - {dataset_name} - {classifier_name}")
    im.save(
        os.path.join(
            output_dir,
            f"{classifier_name}_{grid_size}x{grid_size}_{dataset_name}_hsv{suffix}.png",
        )
    )
    imgs.append(im)
    
    return imgs


def compute_decision_boundary_coords(clf, ssnpgt, X_ssnpgt, grid_size):
    """Compute the decision boundary coordinates."""

    # Do the nicer plots, developed in 07i_Decision_boundaries_2
    xmin_ssnp = np.min(X_ssnpgt[:, 0])
    xmax_ssnp = np.max(X_ssnpgt[:, 0])
    ymin_ssnp = np.min(X_ssnpgt[:, 1])
    ymax_ssnp = np.max(X_ssnpgt[:, 1])

    x_intrvls_ssnp = np.linspace(xmin_ssnp, xmax_ssnp, num=grid_size)
    y_intrvls_ssnp = np.linspace(ymin_ssnp, ymax_ssnp, num=grid_size)
    pts_ssnp = cartesian((x_intrvls_ssnp, y_intrvls_ssnp))

    x_grid = np.linspace(0, grid_size - 1, num=grid_size)
    y_grid = np.linspace(0, grid_size - 1, num=grid_size)
    pts_grid = cartesian((x_grid, y_grid))
    pts_grid = pts_grid.astype(int)

    pred_pts = clf.predict(torch.tensor(
        ssnpgt.inverse_transform(pts_ssnp)
    )).detach().numpy()

    # Pred matrix
    pred_matrix = np.zeros((grid_size, grid_size))
    for i, (x, y) in enumerate(pts_grid):
        pred_matrix[x, y] = pred_pts[i]

    # Boundary matrix
    boundary_matrix = np.zeros((grid_size, grid_size))
    i_last = -1
    for i in range(grid_size):
        j_last = -1
        for j in range(grid_size):
            curr_val = round(pred_matrix[i, j])
            if j_last == -1:
                j_last = j
                continue
            if curr_val != round(pred_matrix[i, j_last]):
                boundary_matrix[i, j] = 1
            j_last = j
            

    # Boundary coordinates in real coordinates (not grid)
    decision_boundary_pts = np.where(boundary_matrix == 1)
    decision_boundary_coords = []
    for i in range(decision_boundary_pts[0].shape[0]):
        x = decision_boundary_pts[0][i]
        y = decision_boundary_pts[1][i]
        x_real = x_intrvls_ssnp[x]
        y_real = y_intrvls_ssnp[y]  
        decision_boundary_coords.append([x_real, y_real])

    decision_boundary_coords = np.array((sorted(decision_boundary_coords, key=lambda x: x[0])))
    return decision_boundary_coords


def rotate(p, origin=(0, 0), degrees=0):
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)