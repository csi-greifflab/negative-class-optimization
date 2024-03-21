"""
Script to compute decision boundaries.
"""


import os
from pathlib import Path
import json
import logging
import multiprocessing
import shutil
import time
from itertools import permutations
from pathlib import Path
from typing import List
import numpy as np

import torch
import torch.optim as optim

from sklearn.utils.extmath import cartesian

from NegativeClassOptimization import config, datasets, preprocessing, ml, decision_boundaries
import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(process)d %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("data/logs/17_compute_decisionboundaries.log"),
        logging.StreamHandler(),
    ],
)


TEST = False
DIR_EXISTS_HANDLE = "ignore"  # "raise" or "skip" or "overwrite" or "ignore"

num_processes = 20
data_dir = Path("data/Frozen_MiniAbsolut_ML/")  # "data/Frozen_MiniAbsolut_ML
loader = datasets.FrozenMiniAbsolutMLLoader(data_dir=data_dir)
task_split_seed_filter = ((42,), (0,))  # split, seed. Set to None for all.
task_types = [
    datasets.ClassificationTaskType.HIGH_VS_LOOSER,
    datasets.ClassificationTaskType.HIGH_VS_95LOW,
    datasets.ClassificationTaskType.ONE_VS_NINE,
    datasets.ClassificationTaskType.ONE_VS_ONE,
]

# SDBM Parameters
dataset_name = "SDBM"
epochs = 200  # 200
patience = 5
verbose = False


def task_generator(task_types=task_types, loader=loader):
    """
    Generate tasks for which to compute attributions.
    """
    seed_split_ids = datasets.FrozenMiniAbsolutMLLoader.generate_seed_split_ids()
    for ag_1, ag_2 in permutations(config.ANTIGENS, r=2):
        for seed_id, split_id in seed_split_ids:
            for task_type in task_types:
                if task_type == datasets.ClassificationTaskType.ONE_VS_ONE:
                    task = datasets.ClassificationTask(
                        task_type=task_type,
                        ag_pos=ag_1,
                        ag_neg=ag_2,
                        seed_id=seed_id,
                        split_id=split_id,
                    )
                else:
                    task = datasets.ClassificationTask(
                        task_type=task_type,
                        ag_pos=ag_1,
                        ag_neg="auto",
                        seed_id=seed_id,
                        split_id=split_id,
                    )
                yield task


def get_model_from_task(task):
    if type(task.model) == torch.optim.swa_utils.AveragedModel:
    # Unwrap the SWA model. We need a module class,
    # that has updated weights, but still has other
    # module funcs, such as forward_logits.
    # Note: swa_model.module has same weights as swa_model.state_dict().
        return task.model.module
    else:
        return task.model



def compute_decisionboundaries(task):

    logger = logging.getLogger()

    task = loader.load(task)
    task.test_dataset = preprocessing.onehot_encode_df(task.test_dataset)
    task.test_dataset["X"] = task.test_dataset["Slide_onehot"]

    X = np.stack(task.test_dataset["X"])
    y = task.test_dataset["y"].astype(float)

    task.basepath = loader.infer_task_basepath(task)
    output_dir = task.basepath / "SDBM"

    # Init and train projector
    n_samples = X.shape[0]
    n_classes = len(np.unique(y))

    X_ssnpgt_proj_file = f'Xtest_SSNP.npy'
    name_projector_ssnp = f"ssnp"

    ssnpgt = decision_boundaries.SSNP(
        epochs=epochs, 
        verbose=verbose, 
        patience=patience, 
        opt='adam', 
        bottleneck_activation='linear'
    )

    if (output_dir / name_projector_ssnp).exists():
        ssnpgt.load_model(output_dir / name_projector_ssnp)
    else: #otherwise it will be fitted
        ssnpgt.fit(X, y)
        ssnpgt.save_model(output_dir / name_projector_ssnp)
    
    # Project points
    if os.path.exists(os.path.join(output_dir, X_ssnpgt_proj_file)):
        logger.info(
            f"Projected SSNP points found! {os.path.join(output_dir,X_ssnpgt_proj_file)}"
        )
        X_ssnpgt = np.load(os.path.join(output_dir, X_ssnpgt_proj_file))
    else:
        logger.info("Projected SSNP points not found! Transforming...")
        X_ssnpgt = ssnpgt.transform(X)
        np.save(os.path.join(output_dir, X_ssnpgt_proj_file), X_ssnpgt)
        logger.info(f"Projected points ({dataset_name}) saved.")

    
    # Classifier
    clf = get_model_from_task(task)
    clf_name = "SN10"
    grid_size = 300

    ssnp_done = False
    out_name = f"{clf_name}_{grid_size}x{grid_size}_{dataset_name}"
    out_file = os.path.join(output_dir, out_name + "_ssnp.npy")

    if os.path.exists(out_file):
        
        # If grid already saved
        
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

        decision_boundaries.results_to_png(
            np_matrix=img_grid_ssnp,
            prob_matrix=prob_grid_ssnp,
            grid_size=grid_size,
            n_classes=n_classes,
            real_points=normalized,
            max_value_hsv=0.8,
            dataset_name=dataset_name,
            classifier_name=clf_name,
            suffix=suffix,
            output_dir=output_dir,
        )

    else:

        # If grid not saved, compute it.
        logger.info("Defining grid around projected 2D points.")
        xmin_ssnp = np.min(X_ssnpgt[:, 0])
        xmax_ssnp = np.max(X_ssnpgt[:, 0])
        ymin_ssnp = np.min(X_ssnpgt[:, 1])
        ymax_ssnp = np.max(X_ssnpgt[:, 1])

        x_intrvls_ssnp = np.linspace(xmin_ssnp, xmax_ssnp, num=grid_size)
        y_intrvls_ssnp = np.linspace(ymin_ssnp, ymax_ssnp, num=grid_size)

        x_grid = np.linspace(0, grid_size - 1, num=grid_size)
        y_grid = np.linspace(0, grid_size - 1, num=grid_size)

        pts_ssnp = cartesian((x_intrvls_ssnp, y_intrvls_ssnp))
        pts_grid = cartesian((x_grid, y_grid))
        pts_grid = pts_grid.astype(int)

        batch_size = min(grid_size**2, 10000)

        # Can probably be moved lower, here not used
        img_grid_ssnp = np.zeros((grid_size, grid_size))
        prob_grid_ssnp = np.zeros((grid_size, grid_size))
    
        pbar = tqdm.tqdm(total=len(pts_ssnp))
        position = 0

        # Iterate over all points in the 2D-grid 
        while True:
            if position >= len(pts_ssnp):
                break

            pts_batch_ssnp = pts_ssnp[position : position + batch_size]
            image_batch_ssnp = ssnpgt.inverse_transform(pts_batch_ssnp)

            probs_ssnp = clf.predict(torch.tensor(image_batch_ssnp)).detach().numpy()
            alpha_ssnp = np.amax(probs_ssnp, axis=1)
            labels_ssnp = probs_ssnp.argmax(axis=1)

            pts_grid_batch = pts_grid[position : position + batch_size]

            img_grid_ssnp[
                pts_grid_batch[:, 0],  # First column
                pts_grid_batch[:, 1],  # Second column
            ] = labels_ssnp

            position += batch_size

            prob_grid_ssnp[
                pts_grid_batch[:, 0],  # First column
                pts_grid_batch[:, 1],  # Second column
            ] = alpha_ssnp

            pbar.update(batch_size)

        pbar.close()
        np.save(os.path.join(output_dir, f"{out_name}_ssnp.npy"), img_grid_ssnp)
        np.save(
            os.path.join(output_dir, f"{out_name}_ssnp_prob.npy"), prob_grid_ssnp
        )

        prob_grid_ssnp = prob_grid_ssnp.clip(max=0.8)

        # Background mode
        normalized = None
        suffix = "ssnp_background"

        decision_boundaries.results_to_png(
            np_matrix=img_grid_ssnp,
            prob_matrix=prob_grid_ssnp,
            grid_size=grid_size,
            n_classes=n_classes,
            real_points=normalized,
            max_value_hsv=0.8,
            dataset_name=dataset_name,
            classifier_name=clf_name,
            suffix=suffix,
            output_dir=output_dir,
        )



if __name__ == "__main__":

    task_data = list(task_generator())
    if task_split_seed_filter is not None:
        task_data = [
            task
            for task in task_data
            if task.split_id in task_split_seed_filter[0]
            and task.seed_id in task_split_seed_filter[1]
        ]
    print(len(task_data))
    if TEST:
        task = task_data[2]
        compute_decisionboundaries(task)
    else:
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.starmap(
                compute_decisionboundaries,
                [(task,) for task in task_data],
            )
