"""
Workflow for NDBK problem.
"""

import multiprocessing
from itertools import combinations
from pathlib import Path
import random
from typing import Union, List
from NegativeClassOptimization.datasets import construct_dataset_atom_combinations

import mlflow
import NegativeClassOptimization.config as config
import NegativeClassOptimization.ml as ml
import NegativeClassOptimization.preprocessing as preprocessing
import NegativeClassOptimization.utils as utils
import NegativeClassOptimization.visualisations as vis
import numpy as np
import pandas as pd
import torch
import yaml
import logging
from joblib import Parallel, delayed

from script_06_SN10_openset_NDB1 import run_main_06


# PARAMS
params_06 = config.PARAMS["06_SN10_openset_NDB1"]
params_07 = config.PARAMS["07_SN10_openset_NDBK"]
experiment_id = params_07["experiment_id"]
run_name = params_07["run_name"]
sample_train = params_07["sample_train"]
run_all_2class_problems = params_07["run_all_2class_problems"]

num_processes = params_06["num_processes"]
epochs = params_06["epochs"]
learning_rate = params_06["learning_rate"]

# CONSTS
TEST = False
TRAINING_SAMPLES_RESTRICTION = 73000


def multiprocessing_wrapper_script_07(
    ag_pair, 
    experiment_id = experiment_id, 
    run_name = run_name if not TEST else "test",
    epochs = epochs,
    learning_rate = learning_rate,
    sample_train = sample_train,
    ) -> None:
    """Function to multiprocess the workflow.

    Args:
        ag_pair (): 
        experiment_id (, optional): . Defaults to experiment_id.
        run_name (, optional): Defaults to run_name.
        epochs (optional): Defaults to epochs.
        learning_rate (optional): Defaults to learning_rate.
        normalize_data_volume (optional): Defaults to normalize_data_volume.
    """

    logger = logging.getLogger()

    if type(sample_train) == int:
        training_samples_restriction = sample_train
    else:
        training_samples_restriction = None

    ag_pos, ag_neg = ag_pair
    logger.info(f"Start mlflow run for ({ag_pos}, {ag_neg}), {experiment_id=}")
    with mlflow.start_run(
        experiment_id=experiment_id, 
        run_name=run_name, 
        description=f"{ag_pos} vs {ag_neg}"
        ):

        run_main_06(
            epochs, 
            learning_rate, 
            ag_pos, 
            ag_neg, 
            save_model=True,
            sample=(1000 if TEST else None),
            sample_train=(None if TEST else training_samples_restriction),
            )
        
        logger.info(f"End mlflow run for ({ag_pos}, {ag_neg})")
    logger.info(f"End workflow wrapper for ({ag_pos}, {ag_neg})")


if __name__ == "__main__":
    
    logging.basicConfig(
        format="%(asctime)s %(process)d %(funcName)s %(levelname)s %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(
                filename="data/logs/07.log",
                mode="a",
            ),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    logger.info("Start")

    utils.nco_seed()

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    experiment = mlflow.set_experiment(experiment_id=experiment_id)

    if not TEST:
        if run_all_2class_problems:
            ag_pairs: List[List[str]] = construct_dataset_atom_combinations()
        else:
            ag_pairs = []
            for ag_pos in config.ANTIGENS_CLOSEDSET:
                antigens_negative_closedset = (
                    set(config.ANTIGENS_CLOSEDSET) - set([ag_pos])
                )
                for ag_neg_cardinality in range(1, len(config.ANTIGENS_CLOSEDSET) + 1):
                    for ag_neg_comb in combinations(antigens_negative_closedset, ag_neg_cardinality):
                        ag_neg_comb_list = list(sorted(ag_neg_comb))
                        ag_pairs.append((ag_pos, ag_neg_comb_list))
    else:
        ag_pairs = [(['1FBI', '1OB1', '1WEJ'], ['3VRL', '1NSN'])]
        print(len(ag_pairs))
        print(ag_pairs)


    # Direct application might consume too much memory
    # with multiprocessing.Pool(processes=num_processes) as pool:
    #     pool.map(multiprocessing_wrapper_script_07, ag_pairs)

    # Split manually, ag_pairs len ~ 600
    index_ranges = []
    for i in range(0, len(ag_pairs), 60):
        index_ranges.append((i, i+60))

    num_processed = 0
    for start_idx, end_idx in index_ranges:
        ag_pairs_batch = ag_pairs[start_idx:end_idx]
        
        logger.info(f"Processing batch for multiprocessing pooling: {ag_pairs_batch}")
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.map(multiprocessing_wrapper_script_07, ag_pairs_batch)
        
        num_processed += len(ag_pairs_batch)

    assert num_processed == len(ag_pairs), (
        f"Processed #ag_pair ({num_processed=}) "
        f"is different than total ({len(ag_pairs_batch)=})"
    )

    # pool -> joblib for potentially better multiprocess memory management.
    # strange interaction bug with mlflow - experiment_id not found.
    # try:
    #     r = Parallel(n_jobs=num_processes)(
    #         delayed(multiprocessing_wrapper_script_07)(
    #             ag_pair,
    #             # experiment_id = experiment_id, 
    #             # run_name = run_name if not TEST else "test",
    #             # epochs = epochs,
    #             # learning_rate = learning_rate,
    #             # sample_train = sample_train,

    #         )
    #         for ag_pair in ag_pairs[-2:]
    #     )
    # except mlflow.exceptions.MlflowException as err:
    #     logger.exception(f"mlflow error: {err=}, {type(err)=}")
    #     raise