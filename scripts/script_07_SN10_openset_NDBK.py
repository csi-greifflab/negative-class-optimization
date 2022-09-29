import multiprocessing
from itertools import combinations
from pathlib import Path
from typing import Union, List

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

from script_06_SN10_openset_NDB1 import run_main_06


# PARAMS
params_06 = config.PARAMS["06_SN10_openset_NDB1"]
params_07 = config.PARAMS["07_SN10_openset_NDBK"]
experiment_id = params_07["experiment_id"]
run_name = params_07["run_name"]
normalize_data_volume = params_07["normalize_data_volume"]

num_processes = params_06["num_processes"]
epochs = params_06["epochs"]
learning_rate = params_06["learning_rate"]

# CONSTS
TEST = False
TRAINING_SAMPLES_RESTRICTION = 73000


def multiprocessing_wrapper_script_07(
    ag_pair, 
    experiment_id = experiment_id, 
    run_name = run_name, 
    epochs = epochs, 
    learning_rate = learning_rate,
    normalize_data_volume = normalize_data_volume,
    ) -> None:
    
    if normalize_data_volume:
        training_samples_restriction = TRAINING_SAMPLES_RESTRICTION
    else:
        training_samples_restriction = None

    ag_pos, ag_neg = ag_pair
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


if __name__ == "__main__":
    
    np.random.seed(config.SEED)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    experiment = mlflow.set_experiment(experiment_id=experiment_id)

    if not TEST:
        ag_pairs = []
        for ag_pos in config.ANTIGENS_CLOSEDSET:
            antigens_negative_closedset = (
                set(config.ANTIGENS_CLOSEDSET) - set([ag_pos])
            )
            for ag_neg_cardinality in range(1, 6):
                for ag_neg_comb in combinations(antigens_negative_closedset, ag_neg_cardinality):
                    ag_neg_comb_list = list(sorted(ag_neg_comb))
                    ag_pairs.append((ag_pos, ag_neg_comb_list))
    else:
        ag_pairs = [('1FBI', ['3VRL', '1NSN'])]
        print(len(ag_pairs))
        print(ag_pairs)

    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(multiprocessing_wrapper_script_07, ag_pairs)