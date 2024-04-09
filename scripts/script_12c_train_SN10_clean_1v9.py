"""Clean SN10 training.
"""

import itertools
import math
import multiprocessing
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

import mlflow
from NegativeClassOptimization import (config, ml, pipelines, preprocessing,
                                       utils)

TEST = False
LOG_ARTIFACTS = False
SAVE_LOCAL = True
EPITOPE_BASED = True

experiment_id = 13
run_name = "dev-v0.2.1-epitopes"
num_processes = 10
# local_dir_base = "data/Frozen_MiniAbsolut_ML_shuffled"
local_dir_base = "data/Frozen_MiniAbsolut_ML"

load_from_miniabsolut = True
shuffle_antigen_labels = False
swa = True
# seed_id = [0, 1, 2, 3]  # default was 0
# load_from_miniabsolut_split_seeds = [0, 1, 2, 3, 4]  # default None --(internally)--> 42
seed_id = [0]
load_from_miniabsolut_split_seeds = []
model_type = "SNN"  # "LogisticRegression"

epochs = 50
learning_rate = 0.001
optimizer_type = "Adam"
momentum = 0.9
weight_decay = 0
batch_size = 64

sample_train = None


def multiprocessing_wrapper_script_12c(
    experiment_id,
    run_name,
    ag_pos,
    ag_neg,
    sample_train,
    seed_id,
    load_from_miniabsolut_split_seed,
):
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=run_name,
        description=f"{ag_pos} vs {ag_neg}",
        tags={"mlflow.runName": run_name},
    ):
        # try:

        # Adjust the load_from_miniabsolut_split_seed
        if load_from_miniabsolut_split_seed is None:
            split_seed = 42
        else:
            split_seed = load_from_miniabsolut_split_seed
            
        local_dir = Path(
            f"{local_dir_base}/1v9/seed_{seed_id}/split_{split_seed}/"
            f"{ag_pos}__vs__9/"
        )
        local_dir.mkdir(parents=True, exist_ok=True)

        pipe = pipelines.BinaryclassPipeline(
            log_mlflow=True,
            save_model_mlflow=False,
            log_artifacts=LOG_ARTIFACTS,
            save_local=SAVE_LOCAL,
            local_dir=local_dir,
        )

        pipe.step_1_process_data(
            ag_pos=ag_pos,
            ag_neg=ag_neg,
            sample_train=sample_train,
            batch_size=batch_size,
            shuffle_antigen_labels=shuffle_antigen_labels,
            load_from_miniabsolut=load_from_miniabsolut,
            load_from_miniabsolut_split_seed=load_from_miniabsolut_split_seed,
        )

        pipe.step_2_train_model(
            epochs=epochs,
            learning_rate=learning_rate,
            optimizer_type=optimizer_type,
            momentum=momentum,
            weight_decay=weight_decay,
            swa=swa,
            seed_id=seed_id,
            model_type=model_type,
        )

        pipe.step_3_evaluate_model()
        # except:
            # pass  # Generate all 1 vs 9 antigens combinations


if __name__ == "__main__":
    utils.nco_seed()
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    experiment = mlflow.set_experiment(experiment_id=experiment_id)

    antigens: List[str] = config.ANTIGENS
    # Generate all 1 vs 9 antigens combinations
    ags_1_vs_9 = []
    for ag in antigens:
        if EPITOPE_BASED:
            if ag in config.ANTIGEN_TO_ANTIGEN_EPITOPES.keys():
                ag_epitope = config.ANTIGEN_TO_ANTIGEN_EPITOPES[ag]
                ags_1_vs_9.append((ag_epitope, tuple(ag_i for ag_i in antigens if ag_i != ag)))    
                continue
            else:
                pass
        ags_1_vs_9.append((ag, tuple(ag_i for ag_i in antigens if ag_i != ag)))

    if TEST:
        epochs = 3 
        learning_rate = 0.001
        optimizer_type = "Adam"
        momentum = 0.9
        weight_decay = 0
        batch_size = 64
        # Select 3VRL vs 9
        ags_1_vs_9_test = list(filter(lambda x: x[0] == "3VRL", ags_1_vs_9))

        multiprocessing_wrapper_script_12c(
            experiment_id,
            "test",
            ags_1_vs_9_test[0][0],
            ags_1_vs_9_test[0][1],
            None,  # sample_train
            0,
            None,
        )

    else:
        # Run batched multiprocessing
        for seed in seed_id:
            for i in range(0, len(ags_1_vs_9), num_processes):
                ags_1_vs_9_batch = ags_1_vs_9[i : i + num_processes]
                with multiprocessing.Pool(processes=num_processes) as pool:
                    pool.starmap(
                        multiprocessing_wrapper_script_12c,
                        [
                            (
                                experiment_id,
                                run_name,
                                ag_perm[0],
                                ag_perm[1],
                                sample_train,
                                seed,
                                None,
                            )
                            for ag_perm in ags_1_vs_9_batch
                        ],
                    )
        for load_from_miniabsolut_split_seed in load_from_miniabsolut_split_seeds:
            for i in range(0, len(ags_1_vs_9), num_processes):
                ags_1_vs_9_batch = ags_1_vs_9[i : i + num_processes]
                with multiprocessing.Pool(processes=num_processes) as pool:
                    pool.starmap(
                        multiprocessing_wrapper_script_12c,
                        [
                            (
                                experiment_id,
                                run_name,
                                ag_perm[0],
                                ag_perm[1],
                                sample_train,
                                0,
                                load_from_miniabsolut_split_seed,
                            )
                            for ag_perm in ags_1_vs_9_batch
                        ],
                    )
