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
from script_12d_train_SN10_clean_high_looser_95low import \
    get_input_dim_from_agpos

import mlflow
from NegativeClassOptimization import (config, ml, pipelines, preprocessing,
                                       utils)

TEST = False
LOG_ARTIFACTS = False
SAVE_LOCAL = True

RESTRICTED_AG_COMBINATIONS = True

experiment_id = 11
run_name = "dev-v0.1.3-expdata"  # "dev-v0.2-shuffled"
num_processes = 10

load_from_miniabsolut = True
shuffle_antigen_labels = False
swa = True
seed_id = [0, 1, 2, 3]  # default was 0
load_from_miniabsolut_split_seeds = [0, 1, 2, 3, 4]  # default None --(internally)--> 42
# seed_id = [0]
# load_from_miniabsolut_split_seeds = []
model_type = "SNN"  # "LogisticRegression"

epochs = 50
learning_rate = 0.001
optimizer_type = "Adam"
momentum = 0.9
weight_decay = 0
batch_size = 64

sample_train = None


def multiprocessing_wrapper_script_12a(
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
        
        # Adjust the load_from_miniabsolut_split_seed
        if load_from_miniabsolut_split_seed is None:
            split_seed = 42
        else:
            split_seed = load_from_miniabsolut_split_seed

        local_dir = Path(
            f"data/Frozen_MiniAbsolut_ML/1v1/seed_{seed_id}/split_{split_seed}/"
            f"{ag_pos}__vs__{ag_neg}/"
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
            input_dim=get_input_dim_from_agpos(ag_pos),
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


if __name__ == "__main__":
    utils.nco_seed()
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    experiment = mlflow.set_experiment(experiment_id=experiment_id)  # type: ignore

    antigens: List[str] = config.ANTIGENS
    ag_perms = list(itertools.permutations(antigens, 2))

    if RESTRICTED_AG_COMBINATIONS:
        ag_perms = [
            ("HR2P", "HR2PSR"),
            ("HR2P", "HR2PIR"),
        ]

        # ag_perms = list(filter(lambda x: x[0] == "1ADQ", ag_perms))
        
        # ag_perms = [
        #     ("1H0D", "1NSN"),
        #     ("3RAJ", "1OB1"),
        #     ("1H0D", "3VRL"),
        #     ("5E94", "1NSN"),
        #     ("5E94", "1OB1"),
        #     ("5E94", "1ADQ"),
        #     ("5E94", "1FBI"),
        #     ("3RAJ", "1FBI"),
        #     ("3RAJ", "1H0D"),
        #     ("3RAJ", "5E94"),
        #     ("3RAJ", "1WEJ"),
        # ]

    if TEST:
        epochs = 3
        learning_rate = 0.001
        optimizer_type = "Adam"
        momentum = 0.9
        weight_decay = 0
        batch_size = 64

        multiprocessing_wrapper_script_12a(
            experiment_id,
            "test",
            "3VRL",  # ag_perms[0][0],
            "1NSN",  # ag_perms[0][1],
            None,
            0,
            None,
        )

    else:
        # Run batched multiprocessing
        for seed in seed_id:
            for i in range(0, len(ag_perms), num_processes):
                print(f"Batch {i} of {len(ag_perms) / num_processes}")
                ag_perms_batch = ag_perms[i : i + num_processes]
                with multiprocessing.Pool(processes=num_processes) as pool:
                    pool.starmap(
                        multiprocessing_wrapper_script_12a,
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
                            for ag_perm in ag_perms_batch
                        ],
                    )
        for load_from_miniabsolut_split_seed in load_from_miniabsolut_split_seeds:
            for i in range(0, len(ag_perms), num_processes):
                print(f"Batch {i} of {len(ag_perms) / num_processes}")
                ag_perms_batch = ag_perms[i : i + num_processes]
                with multiprocessing.Pool(processes=num_processes) as pool:
                    pool.starmap(
                        multiprocessing_wrapper_script_12a,
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
                            for ag_perm in ag_perms_batch
                        ],
                    )
