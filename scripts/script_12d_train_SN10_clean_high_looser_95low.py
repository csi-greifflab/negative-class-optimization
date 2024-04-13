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
from NegativeClassOptimization import (config, datasets, ml, pipelines,
                                       preprocessing, utils)

TEST = False
LOG_ARTIFACTS = False
SAVE_LOCAL = True

experiment_id = 14
run_name = "dev-v0.2.1-simdif"  # "dev-v0.2.1-epitopes" "dev-v0.2.1-shuffled" "dev-v0.2-shuffled" "dev-v0.1.3-expdata"
num_processes = 20
# local_dir_base = "data/Frozen_MiniAbsolut_ML_shuffled"
local_dir_base = "data/Frozen_MiniAbsolut_ML"


load_from_miniabsolut = True
shuffle_antigen_labels = False
swa = True
seed_id = [0] # default was 0  [0, 1, 2, 3]
load_from_miniabsolut_split_seeds = []  # default None --(internally)--> 42  [0, 1, 2, 3, 4]
# seed_id = [0]
# load_from_miniabsolut_split_seeds = []
model_type = "SNN"  # "LogisticRegression"

# antigens = None  # None for the default 10 antigens from Absolut
# antigens = ["HR2B", "HR2P"]
# antigens = ["HELP"]
# antigens = config.ANTIGEN_EPITOPES
antigens = [f"{ag}SIM" for ag in config.ANTIGENS] + [f"{ag}DIF" for ag in config.ANTIGENS]

epochs = 50
learning_rate = 0.001
optimizer_type = "Adam"
momentum = 0.9
weight_decay = 0
batch_size = 64

sample_train = None


def get_input_dim_from_agpos(ag_pos: str) -> int:
    """
    Tmp solution to get the dimension based on the antigen name.
     - if HR2B -> 10*20 = 200
     - if HR2P -> 21*20 = 420
     - otherwise -> 11*20 = 220

    This is an adaptation, so that we can reuse the code for the
    experimental datasets from Brij and Porebski.
    """
    ag = ag_pos.split("_")[0]
    if ag == "HR2B":
        return 200
    elif ag == "HR2P":
        return 420
    elif ag == "HELP":
        return 380  # 19*20
    else:
        return 220


def multiprocessing_wrapper_script_12d(
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
        # Infer task from ag names
        if ag_neg.split("_")[-1] == "looser":
            task = "high_vs_looser"
        elif ag_neg.split("_")[-1] == "95low":
            task = "high_vs_95low"
        else:
            raise ValueError(f"Unknown task for ag_neg: {ag_neg}")

        # Adjust the load_from_miniabsolut_split_seed
        if load_from_miniabsolut_split_seed is None:
            split_seed = 42
        else:
            split_seed = load_from_miniabsolut_split_seed

        local_dir = Path(
            f"{local_dir_base}/{task}/seed_{seed_id}/split_{split_seed}/"
            f"{ag_pos}__vs__{ag_neg}/"
        )
        local_dir.mkdir(parents=True, exist_ok=True)

        pipe = pipelines.BinaryclassBindersPipeline(
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
    experiment = mlflow.set_experiment(experiment_id=experiment_id)

    if antigens is None:
        antigens: List[str] = config.ANTIGENS

    # Generate all datasets
    datasets = []
    for ag in antigens:
        datasets.append((f"{ag}_high", f"{ag}_looser"))
        datasets.append((f"{ag}_high", f"{ag}_95low"))

    if TEST:
        epochs = 3
        learning_rate = 0.001
        optimizer_type = "Adam"
        momentum = 0.9
        weight_decay = 0
        batch_size = 64

        # multiprocessing_wrapper_script_12d(
        #     experiment_id,
        #     "test",
        #     "3VRL_high",
        #     "3VRL_looser",
        #     None,  # sample_train
        #     0,
        #     None,
        # )
        # multiprocessing_wrapper_script_12d(
        #     experiment_id,
        #     "test",
        #     "3VRL_high",
        #     "3VRL_95low",
        #     None,  # sample_train
        #     0,
        #     None,
        # )
        multiprocessing_wrapper_script_12d(
            experiment_id,
            "test",
            "HR2B_high",
            "HR2B_95low",
            None,  # sample_train
            0,
            None,
        )
        multiprocessing_wrapper_script_12d(
            experiment_id,
            "test",
            "HR2P_high",
            "HR2P_95low",
            None,  # sample_train
            0,
            None,
        )

    else:
        # Run batched multiprocessing
        for seed in seed_id:
            for i in range(0, len(datasets), num_processes):
                datasets_batch = datasets[i : i + num_processes]
                with multiprocessing.Pool(processes=num_processes) as pool:
                    pool.starmap(
                        multiprocessing_wrapper_script_12d,
                        [
                            (
                                experiment_id,
                                run_name,
                                ags[0],
                                ags[1],
                                sample_train,
                                seed,
                                None,
                            )
                            for ags in datasets_batch
                        ],
                    )
        for load_from_miniabsolut_split_seed in load_from_miniabsolut_split_seeds:
            for i in range(0, len(datasets), num_processes):
                datasets_batch = datasets[i : i + num_processes]
                with multiprocessing.Pool(processes=num_processes) as pool:
                    pool.starmap(
                        multiprocessing_wrapper_script_12d,
                        [
                            (
                                experiment_id,
                                run_name,
                                ags[0],
                                ags[1],
                                sample_train,
                                0,
                                load_from_miniabsolut_split_seed,
                            )
                            for ags in datasets_batch
                        ],
                    )
