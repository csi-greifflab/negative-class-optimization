"""Clean SN10 training.
"""

import itertools
import multiprocessing
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math


import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F


from NegativeClassOptimization import ml, pipelines
from NegativeClassOptimization import utils
from NegativeClassOptimization import preprocessing
from NegativeClassOptimization import config


TEST = False
experiment_id = 13
run_name = "dev-v0.1.2-2-split_replicates"
num_processes = 20

load_from_miniabsolut = True
shuffle_antigen_labels = True
swa = True
seed_id = [0]
load_from_miniabsolut_split_seeds = [0, 1, 2, 3]

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

        try:
            pipe = pipelines.BinaryclassPipeline(
                log_mlflow=True,
                save_model_mlflow=True,
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
            )

            pipe.step_3_evaluate_model()
        except:
            pass    # Generate all 1 vs 9 antigens combinations



if __name__ == "__main__":

    utils.nco_seed()
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    experiment = mlflow.set_experiment(experiment_id=experiment_id)

    antigens: List[str] = config.ANTIGENS
    
    # Generate all 1 vs 9 antigens combinations
    ags_1_vs_9 = []
    for ag in antigens:
        ags_1_vs_9.append((ag, tuple(ag_i for ag_i in antigens if ag_i != ag)))


    if TEST:
        multiprocessing_wrapper_script_12c(
            experiment_id,
            "test",
            ags_1_vs_9[0][0],
            ags_1_vs_9[0][1],
            sample_train=10000,
        )
    
    else:    
        # Run batched multiprocessing
        for seed in seed_id:
            for load_from_miniabsolut_split_seed in load_from_miniabsolut_split_seeds:
                for i in range(0, len(ags_1_vs_9), num_processes):
                    ags_1_vs_9_batch = ags_1_vs_9[i:i+num_processes]
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
                                    load_from_miniabsolut_split_seed,
                                )
                                for ag_perm in ags_1_vs_9_batch
                            ]
                        )