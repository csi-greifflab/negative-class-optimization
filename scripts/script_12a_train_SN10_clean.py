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

experiment_id = 11
run_name = "dev-v0.1.2-2"
num_processes = 20

load_from_miniabsolut = True
shuffle_antigen_labels = False
swa = True

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
    ):
    
    with mlflow.start_run(
        experiment_id=experiment_id, 
        run_name=run_name, 
        description=f"{ag_pos} vs {ag_neg}",
        tags={"mlflow.runName": run_name},
        ):

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
        )

        pipe.step_2_train_model(
            epochs=epochs,
            learning_rate=learning_rate,
            optimizer_type=optimizer_type,
            momentum=momentum,
            weight_decay=weight_decay,
            swa=swa,
        )

        pipe.step_3_evaluate_model()


if __name__ == "__main__":

    utils.nco_seed()
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    experiment = mlflow.set_experiment(experiment_id=experiment_id)

    antigens: List[str] = config.ANTIGENS
    ag_perms = list(itertools.permutations(antigens, 2))

    if TEST:
        run_name = "test"

        epochs = 3
        learning_rate = 0.001
        optimizer_type = "Adam"
        momentum = 0.9
        weight_decay = 0
        batch_size = 64
        
        sample_train = 1000
        
        multiprocessing_wrapper_script_12a(
            experiment_id,
            "test",
            ag_perms[0][0],
            ag_perms[0][1],
            sample_train=sample_train,
        )
    
    else:    
        # Run batched multiprocessing
        for i in range(0, len(ag_perms), num_processes):
            print(f"Batch {i} of {len(ag_perms) / num_processes}")
            ag_perms_batch = ag_perms[i:i+num_processes]
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
                        )
                        for ag_perm in ag_perms_batch
                    ]
                )