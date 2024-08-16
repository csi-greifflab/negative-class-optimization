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
from docopt import docopt
from script_utils import get_input_dim_from_agpos

# import mlflow
from NegativeClassOptimization import (config, datasets, ml, pipelines,
                                       preprocessing, utils)

docopt_doc = """Run 1v1 training.

Usage:
    script_12d_train_SN10_clean_high_looser_95low.py <run_name> <out_dir> <seed_ids> <split_ids>
    script_12d_train_SN10_clean_high_looser_95low.py <run_name> <out_dir> <seed_ids> <split_ids> --only_generate_datasets
    script_12d_train_SN10_clean_high_looser_95low.py <run_name> <out_dir> <seed_ids> <split_ids> --shuffle_labels
    script_12d_train_SN10_clean_high_looser_95low.py <run_name> <out_dir> <seed_ids> <split_ids> --logistic_regression
    script_12d_train_SN10_clean_high_looser_95low.py <run_name> <out_dir> <seed_ids> <split_ids> --experimental
    script_12d_train_SN10_clean_high_looser_95low.py <run_name> <out_dir> <seed_ids> <split_ids> --epitopes

        
Options:
    -h --help   Show help.
"""


arguments = docopt(docopt_doc, version="NCO")


TEST = False
LOG_ARTIFACTS = False
SAVE_LOCAL = True
experiment_id = 14
num_processes = 20
load_from_miniabsolut = True
swa = True


run_name =  arguments["<run_name>"]  #"dev-v0.2.1-simdif"  # "dev-v0.2.1-epitopes" "dev-v0.2.1-shuffled" "dev-v0.2-shuffled" "dev-v0.1.3-expdata"
local_dir_base = arguments["<out_dir>"]
# local_dir_base = "data/Frozen_MiniAbsolut_ML_shuffled"
# local_dir_base = "data/Frozen_MiniAbsolut_ML"

shuffle_antigen_labels = arguments["--shuffle_labels"]  # False

# seed_id = [0]
# seed_id = [0] # default was 0  [0, 1, 2, 3]
seed_id = [int(i) for i in arguments["<seed_ids>"].split(",")]

if len(arguments["<split_ids>"]) > 0:
    load_from_miniabsolut_split_seeds = [int(i) for i in arguments["<split_ids>"].split(",")]
else:
    load_from_miniabsolut_split_seeds = []
# load_from_miniabsolut_split_seeds = []
# load_from_miniabsolut_split_seeds = []  # default None --(internally)--> 42  [0, 1, 2, 3, 4]

model_type = "SNN" if arguments["--logistic_regression"] == False else "LogisticRegression"


antigens = None  # None for the default 10 antigens from Absolut
if arguments["--epitopes"]:
    antigens = config.ANTIGEN_EPITOPES
if arguments["--experimental"]:
    antigens = ["HR2P"]

# antigens = ["HR2B", "HR2P"]
# antigens = ["HELP"]
# antigens = config.ANTIGEN_EPITOPES
# antigens = [f"{ag}SIM" for ag in config.ANTIGENS] + [f"{ag}DIF" for ag in config.ANTIGENS]


# Standard parameters
epochs = 50
learning_rate = 0.001
optimizer_type = "Adam"
momentum = 0.9
weight_decay = 0
batch_size = 64
sample_train = None


def multiprocessing_wrapper_script_12d(
    experiment_id,
    run_name,
    ag_pos,
    ag_neg,
    sample_train,
    seed_id,
    load_from_miniabsolut_split_seed,
    only_generate_datasets=False,
):
    # with mlflow.start_run(
    #     experiment_id=experiment_id,
    #     run_name=run_name,
    #     description=f"{ag_pos} vs {ag_neg}",
    #     tags={"mlflow.runName": run_name},
    # ):
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

    if local_dir.exists():
        pass
    else:
        local_dir.mkdir(parents=True)

    pipe = pipelines.BinaryclassBindersPipeline(
        log_mlflow=False,
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

    if only_generate_datasets:
        return

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
    
    # mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    # experiment = mlflow.set_experiment(experiment_id=experiment_id)
    experiment = None  # dummy

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
                                arguments["--only_generate_datasets"],
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
                                arguments["--only_generate_datasets"],
                            )
                            for ags in datasets_batch
                        ],
                    )
