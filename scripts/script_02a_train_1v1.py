"""Clean SN10 training.
"""


import itertools
import math
import multiprocessing
from pathlib import Path
from typing import List
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from docopt import docopt
from script_utils import get_input_dim_from_agpos

import mlflow
from NegativeClassOptimization import (config, ml, pipelines, preprocessing,
                                       utils)

docopt_doc = """Run 1v1 training.

Usage:
    script_02a_train_1v1.py <run_name> <out_dir> <seed_ids> <split_ids>
    script_02a_train_1v1.py <run_name> <out_dir> <seed_ids> <split_ids> --only_generate_datasets
    script_02a_train_1v1.py <run_name> <out_dir> <seed_ids> <split_ids> --shuffle_labels 
    script_02a_train_1v1.py <run_name> <out_dir> <seed_ids> <split_ids> --logistic_regression
    script_02a_train_1v1.py <run_name> <out_dir> <seed_ids> <split_ids> --experimental
    script_02a_train_1v1.py <run_name> <out_dir> <seed_ids> <split_ids> --transformer
    script_02a_train_1v1.py <run_name> <out_dir> <seed_ids> <split_ids> --epitopes
    script_02a_train_1v1.py <run_name> <out_dir> <seed_ids> <split_ids> --overfitting_check

Options:
    -h --help   Show help.
"""


arguments = docopt(docopt_doc, version="NCO")

if arguments["--experimental"] or arguments["--epitopes"] or arguments["--overfitting_check"] or arguments["--transformer"]:
    RESTRICTED_AG_COMBINATIONS = True
else:
    RESTRICTED_AG_COMBINATIONS = False

TEST = False
LOG_ARTIFACTS = False
SAVE_LOCAL = True
load_from_miniabsolut = True
experiment_id = 11
num_processes = 1  # 20
swa = True

antigens: List[str] = config.ANTIGENS  # + config.ANTIGEN_EPITOPES

run_name =  arguments["<run_name>"]  # "dev-v0.2.1-epitopes" "dev-v0.2.1-shuffled" "dev-v0.2-shuffled" "dev-v0.1.3-expdata" "dev-v0.1.2-3-with-replicates"
local_dir_base = arguments["<out_dir>"]
# local_dir_base = "data/Frozen_MiniAbsolut_ML"
# local_dir_base = "data/Frozen_MiniAbsolut_ML_shuffled"

shuffle_antigen_labels = arguments["--shuffle_labels"]  # False

seed_id = [int(i) for i in arguments["<seed_ids>"].split(",")]

if len(arguments["<split_ids>"]) > 0:
    load_from_miniabsolut_split_seeds = [int(i) for i in arguments["<split_ids>"].split(",")]
else:
    load_from_miniabsolut_split_seeds = []

# seed_id = [0]  # default was 0. # [0, 1, 2, 3]
# load_from_miniabsolut_split_seeds = [] # default None --(internally)--> 42. # [0, 1, 2, 3, 4]
# seed_id = [0]
# load_from_miniabsolut_split_seeds = []

model_type = "SNN" if arguments["--logistic_regression"] == False else "LogisticRegression"
model = None

if arguments["--transformer"]:
    model_type = "Transformer"
    model = ml.Transformer(
        d_model=24,  # try 20 - 220
        vocab_size=30,  # 21 minimal vocab_size, related to sequence size.
        nhead=4,
        dim_feedforward=128,
        num_layers=1,
        dropout=0.1,
        transformer_type="absolut_dataset",
    )

if arguments["--transformer"]:
    epochs = 15
    learning_rate = 1e-6
    optimizer_type = "Adam"
    momentum = 0.9
    weight_decay = 0
    batch_size = 128
    sample_train = None
else:
    # Standard parameters
    epochs = 50
    learning_rate = 0.001
    optimizer_type = "Adam"
    momentum = 0.9
    weight_decay = 0
    batch_size = 64
    sample_train = None


# Define parameter set and save for transformer
if arguments["--transformer"]:
    parameter_set = {
        "epochs": epochs,
        "learning_rate": learning_rate,
        "optimizer_type": optimizer_type,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "sample_train": sample_train,
        # Model specific
        "model_type": model_type,
        "d_model": model.d_model,
        "vocab_size": model.vocab_size,
        "nhead": model.nhead,
        "dim_feedforward": model.dim_feedforward,
        "num_layers": model.num_layers,
        "dropout": model.dropout,
        "activation": model.activation,
    }
    # Hash parameter set
    import hashlib
    parameter_set_hash = hashlib.md5(
        str(parameter_set).encode("utf-8")
    ).hexdigest()[:8]

    # Save parameter set
    local_dir_base = f"{local_dir_base}/transformer_parameterset_{parameter_set_hash}"
    if not Path(local_dir_base).exists():
        Path(local_dir_base).mkdir()
    with open(f"{local_dir_base}/parameter_set.json", "w+") as f:
        json.dump(parameter_set, f)


def multiprocessing_wrapper_script_12a(
    experiment_id,
    run_name,
    ag_pos,
    ag_neg,
    sample_train,
    seed_id,
    load_from_miniabsolut_split_seed,
    only_generate_datasets=False,
    model_type=model_type,
    model=model,
):
    # with mlflow.start_run(
    #     experiment_id=experiment_id,
    #     run_name=run_name,
    #     description=f"{ag_pos} vs {ag_neg}",
    #     tags={"mlflow.runName": run_name},
    # ):
    
    # Adjust the load_from_miniabsolut_split_seed
    if load_from_miniabsolut_split_seed is None:
        split_seed = 42
    else:
        split_seed = load_from_miniabsolut_split_seed

    local_dir = Path(
        f"{local_dir_base}/1v1/seed_{seed_id}/split_{split_seed}/"
        f"{ag_pos}__vs__{ag_neg}/"
    )

    if local_dir.exists():
        pass
    else:
        local_dir.mkdir(parents=True)

    pipe = pipelines.BinaryclassPipeline(
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
        model=model,
    )

    pipe.step_3_evaluate_model()


if __name__ == "__main__":
    utils.nco_seed()
    
    # mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    # experiment = mlflow.set_experiment(experiment_id=experiment_id)  # type: ignore
    experiment = ""  # dummy

    ag_perms = list(itertools.permutations(antigens, 2))

    if RESTRICTED_AG_COMBINATIONS:
        if arguments["--experimental"]:
            ag_perms = [
                ("HR2P", "HR2PSR"),
                ("HR2P", "HR2PIR"),
            ]

        # ag_perms = list(filter(lambda x: x[0] == "1ADQ", ag_perms))
        
        if arguments["--epitopes"]:
            antigens: List[str] = config.ANTIGENS + config.ANTIGEN_EPITOPES
            ag_perms = list(itertools.permutations(antigens, 2))
            ag_perms = list(filter(lambda x: x[0] in config.ANTIGEN_EPITOPES, ag_perms))

        # [OLD] Using epitopes only
        # 1) most
        # ag_perms = list(filter(lambda x: x[0] in config.ANTIGEN_EPITOPES, ag_perms))
        # 2) 1v1 within epitopes
        # ag_perms = list(filter(lambda x: x[0] in config.ANTIGEN_EPITOPES and x[1] in config.ANTIGEN_EPITOPES, ag_perms))

        if arguments["--overfitting_check"] or arguments["--transformer"]:
            ag_perms = [
                ("3VRL", "1WEJ"),
                ("1ADQ", "1WEJ"),
                ("3RAJ", "1WEJ"),
            ]

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
                                arguments["--only_generate_datasets"],
                                model_type,
                                model,
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
                                arguments["--only_generate_datasets"],
                                model_type,
                                model,
                            )
                            for ag_perm in ag_perms_batch
                        ],
                    )
