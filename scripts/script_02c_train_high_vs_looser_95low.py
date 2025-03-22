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
import json
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
    script_02c_train_high_vs_looser_95low.py <run_name> <out_dir> <seed_ids> <split_ids>
    script_02c_train_high_vs_looser_95low.py <run_name> <out_dir> <seed_ids> <split_ids> --only_generate_datasets
    script_02c_train_high_vs_looser_95low.py <run_name> <out_dir> <seed_ids> <split_ids> --shuffle_labels
    script_02c_train_high_vs_looser_95low.py <run_name> <out_dir> <seed_ids> <split_ids> --logistic_regression
    script_02c_train_high_vs_looser_95low.py <run_name> <out_dir> <seed_ids> <split_ids> --cnn
    script_02c_train_high_vs_looser_95low.py <run_name> <out_dir> <seed_ids> <split_ids> --transformer
    script_02c_train_high_vs_looser_95low.py <run_name> <out_dir> <seed_ids> <split_ids> --experimental --cnn
    script_02c_train_high_vs_looser_95low.py <run_name> <out_dir> <seed_ids> <split_ids> --experimental --transformer
    script_02c_train_high_vs_looser_95low.py <run_name> <out_dir> <seed_ids> <split_ids> --experimental
    script_02c_train_high_vs_looser_95low.py <run_name> <out_dir> <seed_ids> <split_ids> --epitopes
    script_02c_train_high_vs_looser_95low.py <run_name> <out_dir> <seed_ids> <split_ids> --x10under
    script_02c_train_high_vs_looser_95low.py <run_name> <out_dir> <seed_ids> <split_ids> --x50under
    script_02c_train_high_vs_looser_95low.py <run_name> <out_dir> <seed_ids> <split_ids> --overfitting_check
    script_02c_train_high_vs_looser_95low.py <run_name> <out_dir> <seed_ids> <split_ids> --esm2b
    script_02c_train_high_vs_looser_95low.py <run_name> <out_dir> <seed_ids> <split_ids> --experimental --esm2b
    script_02c_train_high_vs_looser_95low.py <run_name> <out_dir> <seed_ids> <split_ids> --antiberta2
    script_02c_train_high_vs_looser_95low.py <run_name> <out_dir> <seed_ids> <split_ids> --experimental --antiberta2
        
Options:
    -h --help   Show help.
"""

arguments = docopt(docopt_doc, version="NCO")

TEST = False
LOG_ARTIFACTS = False
SAVE_LOCAL = True
experiment_id = 14
num_processes = 1 # 20
load_from_miniabsolut = True
swa = True


run_name = arguments["<run_name>"]  #"dev-v0.2.1-simdif"  # "dev-v0.2.1-epitopes" "dev-v0.2.1-shuffled" "dev-v0.2-shuffled" "dev-v0.1.3-expdata"
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

model = None

if arguments["--logistic_regression"]:
    model_type = "LogisticRegression"
elif arguments["--cnn"]:
    model_type = "CNN"
elif arguments["--transformer"] and arguments["--experimental"]:
    model_type = "Transformer"
    model = ml.Transformer(
        d_model=24,  # try 20 - 220
        vocab_size=30,  # 21 minimal vocab_size, related to sequence size.
        nhead=4,
        dim_feedforward=128,
        num_layers=1,
        dropout=0.1,
        transformer_type="experimental_dataset",
    )
elif arguments["--transformer"] and (not arguments["--experimental"]):
    # Note: although for synthetic the len is 20, vocab_size of 30 is still
    # working.
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
else:
    model_type = "SNN"


antigens = None  # None for the default 10 antigens from Absolut
if arguments["--epitopes"]:
    antigens = config.ANTIGEN_EPITOPES
if arguments["--experimental"]:
    antigens = ["HR2P"]
if arguments["--transformer"] and (not arguments["--experimental"]):
    antigens = ["3VRL", "1ADQ"]
if arguments["--overfitting_check"]:
    antigens = ["3VRL", "1ADQ", "3RAJ"]

# antigens = ["HR2B", "HR2P"]
# antigens = ["HELP"]
# antigens = config.ANTIGEN_EPITOPES
# antigens = [f"{ag}SIM" for ag in config.ANTIGENS] + [f"{ag}DIF" for ag in config.ANTIGENS]


# Standard parameters
if arguments["--transformer"] and arguments["--experimental"]:
    epochs = 500
    learning_rate = 1e-6
    optimizer_type = "Adam"
    momentum = 0.9
    weight_decay = 0
    batch_size = 128
    sample_train = None
elif arguments["--transformer"] and (not arguments["--experimental"]):
    epochs = 15  # 15
    learning_rate = 1e-6
    optimizer_type = "Adam"
    momentum = 0.9
    weight_decay = 0
    batch_size = 128
    sample_train = None
else:
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


def multiprocessing_wrapper_script_12d(
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

    subsample_size = None
    if arguments["--x10under"]:
        subsample_size = 0.1
    elif arguments["--x50under"]:
        subsample_size = 0.02

    pipe = pipelines.BinaryclassBindersPipeline(
        log_mlflow=False,
        save_model_mlflow=False,
        log_artifacts=LOG_ARTIFACTS,
        save_local=SAVE_LOCAL,
        local_dir=local_dir,
        subsample_size=subsample_size,
    )

    use_embeddings = None
    if arguments["--esm2b"]:
        use_embeddings = "esm2b"
    elif arguments["--antiberta2"]:
        use_embeddings = "antiberta2"

    pipe.step_1_process_data(
        ag_pos=ag_pos,
        ag_neg=ag_neg,
        sample_train=sample_train,
        batch_size=batch_size,
        shuffle_antigen_labels=shuffle_antigen_labels,
        load_from_miniabsolut=load_from_miniabsolut,
        load_from_miniabsolut_split_seed=load_from_miniabsolut_split_seed,
        use_embeddings=use_embeddings,
    )

    if only_generate_datasets:
        return

    if arguments["--esm2b"]:
        input_dim = 1280
    elif arguments["--antiberta2"]:
        input_dim = 1024
    else:
        input_dim = get_input_dim_from_agpos(ag_pos)

    pipe.step_2_train_model(
        input_dim=input_dim,
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
    # experiment = mlflow.set_experiment(experiment_id=experiment_id)
    experiment = None  # dummy

    if antigens is None:
        antigens: List[str] = config.ANTIGENS

    # Generate all datasets
    datasets = []
    for ag in antigens:
        datasets.append((f"{ag}_high", f"{ag}_looser"))
        datasets.append((f"{ag}_high", f"{ag}_95low"))
    
    print(f"Datasets: {datasets}")

    if TEST:
        epochs = 3
        learning_rate = 0.001
        optimizer_type = "Adam"
        momentum = 0.9
        weight_decay = 0
        batch_size = 64

        multiprocessing_wrapper_script_12d(
            experiment_id,
            "test",
            "3VRL_high",
            "3VRL_looser",
            None,  # sample_train
            0,
            None,
        )
        # multiprocessing_wrapper_script_12d(
        #     experiment_id,
        #     "test",
        #     "3VRL_high",
        #     "3VRL_95low",
        #     None,  # sample_train
        #     0,
        #     None,
        # )
        # multiprocessing_wrapper_script_12d(
        #     experiment_id,
        #     "test",
        #     "HR2B_high",
        #     "HR2B_95low",
        #     None,  # sample_train
        #     0,
        #     None,
        # )
        # multiprocessing_wrapper_script_12d(
        #     experiment_id,
        #     "test",
        #     "HR2P_high",
        #     "HR2P_95low",
        #     None,  # sample_train
        #     0,
        #     None,
        # )

    else:
        # Run batched multiprocessing
        for seed in seed_id:
            for i in range(0, len(datasets), num_processes):
                datasets_batch = datasets[i : i + num_processes]
                print(f"Processing batch {i} to {i+num_processes}: {datasets_batch}")
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
                                model_type,
                                model,
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
                                model_type,
                                model,
                            )
                            for ags in datasets_batch
                        ],
                    )
