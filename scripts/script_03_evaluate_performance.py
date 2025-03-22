# From Notebook 25, for Manuscript Section 1C.

import math
import multiprocessing
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from docopt import docopt

from NegativeClassOptimization import (config, datasets, ml, preprocessing,
                                       utils)

docopt_doc = """Compute metrics and save in convenient form.

Usage:
    script_03_evaluate_performance.py <closed> <open> <input_dir> <closed_out> <open_out>
    script_03_evaluate_performance.py <closed> <open> <input_dir> <closed_out> <open_out> --experimental
    script_03_evaluate_performance.py <closed> <open> <input_dir> <closed_out> <open_out> --transformer
    script_03_evaluate_performance.py <closed> <open> <input_dir> <closed_out> <open_out> --experimental --transformer
    script_03_evaluate_performance.py <closed> <open> <input_dir> <closed_out> <open_out> --esm2b
    script_03_evaluate_performance.py <closed> <open> <input_dir> <closed_out> <open_out> --antiberta2
    script_03_evaluate_performance.py <closed> <open> <input_dir> <closed_out> <open_out> --experimental --esm2b
    script_03_evaluate_performance.py <closed> <open> <input_dir> <closed_out> <open_out> --experimental --antiberta2

Options:
    -h --help   Show help.
"""


arguments = docopt(docopt_doc, version="NCO")


run_parallel = False
SKIP_LOADING_ERRORS = True
SKIP_COMPUTED_TASKS = True
num_processes = 10


COMPUTE_CLOSEDSET_PERFORMANCE = bool(arguments["<closed>"] == "1")  # True > closedset, False > openset
COMPUTE_OPENSET_FROM_CLOSEDSET = bool(arguments["<open>"] == "1")  # True > openset from closedset

# Most cases None. 
# If "pos_epitope", use only epitope specific sequences in positive set.
# If "pos_and_epitope", use epitope specific sequences in positive and negative set.
USE_ALTERNATIVE_TESTSET = None 

fp_loader = Path(arguments["<input_dir>"])
# fp_loader = Path("data/Frozen_MiniAbsolut_ML/")
# fp_loader = Path("data/Frozen_MiniAbsolut_Linear_ML/")


fp_results_closed = Path(arguments["<closed_out>"])
fp_results_open = Path(arguments["<open_out>"])

# fp_results_closed = Path("data/closed_performance.tsv")
# fp_results_open = Path("data/openset_performance.tsv")

# fp_results_closed = Path("data/closed_performance_logistic.tsv")
# fp_results_open = Path("data/openset_performance_logistic.tsv")

# fp_results_closed = Path("data/closed_performance_experimental_data.tsv")
# fp_results_open = Path("data/openset_performance_experimental_data.tsv")

# fp_results_closed = Path("data/closed_performance_epitopes.tsv")  # epitopes
# fp_results_open = Path("data/openset_performance_epitopes.tsv")  # epitopes
# fp_results_closed = Path("data/closed_performance_epitopes_with_sizes.tsv")  # for debugging
# fp_results_closed = Path("data/closed_performance_epitopes_pos.tsv")  # epitopes
# fp_results_open = Path("data/openset_performance_epitopes_pos.tsv")  # epitopes

# fp_results_closed = Path("data/closed_performance_similar_wcounts.tsv")  # similar antigens
# fp_results_open = Path("data/openset_performance_similar.tsv")  # dissimilar antigens
# fp_results_closed = Path("data/closed_performance_dissimilar.tsv")  # similar antigens
# fp_results_open = Path("data/openset_performance_dissimilar.tsv")  # dissimilar antigens


## For shuffled
# fp_loader = Path("data/Frozen_MiniAbsolut_ML_shuffled/")
# fp_results_closed = Path("data/Frozen_MiniAbsolut_ML_shuffled/closed_performance.tsv")
# fp_results_open = Path("data/Frozen_MiniAbsolut_ML_shuffled/openset_performance.tsv")
# fp_results_closed = Path("data/Frozen_MiniAbsolut_ML_shuffled/closed_performance_experimental_data.tsv")
# fp_results_open = Path("data/Frozen_MiniAbsolut_ML_shuffled/openset_performance_experimental_data.tsv")

antigens = config.ANTIGENS
# antigens = ["HR2B", "HR2P", "HR2PSR", "HR2PIR"]  # Experimental dataset

if arguments["--experimental"] and not arguments["--transformer"]:
    antigens = ["HR2P", "HR2PSR", "HR2PIR"]

if arguments["--transformer"] and arguments["--experimental"]:
    antigens = ["HR2P"]

if arguments["--transformer"] and not arguments["--experimental"]:
    antigens = ["3VRL", "1ADQ", "3RAJ", "1WEJ"]

antigens_2 = antigens[:]  # in most cases, exception for epitope-based analysis
## Epitope-based
# antigens = ["1WEJE1", "1OB1E1", "1H0DE1"]
# antigens_2 = config.ANTIGENS + config.ANTIGEN_EPITOPES
## Similar antigens
# antigens = [ag + "SIM" for ag in config.ANTIGENS]
# antigens_2 = [ag + "SIM" for ag in config.ANTIGENS]
## Dissimilar antigens
# antigens = [ag + "DIF" for ag in config.ANTIGENS]
# antigens_2 = [ag + "DIF" for ag in config.ANTIGENS]

def evaluate_model(
    model: torch.nn.Module,
    test_dataset: pd.DataFrame,
):
    """
    Evaluates a model on a test dataset.
    """
    with torch.no_grad():
        
        if arguments["--esm2b"] or arguments["--antiberta2"]:
            X = np.stack(test_dataset["X"].str[1:-1].str.split(", ")).astype(float)  # type: ignore
            X = torch.from_numpy(X)
            X = X.float()
        else:
            test_dataset = preprocessing.onehot_encode_df(test_dataset)
            X = np.stack(test_dataset["Slide_onehot"])  # type: ignore
            X = torch.from_numpy(X).float()

        # For transformer models, we need to adapt the input
        if arguments["--transformer"]:
            X = model.module.adapt_input(X)
            # Torch type Long Int
            X = X.long()

        y_pred = model(X).round().detach().numpy().reshape(-1)
        y_true = test_dataset["binds_a_pos_ag"].values
        metrics = ml.compute_binary_metrics(y_pred, y_true)
    return metrics

def load_transformer(state_dict, is_experimental=False):
    import json
    with open(Path(arguments["<input_dir>"]) / "parameter_set.json", "r") as f:
        parameters = json.load(f)
        if is_experimental:
            parameters["transformer_type"] = "experimental_dataset"
        else:
            parameters["transformer_type"] = "absolut_dataset"
    model = ml.load_model_from_state_dict(state_dict, params_dict=parameters)
    return model


## Generate valid seed_id and split_id combinations
## According to hard-coded logic in script_12*.py
seed_split_ids = datasets.FrozenMiniAbsolutMLLoader.generate_seed_split_ids()

if arguments["--experimental"] and arguments["--transformer"]:
    seed_split_ids = [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 42),
    ]
elif arguments["--transformer"] and not arguments["--experimental"]:
    seed_split_ids = [
        (0, 0),
        (0, 1),
        (0, 42),
    ]

## Generate valid task pairings

if arguments["--experimental"] and arguments["--transformer"]:
    task_types_for_closedset = [
        datasets.ClassificationTaskType.HIGH_VS_95LOW,
        datasets.ClassificationTaskType.HIGH_VS_LOOSER,
    ]
    task_types_for_openset = [
        datasets.ClassificationTaskType.HIGH_VS_95LOW,
        datasets.ClassificationTaskType.HIGH_VS_LOOSER,
    ]
else:
    task_types_for_closedset = [
        datasets.ClassificationTaskType.HIGH_VS_95LOW,
        datasets.ClassificationTaskType.HIGH_VS_LOOSER,
        datasets.ClassificationTaskType.ONE_VS_NINE,
        datasets.ClassificationTaskType.ONE_VS_ONE,
    ]
    task_types_for_openset = [
        datasets.ClassificationTaskType.ONE_VS_ONE,
        datasets.ClassificationTaskType.ONE_VS_NINE,
        datasets.ClassificationTaskType.HIGH_VS_95LOW,
        datasets.ClassificationTaskType.HIGH_VS_LOOSER,
    ]

task_type_combinations = list(product(task_types_for_openset, task_types_for_openset))

## Load data
loader = datasets.FrozenMiniAbsolutMLLoader(
    data_dir=Path(fp_loader)
)

## Compute
try:
    df_closed = pd.read_csv(fp_results_closed, sep="\t")
except FileNotFoundError or pd.errors.EmptyDataError:
    df_closed = pd.DataFrame()
records = []
if COMPUTE_CLOSEDSET_PERFORMANCE:
    for ag1 in antigens:
        for ag2 in antigens_2:
            for seed_id, split_id in seed_split_ids:
                for task_type in task_types_for_closedset:
                    if (
                        task_type == datasets.ClassificationTaskType.ONE_VS_ONE
                        and ag1 == ag2
                    ):
                        continue
                    if (
                        task_type != datasets.ClassificationTaskType.ONE_VS_ONE
                        and ag1 != ag2
                    ):
                        continue

                    if task_type == datasets.ClassificationTaskType.ONE_VS_ONE:
                        task = datasets.ClassificationTask(
                            task_type=task_type,
                            ag_pos=ag1,
                            ag_neg=ag2,
                            seed_id=seed_id,
                            split_id=split_id,
                        )
                    else:
                        task = datasets.ClassificationTask(
                            task_type=task_type,
                            ag_pos=ag1,
                            ag_neg="auto",
                            seed_id=seed_id,
                            split_id=split_id,
                        )

                    if df_closed.shape[0] > 0 and str(task) in df_closed["task"].values and SKIP_COMPUTED_TASKS:
                        print(f"Skipping {task} because it was already computed.")
                        continue

                    if SKIP_LOADING_ERRORS:
                        try:
                            loader.load(task, load_train_dataset=False)
                        except ValueError:
                            print(f"Skipping {task} because it does not exist.")
                            continue
                    else:
                        loader.load(task, load_train_dataset=False)

                    # For experimental data, the state dict is loaded,
                    # not the model. We adjust it here.
                    if task.model is None:
                        assert task.state_dict is not None  # type: ignore
                        if "module.conv1.weight" in task.state_dict.keys() and arguments["--experimental"]:
                            model = ml.load_model_from_state_dict(task.state_dict, input_dim=20*21)
                        elif arguments["--transformer"] and arguments["--experimental"]:
                            model = load_transformer(task.state_dict, is_experimental=True)
                        elif arguments["--transformer"] and not arguments["--experimental"]:
                            model = load_transformer(task.state_dict, is_experimental=False)
                        else:
                            model = ml.load_model_from_state_dict(task.state_dict)  # type: ignore
                        task.model = model  # type: ignore
                    else:
                        model: nn.Module = task.model  # type: ignore

                    test_dataset: pd.DataFrame = task.test_dataset  # type: ignore
                    metrics = evaluate_model(model, test_dataset)

                    records.append(
                        {
                            "task": str(task),
                            "N_pos": (test_dataset["y"] == 1).sum(),
                            "N_neg": (test_dataset["y"] == 0).sum(),
                            **metrics,
                        }
                    )

                df_closed = pd.DataFrame(records)
                df_closed.to_csv(fp_results_closed, sep="\t", index=False)

# First run for compute closedset performance
# Then from those, seek all combinations,
# as openset. Check the paths!
elif COMPUTE_OPENSET_FROM_CLOSEDSET:
    
    assert COMPUTE_CLOSEDSET_PERFORMANCE is False, "Cannot compute openset and closedset at same time."

    df_closed = pd.read_csv(fp_results_closed, sep="\t")
    try:
        df_open = pd.read_csv(fp_results_open, sep="\t")
    except FileNotFoundError:
        df_open = pd.DataFrame()

    task_str_combinations = list(product(df_closed["task"], df_closed["task"]))
    print(f"Num combinations: {len(task_str_combinations)}")

    pairs = []
    for task_str_1, task_str_2 in task_str_combinations:
    
        task_1 = datasets.ClassificationTask.init_from_str(task_str_1)
        task_2 = datasets.ClassificationTask.init_from_str(task_str_2)
        
        if (
            df_open.shape[0] > 0 
            and str(task_1) in df_open["task_1"].values 
            and str(task_2) in df_open["task_2"].values 
            and SKIP_COMPUTED_TASKS
        ):
            print(f"Skipping {task_1} -> {task_2} because it was already computed.")
            continue

        if task_1.task_type not in task_types_for_openset or task_2.task_type not in task_types_for_openset:
            print(f"Skipping {task_1} -> {task_2} because not an openset task.")
            continue

        if task_1.ag_pos != task_2.ag_pos:
            print(f"Skipping {task_1} -> {task_2} because antigens do not match.")
            continue

        print(f"Computing openset performance for {task_1} and {task_2}.")

        pair = (task_1, task_2)
        pairs.append(pair)

    def compute_openset_on_task_pair(task_pair):

        task_1 = task_pair[0]
        task_2 = task_pair[1]

        if SKIP_LOADING_ERRORS:
            try:
                loader.load(task_1)
                loader.load(task_2)
            except ValueError:
                print(f"Skipping {task_1} because it does not exist.")
                return {}
        else:
            loader.load(task_1)
            loader.load(task_2)

        # For experimental data, the state dict is loaded,
        # not the model. We adjust it here.
        if task_1.model is None:
            assert task_1.state_dict is not None  # type: ignore
            if arguments["--transformer"] and arguments["--experimental"]:
                model = load_transformer(task_1.state_dict, is_experimental=True)
            elif arguments["--transformer"] and not arguments["--experimental"]:
                model = load_transformer(task_1.state_dict, is_experimental=False)
            else:
                model = ml.load_model_from_state_dict(task_1.state_dict)  # type: ignore
            task_1.model = model  # type: ignore
        else:
            model = task_1.model  # type: ignore

        test_dataset = task_2.test_dataset  # type: ignore

        if SKIP_LOADING_ERRORS:
            try:
                metrics = evaluate_model(model, test_dataset)
            except RuntimeError:
                print(f"Error in {task_1} -> {task_2}.")
                return None
        else:
            metrics = evaluate_model(model, test_dataset)

        record = {
            "task_1": str(task_1),
            "task_2": str(task_2),
            **metrics,
        }
        return record


    records = []
    for i in range(0, len(pairs), num_processes):
        if run_parallel:
            with multiprocessing.Pool(processes=num_processes) as pool:
                for record in pool.starmap(
                    compute_openset_on_task_pair,
                    [(task_pair,) for task_pair in pairs[i : i + num_processes]],
                ):
                    if record is not None:
                        records.append(record)
        else:
            for task_pair in pairs[i : i + num_processes]:
                record = compute_openset_on_task_pair(task_pair)
                if record is not None:
                    records.append(record)

        df_open = pd.DataFrame(records)
        df_open.to_csv(fp_results_open, sep="\t", index=False)

else:
    for ag in antigens:
        for seed_id, split_id in seed_split_ids:
            for task_type_1, task_type_2 in task_type_combinations:
                task_1 = datasets.ClassificationTask(
                    task_type=task_type_1,
                    ag_pos=ag,
                    ag_neg="auto",
                    seed_id=seed_id,
                    split_id=split_id,
                )
                task_2 = datasets.ClassificationTask(
                    task_type=task_type_2,
                    ag_pos=ag,
                    ag_neg="auto",
                    seed_id=seed_id,
                    split_id=split_id,
                )
                loader.load(task_1)
                loader.load(task_2)

                # For experimental data, the state dict is loaded,
                # not the model. We adjust it here.
                if task_1.model is None:
                    assert task_1.state_dict is not None  # type: ignore
                    model = ml.load_model_from_state_dict(task_1.state_dict)  # type: ignore
                    task_1.model = model  # type: ignore
                else:
                    model = task_1.model  # type: ignore

                test_dataset = task_2.test_dataset  # type: ignore

                metrics = evaluate_model(model, test_dataset)
                records.append(
                    {
                        "task_1": str(task_1),
                        "task_2": str(task_2),
                        **metrics,
                    }
                )

            df_open = pd.DataFrame(records)
            df_open.to_csv(fp_results_open, sep="\t", index=False)
