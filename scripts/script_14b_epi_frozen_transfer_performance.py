"""
Similar computations as for script 14, with specific computations for epitopes, a lot of extra
complexity introduced by the difference in test sets!
"""

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

from NegativeClassOptimization import (config, datasets, ml, pipelines,
                                       preprocessing, utils)


docopt_doc = """Compute metrics and save in convenient form.

Usage:
    script_14b_frozen_transfer_performance.py <closed> <open> <input_dir> <closed_out> <open_out>

Options:
    -h --help   Show help.
"""

arguments = docopt(docopt_doc, version="NCO")


SKIP_LOADING_ERRORS = False
SKIP_COMPUTED_TASKS = True
COMPUTE_CLOSEDSET_PERFORMANCE = bool(arguments["<closed>"] == "1")  # True > closedset, False > openset
COMPUTE_OPENSET_FROM_CLOSEDSET = bool(arguments["<open>"] == "1")  # True > openset from closedset

fp_loader = Path(arguments["<input_dir>"])


fp_results_closed = Path(arguments["<closed_out>"])
fp_results_open = Path(arguments["<open_out>"])


## TESTSET Choice
# "NonEpitope": Non-epitope specific sequences in positive and negative set, like the one used usually.
# "PositiveSet_Epitope", use only epitope specific sequences in positive set.
# "Positive_and_NegativeSet_Epitope", use epitope specific sequences in positive and negative set.
TESTSET = "Positive_and_NegativeSet_Epitope"

num_processes = 10

# fp_loader = Path("data/Frozen_MiniAbsolut_ML/")

# fp_results_closed = Path("data/closed_performance_epitopes.tsv")  # epitopes
# fp_results_open = Path("data/openset_performance_epitopes.tsv")  # epitopes
# fp_results_closed = Path("data/closed_performance_epitopes_pos.tsv")  # epitopes
# fp_results_open = Path("data/openset_performance_epitopes_pos.tsv")  # epitopes
# fp_results_closed = Path("data/closed_performance_epitopes_pos_and_neg.tsv")  # epitopes
# fp_results_open = Path("data/openset_performance_epitopes_pos_and_neg.tsv")  # epitopes


antigens = config.ANTIGEN_EPITOPES
antigens_2 = config.ANTIGENS + config.ANTIGEN_EPITOPES



## Generate valid seed_id and split_id combinations
## According to hard-coded logic in script_12*.py
seed_split_ids = datasets.FrozenMiniAbsolutMLLoader.generate_seed_split_ids(standard_split_only=True)


task_types_for_closedset = [
    datasets.ClassificationTaskType.HIGH_VS_95LOW,
    datasets.ClassificationTaskType.HIGH_VS_LOOSER,
    # datasets.ClassificationTaskType.ONE_VS_NINE,
    datasets.ClassificationTaskType.ONE_VS_ONE,
]

## Generate valid task pairings
task_types_for_openset = [
    datasets.ClassificationTaskType.ONE_VS_NINE,
    datasets.ClassificationTaskType.HIGH_VS_95LOW,
    datasets.ClassificationTaskType.HIGH_VS_LOOSER,
    datasets.ClassificationTaskType.ONE_VS_ONE,
]
task_type_combinations = list(product(task_types_for_openset, task_types_for_openset))

## Load data
loader = datasets.FrozenMiniAbsolutMLLoader(
    data_dir=Path(fp_loader)
)


def evaluate_model(
    model: torch.nn.Module,
    test_dataset: pd.DataFrame,
):
    """
    Evaluates a model on a test dataset.
    """
    with torch.no_grad():
        test_dataset = preprocessing.onehot_encode_df(test_dataset)
        X = np.stack(test_dataset["Slide_onehot"])  # type: ignore
        X = torch.from_numpy(X).float()
        y_pred = model(X).round().detach().numpy().reshape(-1)
        y_true = test_dataset["binds_a_pos_ag"].values
        metrics = ml.compute_binary_metrics(y_pred, y_true)
    return metrics



def compute_closedset_on_task(task) -> dict:
    """Compute record for closedset performance.

    Args:
        task (_type_): Loaded task.

    Raises:
        ValueError: 

    Returns: dict
    """

    # For experimental data, the state dict is loaded,
    # not the model. We adjust it here.
    if task.model is None:
        assert task.state_dict is not None  # type: ignore
        model = ml.load_model_from_state_dict(task.state_dict)  # type: ignore
        task.model = model  # type: ignore
    else:
        model: nn.Module = task.model  # type: ignore

    test_dataset = pipelines.get_test_dataset_for_epitope_analysis(task, test_set=TESTSET)
    metrics = evaluate_model(model, test_dataset)

    record = {
            "task": str(task),
            # "N_train_pos": (task.train_dataset["y"] == 1).sum(),
            # "N_train_neg": (task.train_dataset["y"] == 0).sum(),
            "N_test_pos": (test_dataset["y"] == 1).sum(),
            "N_test_neg": (test_dataset["y"] == 0).sum(),
            **metrics,
        }
    return record

if __name__ == "__main__":

    ## Compute
    try:
        df_closed = pd.read_csv(fp_results_closed, sep="\t")
    except FileNotFoundError:
        df_closed = pd.DataFrame()
    records = []
    if COMPUTE_CLOSEDSET_PERFORMANCE:
        tasks = []
        for ag1 in antigens:
            for ag2 in antigens_2:
                seed_id, split_id = (0, 42)
                for task_type in task_types_for_closedset:
                    
                    if (
                        task_type == datasets.ClassificationTaskType.ONE_VS_ONE
                        and ag1 == ag2
                    ):
                        continue

                    if (
                        TESTSET == "Positive_and_NegativeSet_Epitope"
                        and task_type == datasets.ClassificationTaskType.ONE_VS_ONE
                        and (ag1[-2:] != "E1" or ag2[-2:] != "E1") 
                    ):
                        continue

                    if (
                        task_type != datasets.ClassificationTaskType.ONE_VS_ONE
                        and ag1 != ag2
                    ):
                        continue

                    if task_type == datasets.ClassificationTaskType.ONE_VS_ONE:
                        task_i = datasets.ClassificationTask(
                            task_type=task_type,
                            ag_pos=ag1,
                            ag_neg=ag2,
                            seed_id=seed_id,
                            split_id=split_id,
                        )
                    else:
                        task_i = datasets.ClassificationTask(
                            task_type=task_type,
                            ag_pos=ag1,
                            ag_neg="auto",
                            seed_id=seed_id,
                            split_id=split_id,
                        )

                    if df_closed.shape[0] > 0 and str(task_i) in df_closed["task"].values and SKIP_COMPUTED_TASKS:
                        print(f"Skipping {task_i} because it was already computed.")
                        continue

                    if SKIP_LOADING_ERRORS:
                        try:
                            loader.load(task_i, load_train_dataset=False)
                        except ValueError:
                            print(f"Skipping {task_i} because it does not exist.")
                            continue
                    else:
                        loader.load(task_i, load_train_dataset=False)

                    tasks.append(task_i)


        batching = 1
        for i in range(0, len(tasks), batching*num_processes):
            records = []
            with multiprocessing.Pool(processes=num_processes) as pool:
                for record in pool.starmap(
                    compute_closedset_on_task,
                    [(task,) for task in tasks[i:i+batching*num_processes]],
                ):
                    if record is not None:
                        records.append(record) 
        
            df_closed_new = pd.DataFrame(records)
            df_closed = pd.concat([df_closed, df_closed_new], ignore_index=True)
            df_closed.to_csv(fp_results_closed, sep="\t", index=False)

        # TEST
        # task = tasks[0]
        # record = compute_closedset_on_task(task)
        # print(record)

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
                model = ml.load_model_from_state_dict(task_1.state_dict)  # type: ignore
                task_1.model = model  # type: ignore
            else:
                model = task_1.model  # type: ignore

            # test_dataset = task_2.test_dataset  # type: ignore
            test_dataset = pipelines.get_test_dataset_for_epitope_analysis(task_2, test_set=TESTSET)

            if SKIP_LOADING_ERRORS:
                try:
                    metrics = evaluate_model(model, test_dataset)
                except:
                    print(f"Error in {task_1} -> {task_2}.")
                    metrics = {}

            record = {
                "task_1": str(task_1),
                "task_2": str(task_2),
                "task_2_N_test_pos": (test_dataset["y"] == 1).sum(),
                "task_2_N_test_neg": (test_dataset["y"] == 0).sum(),
                **metrics,
            }
            return record


        records = []
        for i in range(0, len(pairs), 5*num_processes):
            with multiprocessing.Pool(processes=num_processes) as pool:
                for record in pool.starmap(
                    compute_openset_on_task_pair,
                    [(task_pair,) for task_pair in pairs[i:i+5*num_processes]],
                ):
                    records.append(record)
        
            df_open = pd.DataFrame(records)
            df_open.to_csv(fp_results_open, sep="\t", index=False)

        ## TEST
        # record = compute_openset_on_task_pair(pairs[0])

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
