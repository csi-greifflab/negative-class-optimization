# From Notebook 25, for Manuscript Section 1C.

import math
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

from NegativeClassOptimization import (config, datasets, ml, preprocessing,
                                       utils)

SKIP_LOADING_ERRORS = True
COMPUTE_CLOSEDSET_PERFORMANCE = False  # True > closedset, False > openset
COMPUTE_OPENSET_FROM_CLOSEDSET = True

# fp_results_closed = Path("data/closed_performance.tsv")
# fp_results_open = Path("data/openset_performance.tsv")
fp_results_closed = Path("data/closed_performance_experimental_data.tsv")
fp_results_open = Path("data/openset_performance_experimental_data.tsv")

# run_name = "dev-v0.1.2-3-with-replicates"
run_name = "dev-v0.1.3-expdata"

# antigens = config.ANTIGENS
antigens = ["HR2B", "HR2P", "HR2PSR", "HR2PIR"]  # Experimental dataset


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


# experiment_ids = ["11", "13", "14"]
experiment_ids = ["14"]
df = utils.MLFlowTaskAPI.mlflow_results_as_dataframe(  # type: ignore
    exp_list=experiment_ids,
    run_name=run_name,
    classify_tasks=True,
)


## Generate valid seed_id and split_id combinations
## According to hard-coded logic in script_12*.py
seed_split_ids = datasets.FrozenMiniAbsolutMLLoader.generate_seed_split_ids()

## Generate valid task pairings
task_types_for_openset = [
    # datasets.ClassificationTaskType.ONE_VS_NINE,
    datasets.ClassificationTaskType.HIGH_VS_95LOW,
    datasets.ClassificationTaskType.HIGH_VS_LOOSER,
]
task_type_combinations = list(product(task_types_for_openset, task_types_for_openset))

task_types_for_closedset = [
    # datasets.ClassificationTaskType.ONE_VS_NINE,
    datasets.ClassificationTaskType.HIGH_VS_95LOW,
    datasets.ClassificationTaskType.HIGH_VS_LOOSER,
    datasets.ClassificationTaskType.ONE_VS_ONE,
]

## Load data
loader = datasets.FrozenMiniAbsolutMLLoader(
    data_dir=Path("data/Frozen_MiniAbsolut_ML/")
)

## Compute
records = []
if COMPUTE_CLOSEDSET_PERFORMANCE:
    for ag1 in antigens:
        for ag2 in antigens:
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

                    if SKIP_LOADING_ERRORS:
                        try:
                            loader.load(task)
                        except ValueError:
                            print(f"Skipping {task} because it does not exist.")
                            continue
                    else:
                        loader.load(task)

                    # For experimental data, the state dict is loaded,
                    # not the model. We adjust it here.
                    if task.model is None:
                        assert task.state_dict is not None  # type: ignore
                        model = ml.load_model_from_state_dict(task.state_dict)  # type: ignore
                        task.model = model  # type: ignore
                    else:
                        model: nn.Module = task.model  # type: ignore

                    test_dataset: pd.DataFrame = task.test_dataset  # type: ignore

                    metrics = evaluate_model(model, test_dataset)
                    records.append(
                        {
                            "task": str(task),
                            **metrics,
                        }
                    )

                df_open = pd.DataFrame(records)
                df_open.to_csv(fp_results_closed, sep="\t", index=False)

# First run for compute closedset performance
# Then from those, seek all combinations,
# as openset. Check the paths!
elif COMPUTE_OPENSET_FROM_CLOSEDSET:
    
    assert COMPUTE_CLOSEDSET_PERFORMANCE is False, "Cannot compute openset and closedset at same time."

    df_closed = pd.read_csv(fp_results_closed, sep="\t")

    task_str_combinations = list(product(df_closed["task"], df_closed["task"]))

    for task_str_1, task_str_2 in task_str_combinations:
    
        task_1 = datasets.ClassificationTask.init_from_str(task_str_1)
        task_2 = datasets.ClassificationTask.init_from_str(task_str_2)
        
        if SKIP_LOADING_ERRORS:
            try:
                loader.load(task_1)
                loader.load(task_2)
            except ValueError:
                print(f"Skipping {task_1} because it does not exist.")
                continue
        else:
            loader.load(task_1)
            loader.load(task_2)

        print(f"Computing openset performance for {task_1} and {task_2}.")

        # For experimental data, the state dict is loaded,
        # not the model. We adjust it here.
        if task_1.model is None:
            assert task_1.state_dict is not None  # type: ignore
            model = ml.load_model_from_state_dict(task_1.state_dict)  # type: ignore
            task_1.model = model  # type: ignore
        else:
            model = task_1.model  # type: ignore

        test_dataset = task_2.test_dataset  # type: ignore

        if SKIP_LOADING_ERRORS:
            try:
                metrics = evaluate_model(model, test_dataset)
            except RuntimeError:
                print(f"Skipping {task_1} because it does not exist.")
                continue

        records.append(
            {
                "task_1": str(task_1),
                "task_2": str(task_2),
                **metrics,
            }
        )

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
