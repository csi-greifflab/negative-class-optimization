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

from NegativeClassOptimization import (config, datasets, ml, pipelines,
                                       preprocessing, utils)

SKIP_LOADING_ERRORS = True
SKIP_COMPUTED_TASKS = True

## TESTSET Choice
# "NonEpitope": Non-epitope specific sequences in positive and negative set, like the one used usually.
# "PositiveSet_Epitope", use only epitope specific sequences in positive set.
# "Positive_and_NegativeSet_Epitope", use epitope specific sequences in positive and negative set.
TESTSET = "NonEpitope"

num_processes = 10

fp_loader = Path("data/Frozen_MiniAbsolut_ML/")

fp_results = Path("data/jsd_epitopes.tsv")  # epitopes


antigens = config.ANTIGEN_EPITOPES
antigens_2 = config.ANTIGENS + config.ANTIGEN_EPITOPES

## Generate valid seed_id and split_id combinations
## According to hard-coded logic in script_12*.py
seed_split_ids = datasets.FrozenMiniAbsolutMLLoader.generate_seed_split_ids(standard_split_only=True)


task_types = [
    datasets.ClassificationTaskType.HIGH_VS_95LOW,
    datasets.ClassificationTaskType.HIGH_VS_LOOSER,
    datasets.ClassificationTaskType.ONE_VS_NINE,
    datasets.ClassificationTaskType.ONE_VS_ONE,
]

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



def compute_jsd(task) -> dict:
    """Compute record for jsd.

    Args:
        task (_type_): Loaded task.

    Raises:
        ValueError: 

    Returns: dict
    """

    test_dataset = pipelines.get_test_dataset_for_epitope_analysis(task, test_set=TESTSET)

    try:
        slides_pos = set(test_dataset.loc[test_dataset["y"] == 1]["Slide"])
        slides_neg = set(test_dataset.loc[test_dataset["y"] == 0]["Slide"])

        jsd = utils.jensen_shannon_divergence_slides(slides_pos, slides_neg)
    except:
        jsd = np.nan

    record = {
            "task": str(task),
            "jsd": jsd 
        }
    return record

## Compute
if __name__ == "__main__":
    
    try:
        df_closed = pd.read_csv(fp_results, sep="\t")
    except FileNotFoundError:
        df_closed = pd.DataFrame()
    records = []

    tasks = []
    for ag1 in antigens:
        for ag2 in antigens_2:
            seed_id, split_id = (0, 42)
            for task_type in task_types:
                
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
                compute_jsd,
                [(task,) for task in tasks[i:i+batching*num_processes]],
            ):
                if record is not None:
                    records.append(record) 

        df_closed_new = pd.DataFrame(records)
        df_closed = pd.concat([df_closed, df_closed_new], ignore_index=True)
        df_closed.to_csv(fp_results, sep="\t", index=False)

    ## TEST
    # task = tasks[0]
    # record = compute_jsd(task)
    # print(record)
