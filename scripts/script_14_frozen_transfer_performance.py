# From Notebook 25, for Manuscript Section 1C.

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from itertools import product


import torch
import torch.nn as nn
import torch.nn.functional as F


from NegativeClassOptimization import utils, preprocessing, ml
from NegativeClassOptimization import config, datasets


def evaluate_model(
    model: torch.nn.Module,
    test_dataset: pd.DataFrame,
    ):
    """
    Evaluates a model on a test dataset.
    """
    with torch.no_grad():
        test_dataset = preprocessing.onehot_encode_df(test_dataset)
        X = np.stack(test_dataset["Slide_onehot"])
        X = torch.from_numpy(X).float()
        y_pred = model(X).round().detach().numpy().reshape(-1)
        y_true = test_dataset["binds_a_pos_ag"].values
        metrics = ml.compute_binary_metrics(y_pred, y_true)
    return metrics


experiment_ids = ["11", "13", "14"]
df = utils.MLFlowTaskAPI.mlflow_results_as_dataframe(experiment_ids, run_name="dev-v0.1.2-3-with-replicates", classify_tasks=True)


## Generate valid seed_id and split_id combinations
## According to hard-coded logic in script_12*.py
seed_split_ids = []
for seed in [0, 1, 2, 3]:
    split_id_default = 42
    seed_split_ids.append((seed, split_id_default))
for split_id in [0, 1, 2, 3, 4]:
    seed_id_default = 0
    seed_split_ids.append((seed_id_default, split_id))

## Generate valid task pairings
task_types_for_openset = [
    datasets.ClassificationTaskType.ONE_VS_NINE,
    datasets.ClassificationTaskType.HIGH_VS_95LOW,
    datasets.ClassificationTaskType.HIGH_VS_LOOSER,
]
task_type_combinations = list(product(task_types_for_openset, task_types_for_openset))

## Load data
loader = datasets.FrozenMiniAbsolutMLLoader(data_dir=Path("data/Frozen_MiniAbsolut_ML/"))

## Compute
records = []
for ag in config.ANTIGENS:
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

            model = task_1.model
            test_dataset = task_2.test_dataset

            metrics = evaluate_model(model, test_dataset)
            records.append({
                "task_1": str(task_1),
                "task_2": str(task_2),
                **metrics,
            })

        df_open = pd.DataFrame(records)
        df_open.to_csv("data/openset_performance.tsv", sep='\t', index=False)
