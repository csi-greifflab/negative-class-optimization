"""
Functions from 07e* notebooks to be used throught others.
"""
import json
import math
from itertools import permutations
from pathlib import Path

import logomaker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

from NegativeClassOptimization import (
    config,
    datasets,
    ml,
    preprocessing,
    utils,
    visualisations,
)

task_types = (
    datasets.ClassificationTaskType.ONE_VS_ONE,
    datasets.ClassificationTaskType.ONE_VS_NINE,
    datasets.ClassificationTaskType.HIGH_VS_95LOW,
    datasets.ClassificationTaskType.HIGH_VS_LOOSER,
)

loader = datasets.FrozenMiniAbsolutMLLoader(
    data_dir=Path("../data/Frozen_MiniAbsolut_ML/")
)


def task_generator(task_types=task_types, loader=loader):
    """
    Generate tasks for which to compute attributions.
    """
    # Generate 1v1

    # Generate the rest: 1v9, high vs 95low, high vs looser
    seed_split_ids = datasets.FrozenMiniAbsolutMLLoader.generate_seed_split_ids()
    for ag_1 in config.ANTIGENS:
        for seed_id, split_id in seed_split_ids:
            for task_type in task_types:
                if task_type == datasets.ClassificationTaskType.ONE_VS_ONE:
                    continue

                task = datasets.ClassificationTask(
                    task_type=task_type,
                    ag_pos=ag_1,
                    ag_neg="auto",
                    seed_id=seed_id,
                    split_id=split_id,
                )
                yield task

    if datasets.ClassificationTaskType.ONE_VS_ONE in task_types:
        seed_split_ids = datasets.FrozenMiniAbsolutMLLoader.generate_seed_split_ids()
        for ag_1, ag_2 in permutations(config.ANTIGENS, r=2):
            for seed_id, split_id in seed_split_ids:
                task = datasets.ClassificationTask(
                    task_type=datasets.ClassificationTaskType.ONE_VS_ONE,
                    ag_pos=ag_1,
                    ag_neg=ag_2,
                    seed_id=seed_id,
                    split_id=split_id,
                )
                yield task


def get_miniabsolut_dataframes(task, load_energy_contributions=False):
    """Load the dataframes based on MiniAbsolut, from task
    These dataframes contain extra info, such as binding energies.
    """
    if task.split_id == 42:
        miniabsolut_path = Path("../data/MiniAbsolut")
    elif task.split_id in [0, 1, 2, 3, 4]:
        miniabsolut_path = Path(
            f"../data/MiniAbsolut_Splits/MiniAbsolut_Seed{task.split_id}"
        )

    ag_pos_dir = miniabsolut_path / task.ag_pos  # type:ignore
    if load_energy_contributions:
        ag_pos_dir = ag_pos_dir / "energy_contributions"
        suffix = "_absolut_energy_contributions"
        header = 1
    else:
        suffix = ""
        header = 0

    df_pos = pd.read_csv(
        ag_pos_dir / f"high_test_5000{suffix}.tsv", sep="\t", header=header
    )
    df_pos["class"] = "positive"

    if task.task_type == datasets.ClassificationTaskType.HIGH_VS_95LOW:
        df_neg = pd.read_csv(
            ag_pos_dir / f"95low_test_5000{suffix}.tsv", sep="\t", header=header
        )
        df_neg["class"] = "negative"
    elif task.task_type == datasets.ClassificationTaskType.HIGH_VS_LOOSER:
        df_neg = pd.read_csv(
            ag_pos_dir / f"looserX_test_5000{suffix}.tsv", sep="\t", header=header
        )
        df_neg["class"] = "negative"
    elif task.task_type in (
        datasets.ClassificationTaskType.ONE_VS_ONE,
        datasets.ClassificationTaskType.ONE_VS_NINE,
    ):
        # raise ValueError("Not implemented.")
        df_neg = pd.DataFrame()

    df = pd.concat([df_pos, df_neg], axis=0)  # type:ignore

    return df


def compute_for_logits_associations(task, return_df=False):
    assert hasattr(task, "attributions"), "Task has no attributions."

    # Get energies
    df = get_miniabsolut_dataframes(task)
    df["class"] = df["y_true"].map({1: "positive", 0: "negative"})

    # Add logits to df
    df_attr = pd.DataFrame.from_records(task.attributions)
    if "expits" in df_attr.columns:
        cols_to_merge = [
            "slide",
            "logits",
            "expits",
            "y_pred",
            "y_true",
            "is_pred_correct",
        ]
    else:
        cols_to_merge = ["slide", "logits"]
    df = pd.merge(
        df, df_attr[cols_to_merge], left_on="Slide", right_on="slide", how="left"
    )

    # Compute correlation and p-val with scipy for all and positive class
    r, p = pearsonr(df["logits"], df["Energy"])
    r_pos, p_pos = pearsonr(
        df[df["class"] == "positive"]["logits"], df[df["class"] == "positive"]["Energy"]
    )
    r_neg, p_neg = pearsonr(
        df[df["class"] == "negative"]["logits"], df[df["class"] == "negative"]["Energy"]
    )

    # Compute slope for all and positive class
    slope, intercept = np.polyfit(df["logits"], df["Energy"], 1)
    slope_pos, intercept_pos = np.polyfit(
        df[df["class"] == "positive"]["logits"],
        df[df["class"] == "positive"]["Energy"],
        1,
    )

    record = {
        "task_type": task.task_type.to_str(),
        "ag_pos": task.ag_pos,
        "ag_neg": task.ag_neg,
        "seed_id": task.seed_id,
        "split_id": task.split_id,
        "r": r,
        "p": p,
        "r2": r**2,
        "r_pos": r_pos,
        "r_pos2": r_pos**2,
        "p_pos": p_pos,
        "r_neg": r_neg,
        "r_neg2": r_neg**2,
        "p_neg": p_neg,
        "slope": slope,
        "intercept": intercept,
        "slope_pos": slope_pos,
        "intercept_pos": intercept_pos,
        "logits_mean": df["logits"].mean(),
        "logits_std": df["logits"].std(),
        "Energy_mean": df["Energy"].mean(),
        "Energy_std": df["Energy"].std(),
        "logits_pos_mean": df[df["class"] == "positive"]["logits"].mean(),
        "logits_pos_std": df[df["class"] == "positive"]["logits"].std(),
        "Energy_pos_mean": df[df["class"] == "positive"]["Energy"].mean(),
        "Energy_pos_std": df[df["class"] == "positive"]["Energy"].std(),
        "logits_neg_mean": df[df["class"] == "negative"]["logits"].mean(),
        "logits_neg_std": df[df["class"] == "negative"]["logits"].std(),
        "Energy_neg_mean": df[df["class"] == "negative"]["Energy"].mean(),
        "Energy_neg_std": df[df["class"] == "negative"]["Energy"].std(),
    }

    return record


class AA_Index:
    """Class for representing an amino acid and index couple."""

    def __init__(self, aa: str, index: int):
        self.aa = aa
        self.index = index

    @staticmethod
    def from_str(s: str):
        return AA_Index(s[0], int(s[2:]))

    def __eq__(self, other):
        return self.aa == other.aa and self.index == other.index

    def __repr__(self):
        return f"{self.aa}:{self.index}"

    def __str__(self):
        return f"{self.aa}:{self.index}"

    def isin_str(self, s: str):
        return s[self.index] == self.aa


## Loading energy contributions
### COPIED FROM NB 25b (REFACTOR LATER)


def get_attr_from_records(records, attributor_name, y_true):
    """Get the attributions for a given attributor and y_true."""
    attr = np.stack(
        list(
            map(
                lambda x: np.array(x[attributor_name]).reshape((11, 20)),
                filter(lambda x: x["y_true"] in y_true, records),
            )
        )
    )

    return attr


def get_onehotstack_from_records(records, y_true=(0, 1)):
    """Get the onehot stack from the records."""
    slides = [record["slide"] for record in records if record["y_true"] in y_true]
    onehots = [preprocessing.onehot_encode(slide) for slide in slides]
    onehot_stack = np.stack(onehots)
    return onehot_stack


def load_energy_contributions_from_task_linear_version(
    task, loader=loader, load_slide_df=False
):
    # Get energy contributions and attributions
    df = get_miniabsolut_dataframes(task, load_energy_contributions=True)
    energy_dict = df.set_index("Slide").to_dict(orient="index")
    for slide in energy_dict.keys():
        energy_dict[slide]["energies"] = utils.extract_contributions_from_string(
            energy_dict[slide]["contribPerAAparaBind"]
        )[1]
        energy_dict[slide]["energies_fold"] = utils.extract_contributions_from_string(
            energy_dict[slide]["contribPerAAparaFold"]
        )[1]
        energy_dict[slide]["energies_total"] = (
            np.array(energy_dict[slide]["energies"])
            + np.array(energy_dict[slide]["energies_fold"])
        ).tolist()

    # Get attributions per amino acid
    task = loader.load(task, attributions_toload="v0.1.2-3")
    attributor_name = "weights"
    attr_stack = get_attr_from_records(
        task.attributions, attributor_name, (0, 1)  # type: ignore
    )  # Nx11x20
    onehot_stack = get_onehotstack_from_records(
        task.attributions, (0, 1)  # type: ignore
    )  # Nx220
    attr_aa = attr_stack[onehot_stack.reshape((-1, 11, 20)) == 1].reshape(
        -1, 11
    )  # Nx11
    attr_dict = {
        record["slide"]: {**record, **{"attribution_existingaa": attr_aa[i, :]}}
        for i, record in enumerate(task.attributions)  # type: ignore
    }

    # Combine dictionaries
    slide_records = []
    for slide in energy_dict.keys():
        dataset_class = energy_dict[slide]["class"]
        energies = energy_dict[slide]["energies"]
        energies_fold = energy_dict[slide]["energies_fold"]
        energies_total = energy_dict[slide]["energies_total"]
        attrs = attr_dict[slide]["attribution_existingaa"]

        # Correlation between energy and attribution with scipy
        r, p = pearsonr(energies, attrs)
        r_fold, p_fold = pearsonr(energies_fold, attrs)
        r_total, p_total = pearsonr(energies_total, attrs)
        slide_records.append(
            {
                "slide": slide,
                "class": dataset_class,
                "energies": energies,
                "attributions": attrs,
                "r": r,
                "p": p,
                "r_fold": r_fold,
                "p_fold": p_fold,
                "r_total": r_total,
                "p_total": p_total,
            }
        )
    slide_df = pd.DataFrame.from_records(slide_records)

    mean = slide_df["r"].mean()
    std = slide_df["r"].std()
    mean_pos = slide_df[slide_df["class"] == "positive"]["r"].mean()
    std_pos = slide_df[slide_df["class"] == "positive"]["r"].std()
    mean_neg = slide_df[slide_df["class"] == "negative"]["r"].mean()
    std_neg = slide_df[slide_df["class"] == "negative"]["r"].std()

    mean_fold = slide_df["r_fold"].mean()
    std_fold = slide_df["r_fold"].std()
    mean_pos_fold = slide_df[slide_df["class"] == "positive"]["r_fold"].mean()
    std_pos_fold = slide_df[slide_df["class"] == "positive"]["r_fold"].std()
    mean_neg_fold = slide_df[slide_df["class"] == "negative"]["r_fold"].mean()
    std_neg_fold = slide_df[slide_df["class"] == "negative"]["r_fold"].std()

    mean_total = slide_df["r_total"].mean()
    std_total = slide_df["r_total"].std()
    mean_pos_total = slide_df[slide_df["class"] == "positive"]["r_total"].mean()
    std_pos_total = slide_df[slide_df["class"] == "positive"]["r_total"].std()
    mean_neg_total = slide_df[slide_df["class"] == "negative"]["r_total"].mean()
    std_neg_total = slide_df[slide_df["class"] == "negative"]["r_total"].std()

    stats = {
        "task_type": task.task_type.to_str(),
        "ag_pos": task.ag_pos,
        "ag_neg": task.ag_neg,
        "seed_id": task.seed_id,
        "split_id": task.split_id,
        "mean": mean,
        "std": std,
        "mean_pos": mean_pos,
        "std_pos": std_pos,
        "mean_neg": mean_neg,
        "std_neg": std_neg,
        "mean_fold": mean_fold,
        "std_fold": std_fold,
        "mean_pos_fold": mean_pos_fold,
        "std_pos_fold": std_pos_fold,
        "mean_neg_fold": mean_neg_fold,
        "std_neg_fold": std_neg_fold,
        "mean_total": mean_total,
        "std_total": std_total,
        "mean_pos_total": mean_pos_total,
        "std_pos_total": std_pos_total,
        "mean_neg_total": mean_neg_total,
        "std_neg_total": std_neg_total,
    }
    if load_slide_df:
        return stats, slide_df
    else:
        return stats


def load_energy_contributions_from_task_nonlinear_version(task, load_slide_df=False):
    # Get energy contributions and attributions
    df = get_miniabsolut_dataframes(task, load_energy_contributions=True)
    energy_dict = df.set_index("Slide").to_dict(orient="index")
    for slide in energy_dict.keys():
        energy_dict[slide]["energies"] = utils.extract_contributions_from_string(
            energy_dict[slide]["contribPerAAparaBind"]
        )[1]
        energy_dict[slide]["energies_fold"] = utils.extract_contributions_from_string(
            energy_dict[slide]["contribPerAAparaFold"]
        )[1]
        energy_dict[slide]["energies_total"] = (
            np.array(energy_dict[slide]["energies"])
            + np.array(energy_dict[slide]["energies_fold"])
        ).tolist()

    # Get attributions per amino acid
    task = loader.load(task, attributions_toload="v2.0-2")
    attributor_name = "DeepLIFT_LOCAL_v2.0-2"
    attr_stack = get_attr_from_records(
        task.attributions, attributor_name, (0, 1)  # type: ignore
    )  # Nx11x20 # type: ignore
    onehot_stack = get_onehotstack_from_records(
        task.attributions, (0, 1)  # type: ignore
    )  # Nx220 # type: ignore
    attr_aa = attr_stack[onehot_stack.reshape((-1, 11, 20)) == 1].reshape(
        -1, 11
    )  # Nx11
    attr_dict = {
        record["slide"]: {**record, **{"attribution_existingaa": attr_aa[i, :]}}
        for i, record in enumerate(task.attributions)  # type: ignore
    }  # type: ignore

    # Combine dictionaries
    slide_records = []
    for slide in energy_dict.keys():
        dataset_class = energy_dict[slide]["class"]
        energies = energy_dict[slide]["energies"]
        energies_fold = energy_dict[slide]["energies_fold"]
        energies_total = energy_dict[slide]["energies_total"]
        attrs = attr_dict[slide]["attribution_existingaa"]

        # Correlation between energy and attribution with scipy
        r, p = pearsonr(energies, attrs)
        r_fold, p_fold = pearsonr(energies_fold, attrs)
        r_total, p_total = pearsonr(energies_total, attrs)
        slide_records.append(
            {
                "slide": slide,
                "class": dataset_class,
                "energies": energies,
                "attributions": attrs,
                "r": r,
                "p": p,
                "r_fold": r_fold,
                "p_fold": p_fold,
                "r_total": r_total,
                "p_total": p_total,
            }
        )
    slide_df = pd.DataFrame.from_records(slide_records)

    mean = slide_df["r"].mean()
    std = slide_df["r"].std()
    mean_pos = slide_df[slide_df["class"] == "positive"]["r"].mean()
    std_pos = slide_df[slide_df["class"] == "positive"]["r"].std()
    mean_neg = slide_df[slide_df["class"] == "negative"]["r"].mean()
    std_neg = slide_df[slide_df["class"] == "negative"]["r"].std()

    mean_fold = slide_df["r_fold"].mean()
    std_fold = slide_df["r_fold"].std()
    mean_pos_fold = slide_df[slide_df["class"] == "positive"]["r_fold"].mean()
    std_pos_fold = slide_df[slide_df["class"] == "positive"]["r_fold"].std()
    mean_neg_fold = slide_df[slide_df["class"] == "negative"]["r_fold"].mean()
    std_neg_fold = slide_df[slide_df["class"] == "negative"]["r_fold"].std()

    mean_total = slide_df["r_total"].mean()
    std_total = slide_df["r_total"].std()
    mean_pos_total = slide_df[slide_df["class"] == "positive"]["r_total"].mean()
    std_pos_total = slide_df[slide_df["class"] == "positive"]["r_total"].std()
    mean_neg_total = slide_df[slide_df["class"] == "negative"]["r_total"].mean()
    std_neg_total = slide_df[slide_df["class"] == "negative"]["r_total"].std()

    stats = {
        "task_type": task.task_type.to_str(),
        "ag_pos": task.ag_pos,
        "ag_neg": task.ag_neg,
        "seed_id": task.seed_id,
        "split_id": task.split_id,
        "mean": mean,
        "std": std,
        "mean_pos": mean_pos,
        "std_pos": std_pos,
        "mean_neg": mean_neg,
        "std_neg": std_neg,
        "mean_fold": mean_fold,
        "std_fold": std_fold,
        "mean_pos_fold": mean_pos_fold,
        "std_pos_fold": std_pos_fold,
        "mean_neg_fold": mean_neg_fold,
        "std_neg_fold": std_neg_fold,
        "mean_total": mean_total,
        "std_total": std_total,
        "mean_pos_total": mean_pos_total,
        "std_pos_total": std_pos_total,
        "mean_neg_total": mean_neg_total,
        "std_neg_total": std_neg_total,
    }
    if load_slide_df:
        return stats, slide_df
    else:
        return stats
