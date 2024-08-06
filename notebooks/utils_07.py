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

from NegativeClassOptimization import (config, datasets, ml, preprocessing,
                                       utils, visualisations)

task_types = (
    datasets.ClassificationTaskType.ONE_VS_ONE,
    datasets.ClassificationTaskType.ONE_VS_NINE,
    datasets.ClassificationTaskType.HIGH_VS_95LOW,
    datasets.ClassificationTaskType.HIGH_VS_LOOSER,
)

loader = datasets.FrozenMiniAbsolutMLLoader(
    data_dir=Path("../data/Frozen_MiniAbsolut_ML/")
)

loader_linear = datasets.FrozenMiniAbsolutMLLoader(
    data_dir=Path("../data/Frozen_MiniAbsolut_Linear_ML/")
)

palette = {
    "1FBI": "#008080",
    "3VRL": "#FFA07A",
    "2YPV": "#000080",
    "5E94": "#FFD700",
    "1WEJ": "#228B22",
    "1OB1": "#FF69B4",
    "1NSN": "#800080",
    "1H0D": "#FF6347",
    "3RAJ": "#00FF00",
    "1ADQ": "#FF1493",
}
task_order = ["high_vs_95low", "1v1", "1v9", "high_vs_looser"]


def task_generator(task_types=task_types, loader=loader, without_1v1=False):
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

    if datasets.ClassificationTaskType.ONE_VS_ONE in task_types and not without_1v1:
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

    fn = list(ag_pos_dir.glob(f"high_test_[0-9]*{suffix}.tsv"))[0]
    df_pos = pd.read_csv(fn, sep="\t", header=header)
    df_pos["class"] = "positive"

    if task.task_type == datasets.ClassificationTaskType.HIGH_VS_95LOW:
        fn = list(ag_pos_dir.glob(f"95low_test_[0-9]*{suffix}.tsv"))[0]
        df_neg = pd.read_csv(fn, sep='\t', header=header)
        df_neg["class"] = "negative"
    elif task.task_type == datasets.ClassificationTaskType.HIGH_VS_LOOSER:
        fn = list(ag_pos_dir.glob(f"looserX_test_[0-9]*{suffix}.tsv"))[0]
        df_neg = pd.read_csv(fn, sep='\t', header=header)
        df_neg["class"] = "negative"
    elif task.task_type in (datasets.ClassificationTaskType.ONE_VS_ONE, datasets.ClassificationTaskType.ONE_VS_NINE):
        # raise ValueError("Not implemented.")
        df_neg = pd.DataFrame()

    df = pd.concat([df_pos, df_neg], axis=0)  # type:ignore

    return df


def get_miniabsolut_dataframes_for_shuffled(task, load_energy_contributions=False):
    """
    Similar to get_miniabsolut_dataframes, but for shuffled data. It has to 
    account for the fact that when we shuffle pos and neg antigens, we preserve
    the Energy data just for the positive dataset that remains positive after
    shuffling.
    """
    df = get_miniabsolut_dataframes(task, load_energy_contributions=load_energy_contributions)
    
    assert task.test_dataset is not None, "Task has no test dataset."
    df_test = task.test_dataset.copy()
    df = pd.merge(df, df_test[["Slide", "y"]], on="Slide", how="inner")    

    # Redefine positive and negative classes based on y
    df.loc[df["y"] == 1, "class"] = "positive"
    df.loc[df["y"] == 0, "class"] = "negative"

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


def get_miniabsolut_for_epitopes(task):
    df = pd.DataFrame.from_records(task.attributions)
        
        # Get MiniAbsolut
    df_miniabs_pos = pd.read_csv(f"../data/MiniAbsolut/{task.ag_pos}/highepi_test_3000.tsv", sep='\t')
    assert all(df.query("y_true == 1").slide.isin(df_miniabs_pos["Slide"])), "Not all positive slides in MiniAbsolut."
    del df_miniabs_pos

        # Get Energies
    df_energies = load_testrest_from_miniabsolut(task.ag_pos.split("E1")[0])
    assert all(df.query("y_true == 1").slide.isin(df_energies["Slide"])), "Not all positive slides in Energies."

    df = df.merge(df_energies, left_on="slide", right_on="Slide", how="left")

    # Drop duplicated slides
    df = df.sort_values("Energy", ascending=True)
    df = df.drop_duplicates(subset="slide")
    assert df.query("y_true == 1").shape[0] in [3000, 5000]
    df["class"] = np.where(df["y_true"] == 1, "positive", "negative")

    return df


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
    if attributor_name.split("_")[0] == "DeepLIFT":  # For DeepLIFT we need to subset and reshape
        attr = np.stack(
            list(
                map(
                    # lambda x: np.array(x[attributor_name]).reshape((11, 20)),
                    lambda x: np.array(x[attributor_name]).reshape((-1, 20)),
                    filter(lambda x: x["y_true"] in y_true, records),
                )
            )
        )
    else:
        attr = np.stack(
            list(
                map(
                    lambda x: np.array(x[attributor_name]),
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
    task, loader=loader_linear, load_slide_df=False, attributions_toload="v0.1.2-3", load_everything=False
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
    task = loader.load(task, attributions_toload=attributions_toload)
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
    if load_everything:
        return stats, slide_df, energy_dict, attr_dict
    elif load_slide_df:
        return stats, slide_df
    else:
        return stats


def load_energy_contributions_from_task_nonlinear_version(
        task, 
        load_slide_df=False, 
        load_everything=False,
        attributor_name="DeepLIFT_LOCAL_v2.0-2",
        attribution_records_toload="attribution_records.json",
        task_is_loaded=False,
        load_miniabsolut_for_shuffled=False,
        attr_analysis_name="v2.0-2",
        load_miniabsolut_type="standard",
        ):
    # Get energy contributions and attributions

    if not task_is_loaded:
        task = loader.load(task, attributions_toload=attr_analysis_name, attribution_records_toload=attribution_records_toload)
    
    if load_miniabsolut_for_shuffled:
        df = get_miniabsolut_dataframes_for_shuffled(task, load_energy_contributions=True)
    else:
        if load_miniabsolut_type == "standard":
            df = get_miniabsolut_dataframes(task, load_energy_contributions=True)
        elif load_miniabsolut_type == "epitope_analysis":
            df = get_miniabsolut_for_epitopes(task)
            # Dropna from class == negative
            df = df.loc[~((df["Energy"].isna()) & (df["class"] == "negative"))].copy()
        else:
            raise ValueError(f"load_miniabsolut_type {load_miniabsolut_type} not recognized.")
    
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

    if attributor_name.split("_")[0] == "DeepLIFT":  # For DeepLIFT we need to subset and reshape
        attr_stack = get_attr_from_records(
            task.attributions, attributor_name, (0, 1)  # type: ignore
        )  # Nx11x20 # type: ignore
        onehot_stack = get_onehotstack_from_records(
            task.attributions, (0, 1)  # type: ignore
        )  # Nx220 # type: ignore
        attr_aa = attr_stack[onehot_stack.reshape((-1, 11, 20)) == 1].reshape(
            -1, 11
        )  # Nx11
    else:
        attr_aa = get_attr_from_records(
            task.attributions, attributor_name, (0, 1)  # type: ignore
        )
    attr_dict = {
        record["slide"]: {**record, **{"attribution_existingaa": attr_aa[i, :]}}
        for i, record in enumerate(task.attributions)  # type: ignore
    }  # type: ignore

    # Combine dictionaries
    slide_records = []
    for slide in energy_dict.keys():
        
        try:
            
            dataset_class = energy_dict[slide]["class"]
            energies = energy_dict[slide]["energies"]
            energies_fold = energy_dict[slide]["energies_fold"]
            energies_total = energy_dict[slide]["energies_total"]
            attrs = attr_dict[slide]["attribution_existingaa"]

            # Correlation between energy and attribution with scipy
            r, p = pearsonr(energies, attrs)
            r_fold, p_fold = pearsonr(energies_fold, attrs)
            r_total, p_total = pearsonr(energies_total, attrs)

        except:
            print(f"Error in slide {slide}")
            r, p, r_fold, p_fold, r_total, p_total = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            dataset_class = np.nan
            energies = np.nan
            energies_fold = np.nan
            energies_total = np.nan
            attrs = np.nan

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

    if load_everything:
        return stats, slide_df, energy_dict, attr_dict
    elif load_slide_df:
        return stats, slide_df
    else:
        return stats


def plot_1v1_logits_energy(ag_pos, ag_neg, model="linear"):
    if model == "linear":
        fp = f"../data/Frozen_MiniAbsolut_Linear_ML/1v1/seed_0/split_42/{ag_pos}__vs__{ag_neg}/attributions/v0.1.2-3/attribution_records.json"

    elif model == "SN10":
        fp = f"../data/Frozen_MiniAbsolut_ML/1v1/seed_0/split_42/{ag_pos}__vs__{ag_neg}/attributions/v2.0-2/attribution_records.json"

    fp_test_1 = f"../data/MiniAbsolut/{ag_pos}/high_test_5000.tsv"
    fp_test_2 = f"../data/MiniAbsolut/{ag_neg}/high_test_5000.tsv"

    df_test_1 = pd.read_csv(fp_test_1, sep="\t")
    df_test_1["sample_class"] = "positive"
    df_test_2 = pd.read_csv(fp_test_2, sep="\t")
    df_test_2["sample_class"] = "negative"
    df_test = pd.concat([df_test_1, df_test_2], axis=0)

    # Open json from fp path into data variable
    with open(fp, "r") as f:  # type:ignore
        data = json.load(f)

    df = pd.DataFrame.from_records(data)
    df = pd.merge(df, df_test, left_on="slide", right_on="Slide")

    fig, ax = plt.subplots(figsize=(3.14, 3.14), dpi=600)
    sns.set_theme(context="paper")
    sns.set_style("white")

    cmap = [
        "#f1593a",  # red
        "#ffc40d",  # yellow
        "#28a3dc",  # blue
    ]

    sns.scatterplot(
        data=df,
        x="Energy",
        y="logits",
        hue="sample_class",
        ax=ax,
        palette=cmap,
    )

    ax.set_xlabel("Energy, kcal/mol")
    ax.set_ylabel("Logits")
    ax.set_title(f"{ag_pos} vs {ag_neg}")
    ax.set_xticks(np.arange(-110, -89, 5))

    # Add a coefficient of correlation and coeficient of determination
    r, p = pearsonr(
        df.query("sample_class == 'positive'")["Energy"],
        df.query("sample_class == 'positive'")["logits"],
    )
    r2 = r**2
    ax.text(
        -109,
        -10,
        f"r = {r:.2f}\n$r^2$ = {r2:.2f}",
        bbox=dict(facecolor="white", alpha=0.5),
    )


def get_energy_contributions_foldx():
    """
    Return a dataframe with the energy contributions
    from FoldX, computed by Puneet for the Brij dataset.
    """
    path = Path("../data/Experimental_Datasets/energy_contribution_8anstrom/energy_contacts_Tz_hb_CDRH3.csv")
    df_e = pd.read_csv(path)
    df_e["Slide"] = df_e["pdb"].str.split(".").str[0]
    df_e["res_pos"] = df_e["resi_name"].str.split(":").str[0].astype(int)
    df_e["slide_idx"] = df_e["res_pos"] - 99
    df_e = df_e[["Slide", "slide_idx", "res_pos", "total_e"]].copy()
    df_e = df_e.sort_values(["Slide", "slide_idx"])
    df_e = df_e.pivot(index="Slide", columns="slide_idx", values="total_e")
    return df_e


def load_trainrest_from_miniabsolut(ag, base_path = None):

    if base_path is None:
        base_path = config.DATA_MINIABSOLUT / f"{ag}/energy_contributions"

    df_high_train = pd.read_csv(base_path / "high_train_15000_absolut_energy_contributions.tsv", sep='\t', header=1)
    df_high_rest = pd.read_csv(base_path / "high_rest_absolut_energy_contributions.tsv", sep='\t', header=1)

    df_weak_train = pd.read_csv(base_path / "looserX_train_15000_absolut_energy_contributions.tsv", sep='\t', header=1)
    df_weak_rest = pd.read_csv(base_path / "looserX_rest_absolut_energy_contributions.tsv", sep='\t', header=1)

    df_nonb_train = pd.read_csv(base_path / "95low_train_15000_absolut_energy_contributions.tsv", sep='\t', header=1)
    df_nonb_rest = pd.read_csv(base_path / "95low_rest_absolut_energy_contributions.tsv", sep='\t', header=1)

    df_high_train["binder_type"] = f"{ag}_high"
    df_high_rest["binder_type"] = f"{ag}_high"

    df_weak_train["binder_type"] = f"{ag}_looserX"
    df_weak_rest["binder_type"] = f"{ag}_looserX"

    df_nonb_train["binder_type"] = f"{ag}_95low"
    df_nonb_rest["binder_type"] = f"{ag}_95low"

    # Concatenate all
    df = pd.concat(
        [
            df_high_train,
            df_high_rest,
            df_weak_train,
            df_weak_rest,
            df_nonb_train,
            df_nonb_rest,
        ]
    )
    
    return df


def load_testrest_from_miniabsolut(ag, base_path = None):

    if base_path is None:
        base_path = config.DATA_MINIABSOLUT / f"{ag}/energy_contributions"

    df_high_test = pd.read_csv(base_path / "high_test_5000_absolut_energy_contributions.tsv", sep='\t', header=1)
    df_high_rest = pd.read_csv(base_path / "high_rest_absolut_energy_contributions.tsv", sep='\t', header=1)

    df_weak_test = pd.read_csv(base_path / "looserX_test_5000_absolut_energy_contributions.tsv", sep='\t', header=1)
    df_weak_rest = pd.read_csv(base_path / "looserX_rest_absolut_energy_contributions.tsv", sep='\t', header=1)

    df_nonb_test = pd.read_csv(base_path / "95low_test_5000_absolut_energy_contributions.tsv", sep='\t', header=1)
    df_nonb_rest = pd.read_csv(base_path / "95low_rest_absolut_energy_contributions.tsv", sep='\t', header=1)

    df_high_test["binder_type"] = f"{ag}_high"
    df_high_test["origin"] = "test"
    df_high_rest["binder_type"] = f"{ag}_high"
    df_high_rest["origin"] = "rest"

    df_weak_test["binder_type"] = f"{ag}_looserX"
    df_weak_test["origin"] = "test"
    df_weak_rest["binder_type"] = f"{ag}_looserX"
    df_weak_rest["origin"] = "rest"

    df_nonb_test["binder_type"] = f"{ag}_95low"
    df_nonb_test["origin"] = "test"
    df_nonb_rest["binder_type"] = f"{ag}_95low"
    df_nonb_rest["origin"] = "rest"

    # Concatenate all
    df = pd.concat(
        [
            df_high_test,
            df_high_rest,
            df_weak_test,
            df_weak_rest,
            df_nonb_test,
            df_nonb_rest,
        ]
    )
    
    return df
