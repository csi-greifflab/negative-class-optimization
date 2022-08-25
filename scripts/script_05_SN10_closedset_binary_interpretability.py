"""
"""

import os
from pathlib import Path
from typing import List
import yaml

import pandas as pd
import numpy as np
import seaborn as sns
import joypy

import torch
from torch import nn
from torch.utils.data import DataLoader
from captum.attr import IntegratedGradients

import NegativeClassOptimization.config as config
import NegativeClassOptimization.preprocessing as preprocessing
from NegativeClassOptimization.ml import (
    BinaryDataset, preprocess_data_for_pytorch_binary,
    SN10,
    train_loop, test_loop,
    compute_integratedgradients_attribution
)


# Get parameters
with open(config.PARAMS_PATH, "r") as fh:
    params = yaml.safe_load(fh)

ag_pos = params["05_SN10_closedset_binary_interpretability"]["ag_pos"]
ag_neg = params["05_SN10_closedset_binary_interpretability"]["ag_neg"]
learning_rate = params["05_SN10_closedset_binary_interpretability"]["learning_rate"]
epochs = params["05_SN10_closedset_binary_interpretability"]["epochs"]

out_dir = Path("data/SN10_closedset_binary_interpretability")
data_path = config.DATA_SLACK_1_GLOBAL

attributions_savepath = out_dir / f"attributions_{ag_pos}_vs_{ag_neg}.tsv"
agg_attributions_savepath = out_dir / \
    f"aggregated_attributions_{ag_pos}_vs_{ag_neg}.tsv"
attributions_ag_pos_fig_path = out_dir / \
    f"aggregated_attributions_{ag_pos}.png"
attributions_ag_neg_fig_path = out_dir / \
    f"aggregated_attributions_{ag_neg}.png"


def process_data_and_train_model(ag_pos, ag_neg, learning_rate, epochs, data_path):
    """Workflow for processing data and training SN10.

    Args:
        ag_pos (_type_)
        ag_neg (_type_)
        learning_rate (_type_)
        epochs (_type_)
        data_path (_type_)

    Returns:
        _type_
    """    
    df = pd.read_csv(data_path, sep='\t')
    df = df.loc[df["Antigen"].isin([ag_pos, ag_neg])].copy()

    (
        train_data,
        test_data,
        train_loader,
        test_loader) = preprocess_data_for_pytorch_binary(
            df,
            [ag_pos],
            scale_onehot=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    model = SN10().to(device)
    print(model)

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    online_metrics = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        losses = train_loop(train_loader, model, loss_fn, optimizer)
        test_metrics = test_loop(test_loader, model, loss_fn)
        online_metrics.append({
            "train_losses": losses,
            "test_metrics": test_metrics,
        })

    return train_data, test_data, model, online_metrics


def run_attribution_workflow(
        ag_pos,
        ag_neg,
        attributions_savepath,
        agg_attributions_savepath,
        attributions_ag_pos_fig_path,
        attributions_ag_neg_fig_path,
        test_data, model
    ):
    """Workflow for SN10 interpretability.

    Args:
        ag_pos (_type_)
        ag_neg (_type_)
        attributions_savepath (_type_)
        agg_attributions_savepath (_type_)
        attributions_ag_neg_fig_path (_type_)
        test_data (_type_)
        model (_type_)

    Raises:
        RuntimeError
    """    
    records = compute_integratedgradients_attribution(test_data, model)

    attributions = list(map(lambda tupl: tupl[0].numpy(), records))

    abs_sums = []
    for attr in attributions:
        abs_sum = np.abs(attr.reshape((11, 20))).sum(axis=1)
        abs_sums.append(abs_sum)

    slide_pos_cols = (f"Slide position {i}" for i in range(1, 12))
    df_attr = pd.DataFrame(data=np.stack(abs_sums), columns=slide_pos_cols)

    df_ext = pd.DataFrame(
        data=np.stack(map(lambda a: a.reshape(-1), attributions)),
        columns=(
            f"{position} {aa}"
            for position in range(1, 12)
            for aa in config.AMINOACID_ALPHABET
        )
    )
    df_ext.to_csv(attributions_savepath, sep='\t')

    test_antigens = []
    for i in DataLoader(test_data, batch_size=1):
        y_i = i[1].reshape(-1)
        if y_i == 0:
            test_antigens.append(ag_neg)
        elif y_i == 1:
            test_antigens.append(ag_pos)
        else:
            raise RuntimeError()
    df_attr["Antigen"] = test_antigens
    df_attr.to_csv(agg_attributions_savepath, sep='\t')

    fig1, axs1 = joypy.joyplot(
        df_attr.loc[df_attr["Antigen"] == ag_pos].iloc[:, :-1],
        figsize=(7, 7),
        title=(
            f"[{ag_pos}, + test cases] Integrated gradients-based attribution of prediction"
            " (x-axis)\nfor Slide amino-acid position (y-axis)\n"
            f"based on SN10 network for ({ag_pos} vs {ag_neg}) classification"
        )
    )
    fig1.savefig(attributions_ag_pos_fig_path)

    fig2, axs2 = joypy.joyplot(
        df_attr.loc[df_attr["Antigen"] == ag_neg].iloc[:, :-1],
        figsize=(7, 7),
        title=(
            f"[{ag_neg}, - test cases] Integrated gradients-based attribution of prediction"
            " (x-axis)\nfor Slide amino-acid position (y-axis)\n"
            f"based on SN10 network for ({ag_pos} vs {ag_neg}) classification"
        )
    )
    fig2.savefig(attributions_ag_neg_fig_path)


if __name__ == "__main__":

    out_dir.mkdir(exist_ok=True)

    train_data, test_data, model, online_metrics = process_data_and_train_model(
        ag_pos,
        ag_neg,
        learning_rate,
        epochs,
        data_path
    )

    run_attribution_workflow(
        ag_pos,
        ag_neg,
        attributions_savepath,
        agg_attributions_savepath,
        attributions_ag_pos_fig_path,
        attributions_ag_neg_fig_path,
        test_data,
        model
    )
