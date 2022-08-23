"""
"""

import os
from pathlib import Path
from typing import List

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


## Parameters
ag_pos = "3VRL"
ag_neg = "1ADQ"
learning_rate = 0.01
epochs = 5


df = pd.read_csv(config.DATA_SLACK_1_GLOBAL, sep='\t')
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

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_loader, model, loss_fn, optimizer)
    test_loop(test_loader, model, loss_fn)


# Interpretability
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
        for position in range (1, 12) 
        for aa in config.AMINOACID_ALPHABET
    )
)


test_antigens = []
for i in DataLoader(test_data, batch_size=1):
    y_i = i[1].reshape(-1)
    if y_i == 0:
        test_antigens.append("1ADQ")
    elif y_i == 1:
        test_antigens.append("3VRL")
    else:
        raise RuntimeError()
df_attr["Antigen"] = test_antigens


g = joypy.joyplot(
    df_attr.loc[df_attr["Antigen"] == "3VRL"].iloc[:, :-1],
    figsize=(7, 7),
    title=(
        "[3VRL, + test cases] Integrated gradients-based attribution of prediction"
        " (x-axis)\nfor Slide amino-acid position (y-axis)\n"
        "based on SN10 network for (3VRL vs 1ADQ) classification"
    )
)

g = joypy.joyplot(
    df_attr.loc[df_attr["Antigen"] == "1ADQ"].iloc[:, :-1],
    figsize=(7, 7),
    title=(
        "[1ADQ, - test cases] Integrated gradients-based attribution of prediction"
        " (x-axis)\nfor Slide amino-acid position (y-axis)\n"
        "based on SN10 network for (3VRL vs 1ADQ) classification"
    )
)