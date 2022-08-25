from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import NegativeClassOptimization.ml as ml
import NegativeClassOptimization.config as config


N = 1000
epochs = 2
learning_rate = 0.01
ag_pos = "1FBI"
ag_neg = "1NSN"


def process_data(N, ag_pos, ag_neg, data_path):
    df = pd.read_csv(data_path, sep='\t')
    df_closed = df.loc[df["Antigen"].isin([ag_pos, ag_neg])].copy().sample(N)
    df_open = df.loc[df["Antigen"].isin(config.ANTIGENS_OPENSET)].copy()
    df_open = df_open.loc[~df_open["Slide"].isin(df_closed["Slide"])].sample(N)
    df_open = df_open.reset_index(drop=True)

    (   
        train_data,
        test_data,
        open_data,
        train_loader,
        test_loader,
        open_loader,
    ) = ml.preprocess_data_for_pytorch_binary(
            df_closed,
            [ag_pos],
            scale_onehot=True,
            df_openset=df_open,
    )
    
    return (   
        train_data,
        test_data,
        open_data,
        train_loader,
        test_loader,
        open_loader,
    )

if __name__ == "__main__":

    data_path = Path(config.DATA_SLACK_1_GLOBAL)

    (   
        train_data,
        test_data,
        open_data,
        train_loader,
        test_loader,
        open_loader,
    ) = process_data(N, ag_pos, ag_neg, data_path)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    model = ml.SN10().to(device)
    print(model)

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    online_metrics = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        losses = ml.train_loop(train_loader, model, loss_fn, optimizer)
        test_metrics = ml.test_loop(test_loader, model, loss_fn)
        openset_metrics = ml.openset_loop(open_loader, model)
        online_metrics.append({
            "train_losses": losses,
            "test_metrics": test_metrics,
            "openset_metrics": openset_metrics,
        })
    
    # online_metrics, model
    