"""
Everything related to machine learning not fitting in a different file.
Includes models, datasets and data loaders.
"""


from argparse import ArgumentError
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from captum.attr import IntegratedGradients

import NegativeClassOptimization.config as config
import NegativeClassOptimization.preprocessing as preprocessing


class BinaryDataset(Dataset):
    """Pytorch dataset for modelling antigen binding binary classifiers.
    """

    def __init__(self, df):
        self.df = df

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        return (
            torch.tensor(self.df.loc[idx, "X"]).reshape(
                (1, -1)).type(torch.float),
            torch.tensor(self.df.loc[idx, "y"]).reshape((1)).type(torch.float),
        )


class MulticlassDataset(Dataset):
    """Pytorch dataset for modelling antigen binding multiclass classifiers.
    """

    def __init__(self, df):
        self.df = df

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        return (
            torch.tensor(self.df.loc[idx, "X"]).reshape(
                (1, -1)).type(torch.float),
            torch.tensor(self.df.loc[idx, "y"]).reshape((1)).type(torch.uint8),
        )    


class OpenDataset(BinaryDataset):
    """Class for keeping open datasets.
    """

    def __init__(self, df):
        num_binders = sum(df["y"] == 1)
        if num_binders > 0:
            raise ArgumentError(
                "Dataframe doesn't represent an open set"
                f" - there are {num_binders} binders."
            )
        self.df = df


def preprocess_data_for_pytorch_binary(
    df,
    ag_pos: List[str],
    batch_size = 64,
    train_frac = 0.8,
    scale_onehot = True,
):
    """Get train and test pytorch Datasets and DataLoaders.add()

    Args:
        df (pd.DataFrame): dataframe in typical global format.add()
        ag_pos (List[str]): list of antigens labeled as positive.add()
        batch_size (int, optional): Defaults to 64.
        train_frac (float, optional): Defaults to 0.8.

    Returns:
        tuple: (train_data, test_data, train_loader, test_loader).
    """

    df = preprocessing.remove_duplicates_for_binary(df, ag_pos)
    df = preprocessing.onehot_encode_df(df)

    df = df.sample(frac=1).reset_index(drop=True)  # shuffle

    split_idx = int(df.shape[0] * train_frac)
    df_train = df.loc[:split_idx].copy().reset_index(drop=True)
    df_test = df.loc[split_idx:].copy().reset_index(drop=True)

    if scale_onehot:
        train_onehot_stack = np.stack(df_train["Slide_onehot"], axis=0)
        test_onehot_stack = np.stack(df_test["Slide_onehot"], axis=0)
        scaler = StandardScaler()
        scaler.fit(train_onehot_stack)
        df_train["Slide_onehot"] = scaler.transform(train_onehot_stack).tolist()
        df_test["Slide_onehot"] = scaler.transform(test_onehot_stack).tolist()

    df_train["X"] = df_train["Slide_onehot"]
    df_train["y"] = df_train["binds_a_pos_ag"]
    df_test["X"] = df_test["Slide_onehot"]
    df_test["y"] = df_test["binds_a_pos_ag"]

    train_data = BinaryDataset(df_train)
    test_data = BinaryDataset(df_test)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return (train_data, test_data, train_loader, test_loader)


def preprocess_data_for_pytorch_multiclass(
    df,
    batch_size = 64,
    train_frac = 0.8,
    scale_onehot = True,
):
    """Get train and test pytorch Datasets and DataLoaders.

    Args:
        df (pd.DataFrame): dataframe in typical global format.add()
        ag_pos (List[str]): list of antigens labeled as positive.add()
        batch_size (int, optional): Defaults to 64.
        train_frac (float, optional): Defaults to 0.8.

    Returns:
        tuple: (train_data, test_data, train_loader, test_loader).
    """

    df = preprocessing.remove_duplicates_for_multiclass(df)
    df = preprocessing.onehot_encode_df(df)

    df = df.sample(frac=1).reset_index(drop=True)  # shuffle

    split_idx = int(df.shape[0] * train_frac)
    df_train = df.loc[:split_idx].copy().reset_index(drop=True)
    df_test = df.loc[split_idx:].copy().reset_index(drop=True)

    if scale_onehot:
        train_onehot_stack = np.stack(df_train["Slide_onehot"], axis=0)
        test_onehot_stack = np.stack(df_test["Slide_onehot"], axis=0)
        scaler = StandardScaler()
        scaler.fit(train_onehot_stack)
        df_train["Slide_onehot"] = scaler.transform(train_onehot_stack).tolist()
        df_test["Slide_onehot"] = scaler.transform(test_onehot_stack).tolist()

    label_encoder = preprocessing.get_antigen_label_encoder()
    df_train["X"] = df_train["Slide_onehot"]
    df_train["y"] = label_encoder.transform(df_train["Antigen"])
    df_test["X"] = df_test["Slide_onehot"]
    df_test["y"] = label_encoder.transform(df_test["Antigen"])

    train_data = MulticlassDataset(df_train)
    test_data = MulticlassDataset(df_test)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return (train_data, test_data, train_loader, test_loader)


class SN10(nn.Module):
    """The simple neural network 10 (SN10) model from `Absolut!`.
    """

    def __init__(self):
        super(SN10, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(11*20, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train_loop(loader, model, loss_fn, optimizer):
    """Basic training loop for pytorch.

    Args:
        loader (DataLoader)
        model (nn.Model)
        loss_fn (Callable)
        optimizer
    """
    size = len(loader.dataset)
    for batch, (X, y) in enumerate(loader):
        y_pred = model(X)
        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(loader, model, loss_fn):
    """Basic test loop for pytorch.

    Args:
        loader (DataLoader)
        model (nn.Model)
        loss_fn (Callable)
    """
    size = len(loader.dataset)
    num_batches = len(loader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in loader:
            y_pred = model(X)
            test_loss += loss_fn(y_pred, y).item()
            correct += (torch.round(y_pred) ==
                        y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def compute_integratedgradients_attribution(data: Dataset, model: nn.Module) -> List[Tuple[np.array, float]]:
    """Compute Integrated Gradients attribution for a model on a dataset.

    Args:
        data (Dataset)
        model (nn.Module)

    Returns: list of tuples containing attributions and approximation errors (for integration).
    """
    ig = IntegratedGradients(model)

    inputs = tuple(map(
        lambda pair: pair[0].reshape((-1, 11*20)),
        DataLoader(data, batch_size=1)
    ))

    records = []
    for input in inputs:
        attributions, approximation_error = ig.attribute(
            inputs=input,
            baselines=0,
            n_steps=100,
            method="gausslegendre",
            return_convergence_delta=True,
        )
        records.append((attributions, approximation_error))
    return records
