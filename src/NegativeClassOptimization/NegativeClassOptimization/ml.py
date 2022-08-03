"""
Everything related to machine learning not fitting in a different file.
Includes models, datasets and data loaders.
"""


from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from captum.attr import IntegratedGradients

import NegativeClassOptimization.config as config
import NegativeClassOptimization.preprocessing as preprocessing


class PairwiseDataset(Dataset):
    """Pytorch dataset for modelling 2 antigen binder binary classifiers.
    """    
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        return (
            torch.tensor(self.df.loc[idx, "X"]).reshape((1, -1)).type(torch.float),
            torch.tensor(self.df.loc[idx, "y"]).reshape((1)).type(torch.float),
        )


def remove_duplicates_for_pairwise(df: pd.DataFrame, ag_pos: str) -> pd.DataFrame:
    """An important step in preparing data for SN10 training and evaluation. 
    Most importantly - appropriately removes duplicates.

    Args:
        df (pd.DataFrame): typical dataframe used in the project
        pos_ag (str): the antigen assuming the positive dataset role

    Returns:
        pd.DataFrame: df with new columns suitable for modelling.
    """
    
    def infer_antigen_from_duplicate_list(antigens: List[str], pos_antigen: str):
        assert len(antigens) <= 2, ">2 antigens not supported yet."
        if len(antigens) == 1:
            return antigens[0]
        else:
            if pos_antigen in antigens:
                return pos_antigen
            else:
                return list(set(antigens) - set([pos_antigen]))[0]

    df = df.groupby("Slide").apply(
        lambda df_: infer_antigen_from_duplicate_list(df_["Antigen"].unique().tolist(), pos_antigen=ag_pos)
    )
    df = pd.DataFrame(data=df, columns=["Antigen"])
    df = df.reset_index()
    return df


def preprocess_data_for_pytorch_pairwise(
    df,
    ag_pos,
    batch_size = 64,
    train_frac = 0.8
):

    df = remove_duplicates_for_pairwise(df, ag_pos)
    preprocessing.onehot_encode_df(df)

    df["X"] = df["Slide_onehot"]
    df["y"] = np.where(df["Antigen"] == ag_pos, 1, 0)

    df = df.sample(frac=1).reset_index(drop=True)  # shuffle

    split_idx = int(df.shape[0] * train_frac)
    df_train = df.loc[:split_idx].copy().reset_index(drop=True)
    df_test = df.loc[split_idx:].copy().reset_index(drop=True)

    train_data = PairwiseDataset(df_train)
    test_data = PairwiseDataset(df_test)

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
            correct += (torch.round(y_pred) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


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