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


class SN10(nn.Module):
    """The simple neural network 10 (SN10) model from `Absolut!`.
    """

    def __init__(self):
        super(SN10, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(11*20, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward_logits(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def forward(self, x, return_logits = False):
        logits = self.forward_logits(x)
        expits = self.sigmoid(logits)
        if return_logits:
            return expits, logits
        else:
            return expits


def train_loop(loader, model, loss_fn, optimizer):
    """Basic training loop for pytorch.

    Args:
        loader (DataLoader)
        model (nn.Model)
        loss_fn (Callable)
        optimizer
    """
    losses = []
    size = len(loader.dataset)
    for batch, (X, y) in enumerate(loader):
        y_pred = model(X)
        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            losses.append(loss)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return losses


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
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )
    loop_metrics = {
        "test_loss": test_loss,
        "accuracy": 100*correct
    }
    return loop_metrics


def openset_loop(open_loader, test_loader, model):
    open_metrics = None
    return open_metrics


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
