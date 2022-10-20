"""
Everything related to machine learning not fitting in a different file.
Includes models, datasets and data loaders.
"""


from argparse import ArgumentError
from pathlib import Path
from tkinter import Y
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

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
        self.linear_1 = nn.Linear(11*20, 10)
        self.activation = nn.ReLU()
        self.linear_2 = nn.Linear(10, 1)
        self.final = nn.Sigmoid()

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.linear_1(x)
        x = self.activation(x)
        logits = self.linear_2(x)
        return logits

    def forward(
        self, 
        x: torch.Tensor, 
        return_logits = False
        ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        logits = self.forward_logits(x)
        expits = self.final(logits)
        if return_logits:
            return expits, logits
        else:
            return expits


class MulticlassSN10(SN10):

    def __init__(self, num_classes: int):
        
        super().__init__()
        
        assert num_classes < 10
        self.linear_2 = nn.Linear(10, num_classes)

        # Cross-entropy loss expects raw, not softmax
        self.final = nn.Identity()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(
        self,
        x: torch.Tensor,
        ) -> torch.Tensor:

        # confusing but these are the logits
        logits = super().forward(x, return_logits = False)
        return logits
    
    def forward_prob(
        self,
        x: torch.Tensor,
        ) -> torch.Tensor:
        logits = self.forward(x)
        return self.softmax(logits)


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
        
        if type(loss_fn) == nn.CrossEntropyLoss:
            loss = loss_fn(y_pred, y.reshape(-1))
        elif type(loss_fn) == nn.BCELoss:
            loss = loss_fn(y_pred, y)
        else:
            raise NotImplementedError(f"{loss_fn=} not implemented.")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            losses.append(loss)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return losses


def test_loop(loader, model, loss_fn) -> dict:
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

    # TODO: Split above and below into 2 funcs.

    x_test, y_test = list(
        DataLoader(loader.dataset, batch_size=len(loader.dataset))
        )[0]
    closed_metrics = compute_metrics_closed_testset(model, x_test, y_test)

    return {
        **loop_metrics,
        **closed_metrics,
    }


def openset_loop(open_loader, test_loader, model):
    x_open, y = list(
        DataLoader(open_loader.dataset, batch_size=len(open_loader.dataset))
        )[0]
    x_test, y_test = list(
        DataLoader(test_loader.dataset, batch_size=len(test_loader.dataset))
        )[0]
    del y
    del y_test
    open_metrics = compute_metrics_open_testset(model, x_open, x_test)
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


def construct_optimizer(
    optimizer_type,
    learning_rate,
    momentum,
    weight_decay,
    model,
    ) -> torch.optim.Optimizer:
    if optimizer_type == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            )
    elif optimizer_type == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate,
            betas=(momentum, 0.999),  # beta1 ~ momentum
            weight_decay=weight_decay,
            )
    else:
        raise ValueError(f"optimizer_type `{optimizer_type}` not recognized.")
    return optimizer


def train_for_ndb1(
    epochs, 
    learning_rate,
    train_loader,
    test_loader,
    open_loader,
    model,
    optimizer_type: str = "SGD",
    momentum: float = 0,
    weight_decay: float = 0,
    ) -> List[dict]:
    """Train model for the NDB1 problem formalization.

    Args:
        epochs (_type_): _description_
        learning_rate (_type_): _description_
        train_loader (_type_): _description_
        test_loader (_type_): _description_
        open_loader (_type_): _description_
        model (_type_): _description_

    Returns:
        List[dict]: metrics per epoch.
    """

    loss_fn = nn.BCELoss()
    optimizer = construct_optimizer(
        optimizer_type, 
        learning_rate, 
        momentum, 
        weight_decay, 
        model
        )

    online_metrics_per_epoch = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        losses = train_loop(train_loader, model, loss_fn, optimizer)
        # 2 lines below replace with evaluate_on_closed_and_open_testsets
        test_metrics = test_loop(test_loader, model, loss_fn)
        open_metrics = openset_loop(open_loader, test_loader, model)
        online_metrics_per_epoch.append({
            "train_losses": losses,
            "test_metrics": test_metrics,
            "open_metrics": open_metrics,
        })
    return online_metrics_per_epoch


def compute_binary_metrics(y_test_pred, y_test_true) -> dict:
    acc_closed = metrics.accuracy_score(y_true=y_test_true, y_pred=y_test_pred)
    recall_closed = metrics.recall_score(y_true=y_test_true, y_pred=y_test_pred)
    precision_closed = metrics.precision_score(y_true=y_test_true, y_pred=y_test_pred)
    f1_closed = metrics.f1_score(y_true=y_test_true, y_pred=y_test_pred)
    return {
        "acc": acc_closed,
        "recall": recall_closed,
        "precision": precision_closed,
        "f1": f1_closed,
    }


def compute_metrics_closed_testset(model, x_test, y_test):
    """Compute metrics for the closed test set.

    Args:
        model (torch.nn): implements forward_logits.
        x_test (torch.tensor): closed set input examples.
        y_test (torch.tensor): closed set output examples.

    Returns:
        dict: recorded relevant metrics.
    """
    assert hasattr(model, "forward_logits")

    y_test_logits = model.forward_logits(x_test).detach().numpy().reshape(-1)
    y_test_pred = model.forward(x_test).detach().numpy().reshape(-1).round()
    y_test_true = y_test.detach().numpy().reshape(-1)
    binary_metrics: dict = compute_binary_metrics(y_test_pred, y_test_true)
    roc_auc_closed = metrics.roc_auc_score(y_true=y_test_true, y_score=y_test_logits)
    avg_precision_closed = metrics.average_precision_score(y_true=y_test_true, y_score=y_test_logits)
    metrics_closed = {
        "y_test_logits": y_test_logits,
        "y_test_pred": y_test_pred,
        "y_test_true": y_test_true,
        "roc_auc_closed": roc_auc_closed,
        "avg_precision_closed": avg_precision_closed,
        **{f"{k}_closed": v for k, v in binary_metrics.items()},
    }
    return metrics_closed


def compute_metrics_open_testset(model, x_open, x_test):
    """Compute metrics for the open test set.

    Args:
        model (torch.nn): implements forward_logits.
        x_open (torch.tensor): open set input examples.
        x_test (torch.tensor): closed set input examples.

    Returns:
        dict: recorded metrics.
    """
    assert hasattr(model, "forward_logits")
    
    l2_norm = lambda arr: np.linalg.norm(arr, ord=2, axis=1)

    ## Norm implementation works in both binary and multiclass case.
    # open_abs_logits = abs(model.forward_logits(x_open)).detach().numpy().reshape(-1)
    # closed_abs_logits = abs(model.forward_logits(x_test).detach().numpy()).reshape(-1)

    open_abs_logits = l2_norm(model.forward_logits(x_open).detach().numpy()).reshape(-1)
    closed_abs_logits = l2_norm(model.forward_logits(x_test).detach().numpy()).reshape(-1)    

    df_tmp = pd.DataFrame(data=open_abs_logits, columns=["logits"])
    df_tmp = pd.concat(
        (
            pd.DataFrame(data={"abs_logits": open_abs_logits, "test_type": "open"}),
            pd.DataFrame(data={"abs_logits": closed_abs_logits, "test_type": "closed"})
        ),
        axis=0
    ).reset_index(drop=True)
    y_open_abs_logits = df_tmp["abs_logits"].values
    df_tmp["y"] = np.where(df_tmp["test_type"] == "open", 0, 1)
    y_open_true = df_tmp["y"].values
    del df_tmp
    roc_auc_open = metrics.roc_auc_score(y_true=y_open_true, y_score=y_open_abs_logits)
    avg_precision_open = metrics.average_precision_score(y_true=y_open_true, y_score=y_open_abs_logits)

    # Find optimal threshold based on PR and compute binary classification metrics
    th_opt = find_optimal_threshold(y_open_true, y_open_abs_logits, method="f1")
    y_open_pred = (y_open_abs_logits > th_opt).astype(np.int16)
    open_binary_metrics: dict = compute_binary_metrics(y_open_pred, y_open_true)
    ## TODO: wrong computation
    # fpr_abs_logit_model = y_open_pred.sum() / y_open_pred.shape[0]

    # TODO: refactor, document
    # TODO: wrong computation for fpr_naive_model
    # naive_closedset_prediction = model.forward(x_open).detach().numpy().reshape(-1).round()
    # fpr_naive_model = naive_closedset_prediction.sum() / naive_closedset_prediction.shape[0]  # ideally everything is zero here

    metrics_open = {
        "y_open_abs_logits": y_open_abs_logits,
        "y_open_true": y_open_true,
        "roc_auc_open": roc_auc_open,
        "avg_precision_open": avg_precision_open,
        **{f"{k}_open": v for k, v in open_binary_metrics.items()},
        # "fpr_abs_logit_model": fpr_abs_logit_model,
        # "fpr_naive_model": fpr_naive_model,
    }
    return metrics_open


def evaluate_on_closed_and_open_testsets(open_loader, test_loader, model):
    """Compute evaluation metrics for the closed set and open set
    scenarios.

    Args:
        open_loader (DataLoader): loads the open set testing dataset 
        which includes closed set and open set examples. Evaluated as
        a binary classification problem.

        test_loader (DataLoader): loads the typical closed set test data.
        
        model (torch.nn): the model to be evaluated. Must implement 
        forward_logits(x).
    """
    assert hasattr(model, "forward_logits")

    x, y = list(
        DataLoader(open_loader.dataset, batch_size=len(open_loader.dataset))
        )[0]
    x_test, y_test = list(
        DataLoader(test_loader.dataset, batch_size=len(test_loader.dataset))
        )[0]

    metrics_open = compute_metrics_open_testset(model, x, x_test)
    metrics_closed = compute_metrics_closed_testset(model, x_test, y_test)

    eval_metrics = {
        "open": metrics_open,
        "closed": metrics_closed,
    }
    return eval_metrics


def find_optimal_threshold(
    y_true,
    y_score,
    method: str = "roc"
    ) -> float:
    """Finds optimal thresholds for binary classification.

    https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/
    https://www.yourdatateacher.com/2021/06/14/are-you-still-using-0-5-as-a-threshold/
    1. `roc` - [default] based on ROC curve (top-left corner)
    2. `acc` - [not implemented!] maximize accuracy
    3. `f1`  - maximize F1 score

    Returns:
        float
    """

    if method == "roc":
        fpr, tpr, thresholds = metrics.roc_curve(
            y_true=y_true,
            y_score=y_score,
        )
        th_opt = thresholds[
            np.argmin(np.abs(fpr + tpr - 1))
        ]
    elif method == "f1":
        precision, recall, thresholds = metrics.precision_recall_curve(
            y_true=y_true, 
            probas_pred=y_score,
        )
        fscore = (2 * precision * recall) / (precision + recall + 1e-8)
        th_opt = thresholds[np.argmax(fscore)]
    elif method == "acc":
        raise NotImplementedError("Not of interest currently.")
    else:
        raise ValueError(f"Method {method} not recognized/available.")
    return th_opt


def compute_roc_curve(y_true, y_score) -> tuple:
    """Compute fpr, tpr and optimal thr for ROC curve.

    Args:
        y_true (_type_): _description_
        y_score (_type_): _description_

    Returns:
        tuple: fpr, tpr, thresholds, optimal_thr
    """    
    fpr, tpr, thresholds = metrics.roc_curve(
        y_true=y_true,
        y_score=y_score,
    )
    optimal_thr = find_optimal_threshold(y_true, y_score, method="roc")
    return fpr, tpr, thresholds, optimal_thr


def compute_pr_curve(y_true, y_score) -> tuple:
    """Compute precision, recall and f1-score-optimal thr for ROC curve.

    Args:
        y_true (_type_): _description_
        y_score (_type_): _description_

    Returns:
        tuple: precision, recall, thresholds, optimal_thr
    """  
    precision, recall, thresholds = metrics.precision_recall_curve(
        y_true=y_true, 
        probas_pred=y_score,
    )
    optimal_thr = find_optimal_threshold(y_true, y_score, method="f1")
    return precision, recall, thresholds, optimal_thr
