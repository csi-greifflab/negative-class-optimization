"""
Everything related to machine learning not fitting in a different file.
Includes models, datasets and data loaders.
"""


from argparse import ArgumentError
import math
from pathlib import Path
from tkinter import Y
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from captum.attr import IntegratedGradients, DeepLift

import mlflow

import NegativeClassOptimization.config as config
import NegativeClassOptimization.utils as utils
import NegativeClassOptimization.datasets as datasets
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
    
    def predict(
        self,
        x: torch.Tensor,
        ) -> torch.Tensor:
        return self.forward(x, return_logits=False).round()

    def compute_metrics_closed_testset(self, x_test, y_test):
        return SN10.compute_metrics_closed_testset_static(self, x_test, y_test)

    @staticmethod
    def compute_metrics_closed_testset_static(model, x_test, y_test):
        y_test_logits = model.forward_logits(x_test).detach().numpy().reshape(-1)
        y_test_pred = model.forward(x_test).detach().numpy().reshape(-1).round()
        y_test_true = y_test.detach().numpy().reshape(-1)
        binary_metrics: dict = compute_binary_metrics(y_test_pred, y_test_true)
        
        try:
            roc_auc_closed = metrics.roc_auc_score(y_true=y_test_true, y_score=y_test_logits)    
            avg_precision_closed = metrics.average_precision_score(y_true=y_test_true, y_score=y_test_logits)
        except ValueError:
            roc_auc_closed = np.nan
            avg_precision_closed = np.nan
        
        metrics_closed = {
                "y_test_logits": y_test_logits,
                "y_test_pred": y_test_pred,
                "y_test_true": y_test_true,
                "roc_auc_closed": roc_auc_closed,
                "avg_precision_closed": avg_precision_closed,
                **{f"{k}_closed": v for k, v in binary_metrics.items()},
            }
        
        return metrics_closed



class SNN(SN10):

    def __init__(self, num_hidden_units: int, input_dim: int = 11*20):
        super().__init__()
        self.num_hidden_units = num_hidden_units
        self.input_dim = input_dim
        self.linear_1 = nn.Linear(input_dim, num_hidden_units)
        self.linear_2 = nn.Linear(num_hidden_units, 1)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.linear_1(x)
        x = self.activation(x)
        logits = self.linear_2(x)
        return logits


class MulticlassSN10(SN10):

    def __init__(self, num_classes: int):
        
        super().__init__()
        
        self.linear_2 = nn.Linear(10, num_classes)

        # Cross-entropy loss expects raw, not softmax
        # self.final = nn.Identity()
        self.softmax = nn.Softmax(dim=1)

        self.num_classes = num_classes
    
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
    
    def predict(
        self,
        x: torch.Tensor,
        ) -> torch.Tensor:
        return self.forward_prob(x).argmax(dim=1)
    
    def compute_metrics_closed_testset(self, x_test, y_test):
        y_test_pred_prob = self.forward_prob(x_test).detach().numpy()
        y_test_pred = y_test_pred_prob.argmax(axis=1)
        metrics_closed = {
            "acc_closed": metrics.accuracy_score(y_test, y_test_pred),
            "acc_balanced_closed": metrics.balanced_accuracy_score(y_test, y_test_pred),
            **{
                f"{str(func).split(' ')[1].split('_')[0]}_{str(avg_type)}_closed": func(
                    y_test, 
                    y_test_pred, 
                    average=avg_type
                    )
                for func in {
                    metrics.f1_score, 
                    metrics.precision_score, 
                    metrics.recall_score,
                    }
                for avg_type in {"micro", "macro", "weighted", None}
            },
            **{
                f"roc_auc_{avg_type}_closed": metrics.roc_auc_score(
                    y_test, 
                    y_test_pred_prob, 
                    average=avg_type,
                    multi_class="ovr",
                    )
                for avg_type in {"macro", "weighted", None}
            },
            
            "confusion_matrix_closed": metrics.confusion_matrix(
                y_test, 
                y_test_pred,
                ),
            "confusion_matrix_normed_closed": metrics.confusion_matrix(
                y_test,
                y_test_pred,
                normalize="all",
                ),
            "mcc_closed": metrics.matthews_corrcoef(y_test, y_test_pred),
            }
        return metrics_closed


class MulticlassSNN(MulticlassSN10):
    """Generalizes `MulticlassSN10` to a variable size of the hidden dimension.
    """    
    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__(num_classes=num_classes)
        self.linear_1 = nn.Linear(11*20, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, num_classes)
        self.hidden_dim = hidden_dim


class MultilabelSNN(MulticlassSNN):
    """Generalizes `MulticlassSN10` to a multilabel problem.
    Basically, we just need to change:
    - softmax -> sigmoid
    - the loss function CrossEntropy -> BCE.

    We also need custom evaluation metrics.

    """

    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__(hidden_dim=hidden_dim, num_classes=num_classes)
        self.softmax = nn.Identity()
    
    def compute_metrics_closed_testset(self, x_test, y_test):
        y_test_pred_prob = self(x_test).detach().numpy()
        y_test_pred = y_test_pred_prob.round()
        # metrics.multilabel_confusion_matrix(y_test, y_test_pred),
        return {
            "multilabel_fraction": ((y_test == 1.0).sum(axis=1) == 1).sum().item() / y_test.shape[0],
            **{
                f"{str(func).split(' ')[1].split('_')[0]}_{str(avg_type)}_closed": func(
                    y_test.reshape((-1, self.num_classes)),
                    y_test_pred,
                    average=avg_type
                    )
                for func in {
                    metrics.f1_score, 
                    metrics.precision_score, 
                    metrics.recall_score,
                    }
                for avg_type in {"micro", "macro", "weighted", None}
            },
        }


class CNN(nn.Module):
    def __init__(
        self,
        conv1_num_filters=5,
        conv1_filter_size=3,
        conv2_num_filters=3,
        conv2_filter_size=3,
        ):
        super().__init__()
        
        # ConvNet Calculator
        # https://madebyollin.github.io/convnet-calculator/

        # input: 11(W) x 20(H) x 1(#C)

        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=conv1_num_filters,  # filter count
            kernel_size=conv1_filter_size,  # filter size
        )
        conv1_out_w = math.floor((11-conv1_filter_size)/1 + 1)
        conv1_out_h = math.floor((20-conv1_filter_size)/1 + 1)

        self.pool = nn.MaxPool2d(
            kernel_size=2,  # filter size 
            stride=2,
        )
        pool1_out_w = math.floor((conv1_out_w-2)/2 + 1)
        pool1_out_h = math.floor((conv1_out_h-2)/2 + 1)
        
        self.conv2 = nn.Conv2d(
            in_channels=conv1_num_filters, 
            out_channels=conv2_num_filters,  # filter count
            kernel_size=conv2_filter_size,
        )
        conv2_out_w = math.floor((pool1_out_w-conv2_filter_size)/1 + 1)
        conv2_out_h = math.floor((pool1_out_h-conv2_filter_size)/1 + 1)

        pool2_out_w = math.floor((conv2_out_w-2)/2 + 1)
        pool2_out_h = math.floor((conv2_out_h-2)/2 + 1)
        fc1_in_features = pool2_out_w * pool2_out_h * conv2_num_filters
        self.fc1 = nn.Linear(fc1_in_features, 10)
        self.fc2 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, forward_logits=False):
        x = self.conv1(x)
        x = self.pool(F.relu(x))
        x = self.conv2(x)
        x = self.pool(F.relu(x))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        if forward_logits:
            return x
        else:
            x = self.sigmoid(x)
            return x
    
    def forward_logits(self, x):
        return self.forward(x, forward_logits=True)
    
    def compute_metrics_closed_testset(self, x_test, y_test):
        x_test_cnn = x_test.reshape((-1, 1, 11, 20))
        return SN10.compute_metrics_closed_testset_static(self, x_test_cnn, y_test)


class Transformer(nn.Module):
    """
    Text classifier based on a pytorch TransformerEncoder.
    """

    def __init__(
        self,
        vocab_size, 
        d_model,
        nhead=8,
        dim_feedforward=2048,
        num_layers=6,
        dropout=0.1,
        activation="relu",
        classifier_dropout=0.1,
        ):

        super().__init__()

        assert d_model % nhead == 0, "nheads must divide evenly into d_model"

        self.emb = nn.Embedding(vocab_size, d_model)

        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
            vocab_size=vocab_size,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.classifier = nn.Linear(d_model, 1)
        self.d_model = d_model

    def forward(self, x, return_logits=False):
        """Forward loop.

        Args:
            x (nn.Tensor): Input tensor of shape (batch_size, seq_len, vocab_size)
        """
        x = self.emb(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)

        x = x.mean(dim=1)
        x = self.classifier(x)
        if return_logits:
            return x
        else:
            x = nn.Sigmoid()(x)
            return x
    
    def forward_logits(self, x):
        return self.forward(x, return_logits=True)
    
    def compute_metrics_closed_testset(self, x_test, y_test):
        # The transformation below is specific for the transformer
        #  but is manually applied in loss computation and here. This is
        #  not ideal, but it works for now. Will have to be redesigned.
        #  Same for CNN.
        x_test_transformer = x_test.reshape(-1, 11, 20).argmax(axis=2).reshape(-1, 11)
        return SN10.compute_metrics_closed_testset_static(self, x_test_transformer, y_test)


class PositionalEncoding(nn.Module):
    """
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model, vocab_size=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(vocab_size, d_model)
        position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)


    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class Attributor:
    """Class for computing attributions for a given model.
    """

    def __init__(
        self, 
        model: nn.Module,
        type: str = "deep_lift",  # "integrated_gradients"
        baseline_type: str = "shuffle",  # "zero"
        compute_on: str = "expits",  # "logits"
        multiply_by_inputs: bool = True,
        name: Optional[str] = None,
        ):
        
        self.model = model
        self.compute_on = compute_on
        self.multiply_by_inputs = multiply_by_inputs

        if type == "deep_lift":
            self.attributor_class = DeepLift
        elif type == "integrated_gradients":
            self.attributor_class = IntegratedGradients
        else:
            raise ValueError(f"Unknown attributor type {type}")

        if compute_on == "expits":
            self.attributor = self.attributor_class(model, multiply_by_inputs=self.multiply_by_inputs)
        elif compute_on == "logits":
            # https://github.com/pytorch/captum/issues/678
            class LogitWrapper(nn.Module):
                def __init__(self, model):
                    super().__init__()  # mandatory
                    self.model = model
                def forward(self, *args):
                    return self.model.forward_logits(*args)
            
            wrapper = LogitWrapper(model)
            self.attributor = self.attributor_class(wrapper, multiply_by_inputs=self.multiply_by_inputs)
        else:
            raise ValueError(f"Unknown compute_on type {compute_on}")

        if baseline_type not in ["shuffle", "zero"]:
            raise ValueError(f"Unknown baseline type {baseline_type}")
        self.baseline_type = baseline_type

        if name is None:
            self.name = f"{type}__{compute_on}__{baseline_type}__multiply{multiply_by_inputs}"
        else:
            self.name = name


    def __call__(
        self,
        X: Union[torch.tensor, Dataset], 
        return_err: bool = False,
        ):
        
        if type(X) == Dataset:
            return self.attribute_dataset(X, return_err)
        elif type(X) == torch.Tensor:
            return self.attribute(X, return_err)
        else:
            raise ValueError(f"Unknown input type {type(X)}.")

    def attribute(
        self,
        X: torch.tensor,
        return_err: bool = False,
        return_baseline: bool = False,
        ):
        """Compute attributions for a given input using Integrated Gradients.
        """
        assert X.shape == (1, 220), f"Expected input shape (1, 220), got {X.shape}"
        
        if type(self.attributor) == IntegratedGradients:
            baseline = self.compute_baseline(X)
            attribution, err = self.attributor.attribute(
                inputs=X,
                baselines=baseline,
                method="gausslegendre",
                n_steps=100,
                return_convergence_delta=True,
            )
        elif type(self.attributor) == DeepLift:
            baseline = self.compute_baseline(X)
            attribution, err = self.attributor.attribute(
                inputs=X,
                baselines=baseline,
                return_convergence_delta=True,
            )

        if (not return_err) and (not return_baseline):
            return attribution
        else:
            res = [attribution]
            if return_err:
                res.append(err)
            if return_baseline:
                res.append(baseline)
            return tuple(res)


    def attribute_dataset(
        self,
        data: Dataset,
        ):
        """Compute Integrated Gradients attribution for a model on a dataset.

        Args:
            data (Dataset)

        Returns: list of tuples containing attributions and approximation errors (for integration).
        """

        indexes = data._get_indexes()
        records = []
        for index in indexes:
            attributions, approximation_error = self.attribute(
                data[index][0], 
                return_err=True
                )
            records.append((attributions, approximation_error))
        return records

    def compute_baseline(self, X: torch.tensor):
        if self.baseline_type == "shuffle":
            baseline = Attributor.compute_onehot_baseline_by_shuffling(X.reshape(11, 20)).reshape((1, 11*20))
        elif self.baseline_type == "zero":
            baseline = torch.zeros(X.shape)
        return baseline

    @staticmethod
    def shuffle_rows(tensor: torch.tensor) -> torch.tensor:
        tensor_shape = tensor.shape
        tensor = np.copy(tensor)  # create a copy to avoid shuffling the original tensor
        index = np.arange(tensor.shape[0])
        np.random.shuffle(index)
        return torch.tensor(tensor[index,:])

    @staticmethod
    def compute_onehot_baseline_by_shuffling(
        onehot_tensor: torch.tensor,
        num_shuffles: int = 1000,
        ):
        """Compute a baseline for one-hot encoded tensor by shuffling the rows
        and averaging the resulting one-hot encodings.
        """
        shuffles = []
        for _ in range(num_shuffles):
            shuffles.append(
                Attributor.shuffle_rows(onehot_tensor)
            )
        baseline = sum(shuffles) / num_shuffles
        return baseline


AVAILABLE_MODELS = [SN10, SNN, MulticlassSN10, MulticlassSNN, MultilabelSNN, CNN, Transformer]


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
        
        loss = compute_loss(model, loss_fn, X, y)

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

    assert type(model) in AVAILABLE_MODELS, (
        f"Model class {type(model)} not recognized."
    )

    test_loss = compute_avg_test_loss(loader, model, loss_fn)

    loop_metrics = {
        "test_loss": test_loss,
    }

    x_test, y_test = Xy_from_loader(loader=loader)
    closed_metrics: dict = compute_metrics_closed_testset(model, x_test, y_test)

    acc_closed = closed_metrics.get('acc_closed', np.nan)
    print(
        f"Test Error: \n Acc: {100*acc_closed:.1f} Avg loss: {test_loss:>8f} \n"
    )

    return {
        **loop_metrics,
        **closed_metrics,
    }


def compute_avg_test_loss(loader, model, loss_fn):
    num_batches = len(loader)
    test_loss = 0
    with torch.no_grad():
        for X, y in loader:
            test_loss += compute_loss(model, loss_fn, X, y).item()
    test_loss /= num_batches
    return test_loss


def compute_loss(model, loss_fn, X, y):
    if type(loss_fn) == nn.CrossEntropyLoss:
        loss = loss_fn(model(X), y.reshape(-1))
    elif type(loss_fn) == nn.BCELoss:
        has_num_classes = hasattr(model, "num_classes")
        if has_num_classes:
            y_hat = model(X)
            # y = F.one_hot(y, num_classes=model.num_classes).reshape(y_hat.shape)
            loss = loss_fn(y_hat, y.reshape(y_hat.shape).type(torch.float))
        else:
            # binary case

            if hasattr(model, "conv1"):
                # hack for CNN
                X_pred = model(X.reshape(-1, 1, 11, 20))
                loss = loss_fn(X_pred, y)
            elif hasattr(model, "transformer_encoder"):
                # hack for Transformer
                X_pred = model(X.reshape(-1, 11, 20).argmax(axis=2).reshape(-1, 11))
                loss = loss_fn(X_pred, y)
            else:
                loss = loss_fn(model(X), y)

    else:
        raise NotImplementedError(f"{loss_fn=} not implemented.")
    return loss


def openset_loop(open_loader, test_loader, model):
    x_open, _ = Xy_from_loader(open_loader)
    x_test, _ = Xy_from_loader(test_loader)
    open_metrics = compute_metrics_open_testset(model, x_open, x_test)
    return open_metrics


def Xy_from_loader(loader: DataLoader):
    X, y = list(
        DataLoader(loader.dataset, batch_size=len(loader.dataset))
        )[0]
    return X, y


def construct_dataset_loader_multiclass(
    df: pd.DataFrame,
    batch_size: int = 64,
    ):
    dataset = datasets.MulticlassDataset(df.reset_index(drop=True))
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    return dataset, loader


def construct_dataset_loader(
    df: pd.DataFrame,
    batch_size: int = 64,
    dataset_class = datasets.MulticlassDataset,
    ):
    dataset = dataset_class(df.reset_index(drop=True))
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    return dataset, loader


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
    optimizer_type: str,
    momentum: float = 0,
    weight_decay: float = 0,
    callback_on_model_end_epoch: callable = None,
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
    if callback_on_model_end_epoch is None:
        callback_on_model_end_epoch = lambda x, t: None

    online_metrics_per_epoch = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        losses = train_loop(train_loader, model, loss_fn, optimizer)
        # 2 lines below replace with evaluate_on_closed_and_open_testsets
        test_metrics = test_loop(test_loader, model, loss_fn)
        
        if open_loader is not None:
            open_metrics = openset_loop(open_loader, test_loader, model)
        else:
            open_metrics = {}
        
        online_metrics_per_epoch.append({
            "train_losses": losses,
            "test_metrics": test_metrics,
            "open_metrics": open_metrics,
        })

        callback_on_model_end_epoch(model, t)

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
    assert type(model) in AVAILABLE_MODELS

    metrics_closed = model.compute_metrics_closed_testset(x_test, y_test)
    return metrics_closed


def compute_metrics_open_testset(model, x_open, x_test) -> dict:
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

    x_test, y_test = Xy_from_loader(test_loader)
    metrics_closed = compute_metrics_closed_testset(model, x_test, y_test)
    
    if open_loader is not None:
        x, y = Xy_from_loader(open_loader)
        metrics_open = compute_metrics_open_testset(model, x, x_test)
    else:
        metrics_open = {}

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
