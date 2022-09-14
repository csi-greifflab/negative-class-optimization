import multiprocessing
from itertools import combinations
from pathlib import Path

import mlflow
import NegativeClassOptimization.config as config
import NegativeClassOptimization.ml as ml
import NegativeClassOptimization.preprocessing as preprocessing
import NegativeClassOptimization.utils as utils
import NegativeClassOptimization.visualisations as vis
import pandas as pd
import torch
import yaml


## PARAMETERS
params_06 = config.PARAMS["06_SN10_openset_NDB1"]

experiment_id = params_06["experiment_id"]
run_name = params_06["run_name"]
num_processes = params_06["num_processes"]
epochs = params_06["epochs"]
learning_rate = params_06["learning_rate"]


def multiprocessing_wrapper_run_main_06(
    ag_pair, 
    experiment_id=experiment_id, 
    run_name=run_name, 
    epochs=epochs, 
    learning_rate=learning_rate, 
    ):
    
    ag_pos, ag_neg = ag_pair
    with mlflow.start_run(
        experiment_id=experiment_id, 
        run_name=run_name, 
        description=f"{ag_pos} vs {ag_neg}"
        ):

        run_main_06(epochs, learning_rate, ag_pos, ag_neg)


def run_main_06(
    epochs, 
    learning_rate, 
    ag_pos, 
    ag_neg,
    optimizer_type = "SGD",
    momentum = 0,
    weight_decay = 0,
    batch_size = 64,
    save_model = False,
    sample = None,
    ):

    mlflow.log_params({
            "epochs": epochs,
            "learning_rate": learning_rate,
            "optimizer_type": optimizer_type,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "ag_pos": ag_pos,
            "ag_neg": ag_neg,
        })

    
    processed_dfs: dict = utils.load_processed_dataframes(sample=sample)
    train_loader, test_loader, open_loader = construct_loaders_06(
        processed_dfs, 
        ag_pos, 
        ag_neg, 
        batch_size=batch_size,
        )

    mlflow.log_params({
            "N_train": len(train_loader.dataset),
            "N_closed": len(test_loader.dataset),
            "N_open": len(open_loader.dataset),
        })

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ml.SN10().to(device)
    online_metrics = ml.train_for_ndb1(
        epochs, 
        learning_rate, 
        train_loader, 
        test_loader, 
        open_loader, 
        model,
        optimizer_type=optimizer_type,
        momentum=momentum,
        weight_decay=weight_decay,
        )
    for i, epoch_metrics in enumerate(online_metrics):
        epoch = i+1
        mlflow.log_metrics(
                {
                    "train_loss": epoch_metrics["train_losses"][-1],
                    "test_loss": epoch_metrics["test_metrics"]["test_loss"],
                    "test_acc": epoch_metrics["test_metrics"]["accuracy"],
                    "closed_roc_auc": epoch_metrics["test_metrics"]["roc_auc_closed"],
                    "closed_recall": epoch_metrics["test_metrics"]["recall_closed"],
                    "closed_precision": epoch_metrics["test_metrics"]["precision_closed"],
                    "closed_f1": epoch_metrics["test_metrics"]["f1_closed"],
                    "open_roc_auc": epoch_metrics["open_metrics"]["roc_auc_open"],
                }, 
                step=epoch
            )

    eval_metrics = ml.evaluate_on_closed_and_open_testsets(open_loader, test_loader, model)
    # TODO: log eval_metrics

    metadata={
            "ag_pos": ag_pos,
            "ag_neg": ag_neg,
            "N_train": len(train_loader.dataset),
            "N_closed": len(test_loader.dataset),
            "N_open": len(open_loader.dataset),
        }
    fig_abs_logit_distr, ax_abs_logit_distr = vis.plot_abs_logit_distr(
            eval_metrics, 
            metadata=metadata,
        )
    mlflow.log_figure(fig_abs_logit_distr, "fig_abs_logit_distr.png")
        
    fig_roc, _ = vis.plot_roc_open_and_closed_testsets(eval_metrics, metadata=metadata)
    mlflow.log_figure(fig_roc, "fig_roc.png")

    if save_model:
        mlflow.pytorch.log_model(model, "pytorch_model")

    return model


def construct_loaders_06(processed_dfs, ag_pos, ag_neg, batch_size):
    df_train_val = processed_dfs["train_val"]
    df_train_val = df_train_val.loc[df_train_val["Antigen"].isin([ag_pos, ag_neg])]
    df_test_closed = processed_dfs["test_closed_exclusive"]
    df_test_closed = df_test_closed.loc[df_test_closed["Antigen"].isin([ag_pos, ag_neg])]
    df_test_open = processed_dfs["test_open_exclusive"]
    df_test_open = df_test_open.drop_duplicates(["Slide"], keep="first").reset_index(drop=True)

    (   
        _,
        _,
        _,
        train_loader,
        test_loader,
        open_loader,
        ) = preprocessing.preprocess_data_for_pytorch_binary(
            df_train_val=df_train_val,
            df_test_closed=df_test_closed,
            ag_pos=[ag_pos],
            scale_onehot=True,
            batch_size=batch_size,
            df_test_open=df_test_open,
        )
    
    return train_loader, test_loader, open_loader


if __name__ == "__main__":

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    experiment = mlflow.set_experiment(experiment_id=experiment_id)

    ag_pairs = []
    for (ag_pos, ag_neg) in combinations(config.ANTIGENS_CLOSEDSET, 2):
        ag_pairs.append((ag_pos, ag_neg))

    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(multiprocessing_wrapper_run_main_06, ag_pairs)
