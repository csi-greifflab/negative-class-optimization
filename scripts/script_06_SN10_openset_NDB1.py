from pathlib import Path
import json
from itertools import combinations
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import mlflow

import NegativeClassOptimization.config as config
import NegativeClassOptimization.preprocessing as preprocessing
import NegativeClassOptimization.ml as ml
import NegativeClassOptimization.visualisations as vis


## CONSTANTS
data_path = Path(config.DATA_SLACK_1_GLOBAL)
out_path = Path("data/SN10_openset_NDB1")


## PARAMETERS
experiment_id = 2
run_name = "DEV"
num_processes = 20
epochs = 30
learning_rate = 0.01
ag_pos = "1FBI"
ag_neg = "1NSN"


def run_main(
    ag_pair, 
    experiment_id=experiment_id, 
    run_name=run_name, 
    epochs=epochs, 
    learning_rate=learning_rate, 
    data_path=data_path, 
    out_path=out_path
    ):
    
    ag_pos, ag_neg = ag_pair
    with mlflow.start_run(
        experiment_id=experiment_id, 
        run_name=run_name, 
        description=f"{ag_pos} vs {ag_neg}"
        ):
        mlflow.log_params({
            "epochs": 5,
            "learning_rate": 0.01,
            "ag_pos": ag_pos,
            "ag_neg": ag_neg,
        })

        out_path_i = out_path / f"{ag_pos}_vs_{ag_neg}"
        out_path_i.mkdir(exist_ok=True)

        ## ETL
        df = pd.read_csv(data_path, sep='\t')
        df_closed = df.loc[df["Antigen"].isin([ag_pos, ag_neg])].copy()
        df_open = df.loc[df["Antigen"].isin(config.ANTIGENS_OPENSET)].copy()
        df_open = df_open.drop_duplicates(["Slide"], keep="first")
        df_open = df_open.loc[~df_open["Slide"].isin(df_closed["Slide"])]
        df_open = df_open.reset_index(drop=True)
        
        (   
            _,
            _,
            _,
            train_loader,
            test_loader,
            open_loader,
            ) = preprocessing.preprocess_data_for_pytorch_binary(
                df_closed,
                [ag_pos],
                scale_onehot=True,
                df_openset=df_open,
        )

        mlflow.log_params({
            "N_train": len(train_loader.dataset),
            "N_closed": len(test_loader.dataset),
            "N_open": len(open_loader.dataset),
        })

        ## TRAIN
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = ml.SN10().to(device)
        online_metrics = ml.train_for_ndb1(epochs, learning_rate, train_loader, test_loader, open_loader, model)
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

        ## EVALUATE
        eval_metrics = ml.evaluate_on_closed_and_open_testsets(open_loader, test_loader, model)

        ## PLOTS
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


if __name__ == "__main__":

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    experiment = mlflow.set_experiment(experiment_id=experiment_id)

    ag_pairs = []
    for (ag_pos, ag_neg) in combinations(config.ANTIGENS, 2):
        ag_pairs.append((ag_pos, ag_neg))

    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(run_main, ag_pairs)
