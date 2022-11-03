import logging
import multiprocessing
from pathlib import Path
import re
from typing import List, Optional
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import mlflow

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

import NegativeClassOptimization.config as config
import NegativeClassOptimization.utils as utils
import NegativeClassOptimization.datasets as datasets
import NegativeClassOptimization.preprocessing as preprocessing
import NegativeClassOptimization.ml as ml
from NegativeClassOptimization import visualisations


TEST = False

experiment_id = 6
run_name = "v0.1.2_2"
sample_train = 10000  # 70000
batch_size = 64
epochs = 20
learning_rate = 0.01
num_processes = 20


def multiprocessing_wrapper_script_08(
    ags: List[str],
    ):
    with mlflow.start_run(
        experiment_id=experiment_id, 
        run_name=run_name, 
        description=" vs ".join(ags),
        ):

        run_main_08(
            epochs, 
            learning_rate, 
            ags, 
            save_model=True,
            sample = (1000 if TEST else None),
            sample_train=sample_train if type(sample_train) == int else None,
            )


def run_main_08(
    epochs, 
    learning_rate, 
    ags, 
    save_model = True,
    sample = None,
    sample_train = None,
    ):
    
    logger = logging.getLogger()
    logger.info("Start run_main_08")

    mlflow.log_params({
        "epochs": epochs,
        "learning_rate": learning_rate,
        "optimizer_type": "Adam",
        "momentum": 0.9,
        "weight_decay": 0,
        "batch_size": batch_size,
        "ags": "_".join(ags),
        "k": len(ags),
        "sample": sample,
        "sample_train": sample_train,
    })

    ###########

    dfs = utils.load_processed_dataframes(sample=sample)

    df_train = dfs["train_val"]
    df_train, scaler, encoder = preprocess_df_for_multiclass(df_train, ags, sample_train=sample_train)
    mlflow.log_params({"encoder_classes": "_".join(encoder.classes_)})
    

    df_test = dfs["test_closed_exclusive"]
    df_test, _, _ = preprocess_df_for_multiclass(
        df_test,
        ags,
        scaler,
        encoder
    )


    _, train_loader = construct_dataset_loader(df_train, batch_size)
    _, test_loader = construct_dataset_loader(df_test, batch_size)
    _, open_loader = preprocessing.construct_open_dataset_loader(
        dfs["test_open_exclusive"],
        batch_size=batch_size,
        scaler=scaler
    )
    mlflow.log_params({
            "N_train": len(train_loader.dataset),
            "N_closed": len(test_loader.dataset),
            "N_open": len(open_loader.dataset),
        })

    ####

    model = ml.MulticlassSN10(num_classes=len(ags))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = ml.construct_optimizer(
        optimizer_type="Adam",
        learning_rate=0.01,
        momentum=0.9,
        weight_decay=0,
        model=model,
    )

    online_metrics = []
    for t in range(epochs):
        
        print(f"Epoch {t+1}\n-------------------------------")
        
        losses = ml.train_loop(train_loader, model, loss_fn, optimizer)
        test_metrics = ml.test_loop(test_loader, model, loss_fn)
        open_metrics = ml.openset_loop(open_loader, test_loader, model)
        
        online_metrics.append({
                "train_losses": losses,
                "test_metrics": test_metrics,
                "open_metrics": open_metrics,
            })
    utils.mlflow_log_params_online_metrics(online_metrics)
    eval_metrics = ml.evaluate_on_closed_and_open_testsets(
        open_loader, 
        test_loader, 
        model
        )
    mlflow.log_dict(
        {
            **{k1: v1.tolist() if type(v1) == np.ndarray else v1 for k1, v1 in eval_metrics["closed"].items()},
            **{k2: v2.tolist() if type(v2) == np.ndarray else v2 for k2, v2 in eval_metrics["open"].items()},
        }, 
        "eval_metrics.json"
    )

    mlflow.log_metrics(
        {
            k1: v1.tolist() for k1, v1 in eval_metrics["closed"].items() if type(v1) != np.ndarray
        }
    )
    # Instead of mlflow.log_metrics(eval_metrics["open"])
    #  does some renaming.
    mlflow.log_metrics(
        {
            'open_avg_precision': eval_metrics["open"]["avg_precision_open"],
            'open_acc': eval_metrics["open"]["acc_open"],
            'open_recall': eval_metrics["open"]["recall_open"],
            'open_precision': eval_metrics["open"]["precision_open"],
            'open_f1': eval_metrics["open"]["f1_open"],
        }
    )

    # Other artifacts
    x_test, y_test = ml.Xy_from_loader(test_loader)
    y_test_pred = model.predict(x_test)
    report: dict = metrics.classification_report(
        y_test, 
        y_test_pred,
        target_names=encoder.classes_,
        output_dict=True
        )
    mlflow.log_dict(report, "classification_report.json")

    fig_abs_logit_distr, _ = visualisations.plot_abs_logit_distr(
            eval_metrics["open"], 
            metadata={
                "ag_pos": ags,
                "ag_neg": "",
                "N_train": len(train_loader.dataset),
                "N_closed": len(test_loader.dataset),
                "N_open": len(open_loader.dataset),
            },
        )
    mlflow.log_figure(fig_abs_logit_distr, "fig_abs_logit_distr.png")
    
    fig_confusion_matrices, _ = visualisations.plot_confusion(
        cm=eval_metrics["closed"]["confusion_matrix_closed"],
        cm_normed=eval_metrics["closed"]["confusion_matrix_normed_closed"],
        class_names=encoder.classes_,
    )
    mlflow.log_figure(fig_confusion_matrices, "fig_confusion_matrices.png")

    if save_model:
        mlflow.pytorch.log_model(model, "pytorch_model")

    return model


def preprocess_df_for_multiclass(
    df,
    ags: List[str],
    scaler = None,
    encoder = None,
    sample_train = None,
    ):
    
    df = df.loc[df["Antigen"].isin(ags)].copy()


    df = preprocessing.remove_duplicates_for_multiclass(df)    
    if sample_train is not None:
        df = preprocessing.sample_train_val(df, sample_train)

    df = preprocessing.onehot_encode_df(df)

    arr = preprocessing.arr_from_list_series(df["Slide_onehot"])
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(arr)
    df["X"] = scaler.transform(arr).tolist()

    if encoder is None:
        antigens = df["Antigen"].unique().tolist()
        encoder = LabelEncoder().fit(antigens)

    df["y"] = encoder.transform(df["Antigen"])
    df = df[["X", "y"]]
    return df, scaler, encoder


def construct_dataset_loader(
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


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s %(process)d %(funcName)s %(levelname)s %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(
                filename="data/logs/08.log",
                mode="a",
            ),
            logging.StreamHandler()
        ]
    )

    utils.nco_seed()

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    experiment = mlflow.set_experiment(experiment_id=experiment_id)

    if TEST:
        ags = [config.ANTIGENS_CLOSEDSET[:3], config.ANTIGENS_CLOSEDSET[:5]]
    else:
        atoms = datasets.construct_dataset_atoms(config.ANTIGENS_CLOSEDSET)
        atoms = list(filter(lambda atom: len(atom) > 2, atoms))
        np.random.shuffle(atoms)
        ags = atoms[:]

    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(multiprocessing_wrapper_script_08, ags)
