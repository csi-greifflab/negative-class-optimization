import multiprocessing
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math


import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F


from NegativeClassOptimization import ml
from NegativeClassOptimization import utils
from NegativeClassOptimization import preprocessing
from NegativeClassOptimization import config


TEST = False
experiment_id = 10
run_name = "dev-v0.1.2"
num_processes = 20
num_splits = 5
num_seeds = 5

N = 20000
ag_pos = "3VRL"
ag_neg = "1ADQ"

epochs = 50
learning_rate = 0.001
optimizer_type = "Adam"
momentum = 0.9
weight_decay = 0
batch_size = 64


def get_data(ag_pos, ag_neg, N):
    # df = utils.load_global_dataframe()

    # df = df.loc[df["Antigen"].isin([ag_pos, ag_neg])].copy()
    # df = df.drop_duplicates(["Slide"])

    # df = df.sample(n=N, random_state=42)
    # df = df.sample(frac=1, random_state=42)

    # df.reset_index(drop=True, inplace=True)
    df = utils.load_1v1_binary_dataset(
        ag_pos=ag_pos,
        ag_neg=ag_neg,
        num_samples=N,
    )
    return df


def build_models():
    models = {
        "SN10": ml.SN10(),
        "SN48": ml.SNN(num_hidden_units=48),
        "CNN": ml.CNN(
            conv1_num_filters=16,
            conv1_filter_size=3,
            conv2_num_filters=12,
            conv2_filter_size=3,
        ),
        "CNN+": ml.CNN(
            conv1_num_filters=32,
            conv1_filter_size=3,
            conv2_num_filters=32,
            conv2_filter_size=3,
        ),
        "Transformer": ml.Transformer(
            vocab_size=20, 
            d_model=12,
            nhead=2,
            dim_feedforward=10,
            num_layers=2,
            dropout=0.1,
            activation="relu",
            classifier_dropout=0.1,
        ),
        "Transformer+": ml.Transformer(
            vocab_size=20, 
            d_model=18,
            nhead=3,
            dim_feedforward=30,
            num_layers=4,
            dropout=0.1,
            activation="relu",
            classifier_dropout=0.1,
        ),
        "Transformer_FF": ml.Transformer(
            vocab_size=20, 
            d_model=8,
            nhead=2,
            dim_feedforward=42,
            num_layers=2,
            dropout=0.1,
            activation="relu",
            classifier_dropout=0.1,
        ),
        "Transformer_FF+": ml.Transformer(
            vocab_size=20, 
            d_model=12,
            nhead=3,
            dim_feedforward=74,
            num_layers=4,
            dropout=0.1,
            activation="relu",
            classifier_dropout=0.1,
        ),
    }
    return models


def init_model(model_name: str) -> nn.Module:
    models = build_models()
    return models[model_name]

def get_model_names():
    models = build_models()
    return list(models.keys())


def multiprocessing_wrapper_script_09(run_name, df, model_name, split_id, seed_id):
    
    with mlflow.start_run(
        experiment_id=experiment_id, 
        run_name=run_name, 
        description=f"{ag_pos} vs {ag_neg}",
        tags={"mlflow.runName": run_name},
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
            "model_name": model_name,
            "split_id": split_id,
            "seed_id": seed_id,
        })

        df_train_val = df.loc[df["Slide_farmhash_mod_10"] != split_id].copy()
        df_test_closed = df.loc[df["Slide_farmhash_mod_10"] == split_id].copy()
        train_data, test_data, train_loader, test_loader = (
            preprocessing.preprocess_data_for_pytorch_binary(
                df_train_val=df_train_val,
                df_test_closed=df_test_closed,
                ag_pos=[ag_pos],
                batch_size=64,
                scale_onehot=False,
        ))
        mlflow.log_params({
            "N_train": len(train_loader.dataset),
            "N_closed": len(test_loader.dataset),
        })

        torch.manual_seed(seed_id)
        model = init_model(model_name)
        model = model.to("cpu")

        ## Adjust train_for_ndb1 w/o open set
        online_metrics = ml.train_for_ndb1(
            epochs,
            learning_rate, 
            train_loader, 
            test_loader, 
            None,  # open_loader
            model,
            optimizer_type=optimizer_type,
            momentum=momentum,
            weight_decay=weight_decay,
            )
        
        utils.mlflow_log_params_online_metrics(online_metrics)

        eval_metrics = ml.evaluate_on_closed_and_open_testsets(
            None,  # open_loader 
            test_loader, 
            model)
        mlflow.log_dict(
            {
                **{k1: v1.tolist() if type(v1) == np.ndarray else v1 for k1, v1 in eval_metrics["closed"].items()},
                **{k2: v2.tolist() if type(v2) == np.ndarray else v2 for k2, v2 in eval_metrics["open"].items()},
            }, 
            "eval_metrics.json"
        )


if __name__ == "__main__":

    utils.nco_seed()
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    experiment = mlflow.set_experiment(experiment_id=experiment_id)

    df = get_data(ag_pos, ag_neg, N)
    df["Slide_farmhash_mod_10"] = df["Slide"].apply(lambda x: preprocessing.farmhash_mod_10(x))

    if TEST:
        model_name = "SN10"
        split_id = 0
        seed_id = 0
        multiprocessing_wrapper_script_09(run_name, df, model_name, split_id, seed_id)
    
    else:    
        # Build run_params
        model_names = get_model_names()
        run_params = []
        for split_id in range(num_splits):
            for seed_id in range(num_seeds):
                for model_name in model_names:
                    run_params.append({
                        "model_name": model_name,
                        "split_id": split_id,
                        "seed_id": seed_id,
                    })

        # Run batched multiprocessing
        for i in range(0, len(run_params), num_processes):
            params_batch = run_params[i:i+num_processes]
            with multiprocessing.Pool(processes=num_processes) as pool:
                pool.starmap(
                    multiprocessing_wrapper_script_09,
                    [
                        (
                            run_name,
                            df,
                            params["model_name"],
                            params["split_id"],
                            params["seed_id"],
                        )
                        for params in params_batch
                    ]
                )