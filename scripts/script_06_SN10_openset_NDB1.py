import multiprocessing
from itertools import combinations
from pathlib import Path
from typing import Optional, Union, List

import mlflow
import NegativeClassOptimization.config as config
import NegativeClassOptimization.ml as ml
import NegativeClassOptimization.preprocessing as preprocessing
import NegativeClassOptimization.utils as utils
import NegativeClassOptimization.visualisations as vis
import numpy as np
import pandas as pd
import torch
import yaml
import farmhash
import logging


## PARAMETERS
params_06 = config.PARAMS["06_SN10_openset_NDB1"]

experiment_id = params_06["experiment_id"]
run_name = params_06["run_name"]
num_processes = params_06["num_processes"]
epochs = params_06["epochs"]
learning_rate = params_06["learning_rate"]
add_reverse_pos_neg = params_06["add_reverse_pos_neg"]
sample_train = params_06["sample_train"]


def multiprocessing_wrapper_run_main_06(
    ag_pair, 
    experiment_id=experiment_id, 
    run_name=run_name, 
    epochs=epochs, 
    learning_rate=learning_rate, 
    sample_train=sample_train,
    ):
    """Function to multiprocess the workflow.

    Args:
        ag_pair ()
        experiment_id (optional): Defaults to experiment_id.
        run_name (optional): Defaults to run_name.
        epochs (optional): Defaults to epochs.
        learning_rate (optional): Defaults to learning_rate.
    """    
    logger = logging.getLogger()
    
    ag_pos, ag_neg = ag_pair
    logger.info(f"Start run for ({ag_pos}, {ag_neg})")

    with mlflow.start_run(
        experiment_id=experiment_id, 
        run_name=run_name, 
        description=f"{ag_pos} vs {ag_neg}"
        ):

        run_main_06(
            epochs, 
            learning_rate, 
            ag_pos, 
            ag_neg, 
            save_model=True,
            sample_train=sample_train if type(sample_train) == int else None,
            )


def resolve_ag_type(ag: Union[str, List[str]]) -> List[str]:
    """Utility function to convert ag representation to list.

    Args:
        ag (Union[str, List[str]])

    Raises:
        ValueError: if ag of not supported type.

    Returns:
        List[str]
    """    
    if type(ag) == list:
        return ag
    elif type(ag) == str:
        return [ag]
    else:
        raise ValueError(f"ag type {type(ag)} not recognized.")


def run_main_06(
    epochs, 
    learning_rate, 
    ag_pos: Union[str, List[str]],
    ag_neg: Union[str, List[str]],
    optimizer_type = "Adam",
    momentum = 0,
    weight_decay = 0,
    batch_size = 64,
    save_model = False,
    sample = None,
    sample_train = None,
    ):

    logger = logging.getLogger()

    ag_pos = resolve_ag_type(ag_pos)
    ag_neg = resolve_ag_type(ag_neg)

    mlflow.log_params({
            "epochs": epochs,
            "learning_rate": learning_rate,
            "optimizer_type": optimizer_type,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "ag_pos": ag_pos,
            "ag_neg": ag_neg,
            "sample": sample,
            "sample_train": sample_train,
        })

    
    processed_dfs: dict = utils.load_processed_dataframes(
        sample=sample
        )
    
    train_loader, test_loader, open_loader = construct_loaders_06(
        processed_dfs, 
        ag_pos, 
        ag_neg, 
        batch_size=batch_size,
        sample_train=sample_train,
        )
    logger.info(f"Loaders ready.")

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
    logger.info("Model trained.")
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
    mlflow.log_dict(
        {
            **{k1: v1.tolist() if type(v1) == np.ndarray else v1 for k1, v1 in eval_metrics["closed"].items()},
            **{k2: v2.tolist() if type(v2) == np.ndarray else v2 for k2, v2 in eval_metrics["open"].items()},
        }, 
        "eval_metrics.json"
    )
    mlflow.log_metrics(
        {
            'open_avg_precision' :eval_metrics["open"]["avg_precision_open"],
            'open_acc' :eval_metrics["open"]["acc_open"],
            'open_recall' :eval_metrics["open"]["recall_open"],
            'open_precision' :eval_metrics["open"]["precision_open"],
            'open_f1' :eval_metrics["open"]["f1_open"],
            'open_fpr_abs_logit_model' :eval_metrics["open"]["fpr_abs_logit_model"],
            'open_fpr_naive_model' :eval_metrics["open"]["fpr_naive_model"],
        }
    )


    metadata={
            "ag_pos": ag_pos,
            "ag_neg": ag_neg,  # ok to be a list
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

    fig_pr, _ = vis.plot_pr_open_and_closed_testsets(eval_metrics, metadata=metadata)
    mlflow.log_figure(fig_pr, "fig_pr.png")

    if save_model:
        mlflow.pytorch.log_model(model, "pytorch_model")

    return model


def construct_loaders_06(
    processed_dfs, 
    ag_pos: Union[str, List[str]], 
    ag_neg: Union[str, List[str]], 
    batch_size: int,
    sample_train: Optional[int] = None,
    ):

    ag_pos: List[str] = resolve_ag_type(ag_pos)
    ag_neg: List[str] = resolve_ag_type(ag_neg)

    df_train_val: pd.DataFrame = processed_dfs["train_val"]
    df_train_val = df_train_val.loc[df_train_val["Antigen"].isin([*ag_pos, *ag_neg])]
    
    df_test_closed = processed_dfs["test_closed_exclusive"]
    df_test_closed = df_test_closed.loc[df_test_closed["Antigen"].isin([*ag_pos, *ag_neg])]
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
            ag_pos=ag_pos,
            scale_onehot=True,
            batch_size=batch_size,
            df_test_open=df_test_open,
            sample_train=sample_train,
        )
    
    return train_loader, test_loader, open_loader


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s %(process)d %(funcName)s %(levelname)s %(message)s",
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(
                filename="data/logs/06.log",
                mode="a",
            ),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    logger.info(f"Start")

    utils.nco_seed()

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    experiment = mlflow.set_experiment(experiment_id=experiment_id)

    ag_pairs = []
    for (ag_pos, ag_neg) in combinations(config.ANTIGENS_CLOSEDSET, 2):
        ag_pairs.append((ag_pos, ag_neg))
        if add_reverse_pos_neg:
            ag_pairs.append((ag_neg, ag_pos))

    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(multiprocessing_wrapper_run_main_06, ag_pairs)
