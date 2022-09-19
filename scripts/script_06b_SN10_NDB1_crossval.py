import multiprocessing
from itertools import combinations
import mlflow
import NegativeClassOptimization.config as config
import NegativeClassOptimization.ml as ml
import NegativeClassOptimization.preprocessing as preprocessing
import NegativeClassOptimization.utils as utils
import NegativeClassOptimization.visualisations as vis
import numpy as np
import pandas as pd
import torch
import optuna

from script_06_SN10_openset_NDB1 import construct_loaders_06, run_main_06


# PARAMETERS
params_06b = config.PARAMS["06b_SN10_NDB1_crossval"]

experiment_id = params_06b["experiment_id"]
run_name = params_06b["run_name"]
num_processes = params_06b["num_processes"]
epochs_val = params_06b["epochs_val"]
epochs_train = params_06b["epochs_train"]
learning_rate_range = params_06b["learning_rate_range"]
batch_sizes = params_06b["batch_sizes"]
momentum_range = params_06b["momentum_range"]
weight_decays = params_06b["weight_decays"]
n_trials = params_06b["n_trials"]

def run_main_06b(
    ag_pair, 
    farmhash_mod_10_val_mask: int = config.FARMHASH_MOD_10_VAL_MASK,
    experiment_id = experiment_id,
    run_name = run_name,
    epochs_val = epochs_val,
    epochs_train = epochs_train,
    learning_rate_range = learning_rate_range,
    batch_sizes = batch_sizes,
    momentum_range = momentum_range,
    weight_decays = weight_decays,
    n_trials = n_trials,
    ):

    ag_pos, ag_neg = ag_pair

    def crossval_objective(trial):
        
        learning_rate = trial.suggest_float(
            "learning_rate", 
            learning_rate_range[0], 
            learning_rate_range[1],
            log=True)
        
        batch_size = trial.suggest_categorical(
            "batch_size",
            batch_sizes
        )
        
        momentum = trial.suggest_float(
            "momentum",
            momentum_range[0],
            momentum_range[1],
            log=False,
            step=0.01,
        )

        weight_decay = trial.suggest_categorical(
            "weight_decay",
            weight_decays
        )

        with mlflow.start_run(
            experiment_id=experiment_id, 
            run_name=f"{run_name}_val", 
            description=f"{ag_pos} vs {ag_neg}",
            nested=True,
            ):
            
            mlflow.log_params({
                "epochs": epochs_val,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "momentum": momentum,
                "weight_decay": weight_decay,
                "ag_pos": ag_pos,
                "ag_neg": ag_neg,
            })
            
            test_loader, open_loader, train_loader, val_loader = construct_loaders_06b(
                farmhash_mod_10_val_mask, 
                ag_pos, 
                ag_neg,
                train_batch_size=batch_size
                )


            mlflow.log_params({
                "N_train": len(train_loader.dataset),
                "N_val": len(val_loader.dataset),
                "N_closed": len(test_loader.dataset),
                "N_open": len(open_loader.dataset),
            })


            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = ml.SN10().to(device)
            online_metrics = ml.train_for_ndb1(
                epochs_val, 
                learning_rate, 
                train_loader, 
                val_loader, 
                open_loader, 
                model,
                optimizer_type="Adam",
                momentum=momentum,
                weight_decay=weight_decay,
                )

            val_acc = online_metrics[-1]["test_metrics"]["accuracy"]
            crossval_optimizaion_metric = val_acc
            for i, epoch_metrics in enumerate(online_metrics):
                epoch = i+1
                mlflow.log_metrics(
                    {
                        "train_loss": epoch_metrics["train_losses"][-1],
                        "val_loss": epoch_metrics["test_metrics"]["test_loss"],
                        "val_acc": epoch_metrics["test_metrics"]["accuracy"],
                        "val_roc_auc": epoch_metrics["test_metrics"]["roc_auc_closed"],
                        "val_recall": epoch_metrics["test_metrics"]["recall_closed"],
                        "val_precision": epoch_metrics["test_metrics"]["precision_closed"],
                        "val_f1": epoch_metrics["test_metrics"]["f1_closed"],
                        "open_roc_auc": epoch_metrics["open_metrics"]["roc_auc_open"],
                    },
                    step=epoch
                )


            eval_metrics = ml.evaluate_on_closed_and_open_testsets(
                open_loader, test_loader, model
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
                    "test_acc": eval_metrics["closed"]["acc_closed"],
                    "closed_roc_auc": eval_metrics["closed"]["roc_auc_closed"],
                    "closed_recall": eval_metrics["closed"]["recall_closed"],
                    "closed_precision": eval_metrics["closed"]["precision_closed"],
                    "closed_f1": eval_metrics["closed"]["f1_closed"],

                }
            )


            metadata = {
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

            fig_roc, _ = vis.plot_roc_open_and_closed_testsets(
                eval_metrics, metadata=metadata)
            mlflow.log_figure(fig_roc, "fig_roc.png")

            mlflow.pytorch.log_model(model, "pytorch_model")
            
            return crossval_optimizaion_metric 


    with mlflow.start_run(
        experiment_id=experiment_id, 
        run_name=run_name, 
        description=f"{ag_pos} vs {ag_neg}",
        ):

        study = optuna.create_study(direction="maximize")
        study.optimize(crossval_objective, n_trials=n_trials)

        # Train model with best parameters
        run_main_06(
            epochs=epochs_train, 
            learning_rate=study.best_params["learning_rate"],
            ag_pos=ag_pos, 
            ag_neg=ag_neg,
            optimizer_type="Adam",
            momentum=study.best_params["momentum"],
            weight_decay=study.best_params["weight_decay"],
            batch_size=study.best_params["batch_size"],
            save_model=True,
        )


def construct_loaders_06b(farmhash_mod_10_val_mask, ag_pos, ag_neg, train_batch_size=64):
    processed_dfs: dict = utils.load_processed_dataframes()
    _train_val_loader, test_loader, open_loader = construct_loaders_06(
        processed_dfs,
        ag_pos,
        ag_neg,
        batch_size=64  # doesn't matter here
        )
    
    df_train_val = processed_dfs["train_val"]
    df_train_val = df_train_val.loc[df_train_val["Antigen"].isin([ag_pos, ag_neg])].copy()
    
    df_train_val["Slide_farmhash_mod_10"] = list(map(
        preprocessing.farmhash_mod_10,
        df_train_val["Slide"]
    ))
    val_mask = df_train_val["Slide_farmhash_mod_10"] == farmhash_mod_10_val_mask
    df_train = df_train_val.loc[~val_mask].copy()
    df_val = df_train_val.loc[val_mask].copy()
    (   
        _,
        _,
        train_loader,
        val_loader,
        ) = preprocessing.preprocess_data_for_pytorch_binary(
            df_train_val=df_train,
            df_test_closed=df_val,
            ag_pos=[ag_pos],
            scale_onehot=True,
            batch_size=train_batch_size,
            df_test_open=None,
    )
    
    return test_loader, open_loader, train_loader, val_loader


if __name__ == "__main__":

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    experiment = mlflow.set_experiment(experiment_id=experiment_id)

    ag_pairs = []
    for (ag_pos, ag_neg) in combinations(config.ANTIGENS_CLOSEDSET, 2):
        ag_pairs.append((ag_pos, ag_neg))

    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(run_main_06b, ag_pairs)
