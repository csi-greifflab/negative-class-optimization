"""
This script runs workflow 05 for all the antigen pairs.
"""

from itertools import combinations
from pathlib import Path
from typing import List
import yaml
import tqdm
import multiprocessing
import mlflow

import NegativeClassOptimization.config as config
from script_05_SN10_closedset_binary_interpretability import process_data_and_train_model, run_attribution_workflow


# Get parameters
with open(config.PARAMS_PATH, "r") as fh:
    params = yaml.safe_load(fh)

experiment_id = params["05b_all_SN10_closedset_binary_interpretability"]["experiment_id"]
run_name = params["05b_all_SN10_closedset_binary_interpretability"]["run_name"]
num_processes = params["05b_all_SN10_closedset_binary_interpretability"]["num_processes"]
learning_rate = params["05b_all_SN10_closedset_binary_interpretability"]["learning_rate"]
epochs = params["05b_all_SN10_closedset_binary_interpretability"]["epochs"]

out_dir = Path("data/SN10_closedset_binary_interpretability_all")
data_path = config.DATA_SLACK_1_GLOBAL


def run_main(
        ag_pair,
        experiment_id=experiment_id,
        run_name=run_name,
        learning_rate=learning_rate,
        epochs=epochs,
        out_dir=out_dir,
        data_path=data_path,
    ):

    ag_pos, ag_neg = ag_pair

    out_dir_i = out_dir / f"{ag_pos}_vs_{ag_neg}"
    out_dir_i.mkdir(exist_ok=True)
    
    with mlflow.start_run(
        experiment_id=experiment_id, 
        run_name=run_name, 
        description=f"{ag_pos} vs {ag_neg}"
        ):

        train_data, test_data, model, online_metrics = process_data_and_train_model(
            ag_pos,
            ag_neg,
            learning_rate,
            epochs,
            data_path
        )

        attributions_savepath = out_dir_i / \
            f"attributions_{ag_pos}_vs_{ag_neg}.tsv"
        agg_attributions_savepath = out_dir_i / \
            f"aggregated_attributions_{ag_pos}_vs_{ag_neg}.tsv"
        attributions_ag_pos_fig_path = out_dir_i / \
            f"aggregated_attributions_{ag_pos}.png"
        attributions_ag_neg_fig_path = out_dir_i / \
            f"aggregated_attributions_{ag_neg}.png"
        run_attribution_workflow(
            ag_pos,
            ag_neg,
            attributions_savepath,
            agg_attributions_savepath,
            attributions_ag_pos_fig_path,
            attributions_ag_neg_fig_path,
            test_data,
            model
        )

        mlflow.log_params({
            "ag_pos": ag_pos,
            "ag_neg": ag_neg,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "train_data_nrows": train_data.df.shape[0],
            "train_data_pos_ratio": train_data.df.y.sum() / train_data.df.shape[0]
        })

        for i, epoch_record in enumerate(online_metrics):
            mlflow.log_metrics({
                "train_loss": epoch_record["train_losses"][-1],
                "test_loss": epoch_record["test_metrics"]["test_loss"],
                "test_accuracy": epoch_record["test_metrics"]["accuracy"],
            }, step=i)
        
        mlflow.log_artifacts(out_dir_i)



if __name__ == "__main__":

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    experiment = mlflow.set_experiment(experiment_id=experiment_id)

    out_dir.mkdir(exist_ok=True)

    ag_pairs = []
    for (ag_pos, ag_neg) in combinations(config.ANTIGENS_CLOSEDSET, 2):
        ag_pairs.append((ag_pos, ag_neg))

    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(run_main, ag_pairs)
