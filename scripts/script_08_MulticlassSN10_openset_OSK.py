import logging
import multiprocessing
from typing import List
import numpy as np
import torch.nn as nn

import mlflow

import NegativeClassOptimization.config as config
import NegativeClassOptimization.utils as utils
import NegativeClassOptimization.datasets as datasets
import NegativeClassOptimization.ml as ml
import NegativeClassOptimization.pipelines as pipelines


TEST = False

experiment_id = 6
run_name = "test"
input_data_dir = config.DATA_SLACK_1_PROCESSED_DIR
sample_data_source = None
sample_train_val = 70000
sample_test = None
batch_size = 64
model = "SN10_MULTICLASS"
hidden_dim = None
epochs = 20
learning_rate = 0.01
num_processes = 20


def multiprocessing_wrapper_script_08(
    ags: List[str],
    experiment_id,
    run_name,
    input_data_dir,
    sample_data_source,
    sample_train_val,
    sample_test,
    batch_size,
    model,
    hidden_dim,
    epochs,
    learning_rate,
    ):

    assert model in {"SN10_MULTICLASS", "SNN_MULTICLASS"}

    # run_name issue, next version should work
    #  https://github.com/mlflow/mlflow/issues/7217
    #  workaround is to use tags to setup run_name
    with mlflow.start_run(
        experiment_id=experiment_id, 
        run_name=run_name, 
        description=" vs ".join(ags),
        tags={"mlflow.runName": run_name},
        ):

        logger = logging.getLogger()
        logger.info(f"{run_name=}")
        logger.info("Start multiprocessing_wrapper_script_08")

        mlflow.log_params({
            "input_data_dir": str(input_data_dir),
            "test": str(TEST),
            "model": model,
            "hidden_dim": hidden_dim,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "optimizer_type": "Adam",
            "momentum": 0.9,
            "weight_decay": 0,
            "batch_size": batch_size,
            "ags": "__".join(ags),
            "k": len(ags),
            "sample": None,
            "sample_train": sample_train_val,
            "sample_test": sample_test,
        })

        pipeline = pipelines.MulticlassPipeline(
            log_mlflow=True,
            save_model_mlflow=True
        )

        pipeline.step_1_process_data(
            input_data_dir=input_data_dir,
            ags=ags,
            batch_size=batch_size,
            sample_data_source=sample_data_source,
            sample_train_val=sample_train_val,
            sample_test=sample_test,
        )

        if model == "SN10_MULTICLASS":
            model = ml.MulticlassSN10(num_classes=len(ags))
            assert hidden_dim is None
        elif model == "SNN_MULTICLASS":
            model = ml.MulticlassSNN(hidden_dim=40, num_classes=len(ags))
            assert type(hidden_dim) == int
        loss_fn = nn.CrossEntropyLoss()
        optimizer = ml.construct_optimizer(
            optimizer_type="Adam",
            learning_rate=learning_rate,
            momentum=0.9,
            weight_decay=0,
            model=model,
        )
        pipeline.step_2_train_model(
            model,
            loss_fn,
            optimizer,
            epochs=epochs,
        )

        pipeline.step_3_evaluate_model()
        pipeline.step_4_visualise()


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
        pool.starmap(
            multiprocessing_wrapper_script_08, 
            [
                (
                    ag_list, 
                    experiment_id,
                    run_name,
                    input_data_dir,
                    sample_data_source,
                    sample_train_val,
                    batch_size,
                    model,
                    hidden_dim,
                    epochs,
                    learning_rate,
                )
                for ag_list in ags
            ],
        )
