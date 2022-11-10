import logging
import multiprocessing
import mlflow
import numpy as np

from NegativeClassOptimization import config, utils, datasets
from script_08_MulticlassSN10_openset_OSK import multiprocessing_wrapper_script_08


TEST = True

experiment_id = 7
run_name = "test"
input_data_dir = config.DATA_ABSOLUT_PROCESSED_MULTICLASS_DIR
sample_data_source = None
sample_train_val = 500  # 70000
batch_size = 64
epochs = 3  # 20
learning_rate = 0.01
num_processes = 20


if __name__ == "__main__":

    raise NotImplementedError()

    logging.basicConfig(
        format="%(asctime)s %(process)d %(funcName)s %(levelname)s %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(
                filename="data/logs/08b.log",
                mode="a",
            ),
            logging.StreamHandler()
        ]
    )
    utils.nco_seed()
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    experiment = mlflow.set_experiment(experiment_id=experiment_id)


    if TEST:
        ags = None
        raise NotImplementedError()
    else:
        raise NotImplementedError()
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
                    epochs,
                    learning_rate,
                )
                for ag_list in ags
            ],
        )
