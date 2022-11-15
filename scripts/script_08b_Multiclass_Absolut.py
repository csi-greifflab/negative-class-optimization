import logging
import multiprocessing
import mlflow
import numpy as np

from NegativeClassOptimization import config, utils, datasets
from script_08_MulticlassSN10_openset_OSK import multiprocessing_wrapper_script_08


TEST = False

experiment_id = 7
run_name = "dev-0.1.2"
input_data_dir = config.DATA_ABSOLUT_PROCESSED_MULTICLASS_DIR
sample_data_source = None
sample_train_val = 50000
sample_test = 50000
batch_size = 64
model = "SN10_MULTICLASS"
hidden_dim = None
epochs = 20
learning_rate = 0.01
num_processes = 20


if __name__ == "__main__":

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

    antigens = datasets.AbsolutDataset3.get_antigens()
    antigens_shuffled = utils.shuffle_antigens(antigens)

    if TEST:
        ags = [
            antigens_shuffled[:5],
            antigens_shuffled[:10],
        ]
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
                    sample_test,
                    batch_size,
                    model,
                    hidden_dim,
                    epochs,
                    learning_rate,
                )
                for ag_list in ags
            ],
        )
