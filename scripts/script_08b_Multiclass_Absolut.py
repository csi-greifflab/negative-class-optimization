import logging
import multiprocessing
import time
from typing import List
import mlflow
import numpy as np

from NegativeClassOptimization import config, utils, datasets
from script_08_MulticlassSN10_openset_OSK import multiprocessing_wrapper_script_08


TEST = False

experiment_id = 9
run_name = "dev-0.1.2-1"
input_data_dir = config.DATA_ABSOLUT_PROCESSED_MULTILABEL_DIR
sample_data_source = None
sample_per_ag_train = 1000
sample_per_ag_test = None
batch_size = 64
model = "SNN_MULTILABEL"
hidden_dim = None
epochs = 30
learning_rate = 0.001
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
                        sample_per_ag_train,
                        sample_per_ag_test,
                        batch_size,
                        model,
                        hidden_dim,
                        epochs,
                        learning_rate,
                    )
                    for ag_list in ags
                ],
            )

    else:

        ags_closed = datasets.AbsolutDataset3.get_closed_antigens(input_data_dir)

        NUM_CHAINS_PER_ITERATION = 3
        AG_SET_SIZES = [3, 6, 12, 24, 48]
        AG_SEED_SET_BANLIST = []
        while True:
            try:
                ags: List[str] = []
                for _ in range(NUM_CHAINS_PER_ITERATION):
                    chain = utils.generate_ag_set_chain(ags_closed, AG_SET_SIZES, AG_SEED_SET_BANLIST)
                    AG_SEED_SET_BANLIST.append(chain[0])
                    ags += chain
                
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
                                sample_per_ag_train,
                                sample_per_ag_test,
                                batch_size,
                                model,
                                hidden_dim,
                                epochs,
                                learning_rate,
                            )
                            for ag_list in ags
                        ],
                    )

            except KeyboardInterrupt as error:
                logging.info("KeyboardInterrupt")
                break
