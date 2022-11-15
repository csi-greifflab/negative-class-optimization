import logging
import multiprocessing
from typing import List
import mlflow
import numpy as np

from NegativeClassOptimization import config, utils, datasets
from script_08_MulticlassSN10_openset_OSK import multiprocessing_wrapper_script_08


TEST = True

experiment_id = 7
run_name = "dev-0.1.2"
input_data_dir = config.DATA_ABSOLUT_PROCESSED_MULTICLASS_DIR
sample_data_source = None
sample_train_val = 500
sample_test = 500
batch_size = 64
model = "SN10_MULTICLASS"
hidden_dim = None
epochs = 3
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

    else:
        num_closed = config.NUM_CLOSED_ANTIGENS_ABSOLUT_DATASET3
        ags_c = antigens_shuffled[:num_closed]
        # ags_o = antigens_shuffled[num_closed:]

        NUM_CHAINS_PER_ITERATION = 3
        AG_SET_SIZES = [10, 20, 40, 80, 100]
        AG_SEED_SET_BANLIST = []
        while True:
            try:
                ags: List[str] = []
                for _ in range(NUM_CHAINS_PER_ITERATION):
                    chain = utils.generate_ag_set_chain(ags_c, AG_SET_SIZES, AG_SEED_SET_BANLIST)
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

            except KeyboardInterrupt as error:
                logging.info("KeyboardInterrupt")
                break
