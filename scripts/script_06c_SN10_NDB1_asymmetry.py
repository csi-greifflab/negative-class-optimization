"""

"""


import multiprocessing
from NegativeClassOptimization import config, datasets, ml, pipelines, preprocessing, utils, visualisations
import torch
import mlflow
import numpy as np
from script_06_SN10_openset_NDB1 import resolve_ag_type


TEST = False

NUM_RANDOM_PAIRS = 10
NUM_TRAIN_TEST_SPLITS = 5
NUM_REPLICATES_PER_SPLIT = 5
NUM_TEST_KEYS = 3

experiment_id = 8
run_name = "dev-v0.1.2-3"
num_samples_closed = 30000
num_samples_open = 10000
batch_size = 64
epochs = 30
learning_rate = 0.001
optimizer_type = "Adam"
momentum = 0.9
weight_decay = 0
save_model = True
num_processes = 20


def num_runs():
    return NUM_RANDOM_PAIRS * NUM_TRAIN_TEST_SPLITS * NUM_REPLICATES_PER_SPLIT * 2


# ags_closed 
# getting data and train/test split.


def multiprocessing_wrapper_script_06c(
    params,
    df_global,
    ags_open,
    num_samples_closed,
    df_test_open_exclusive_shared,
    batch_size,
    epochs,
    learning_rate,
    optimizer_type,
    momentum,
    weight_decay,
    save_model,
    experiment_id,
    run_name,
    ):
    
    with mlflow.start_run(
        experiment_id=experiment_id, 
        run_name=run_name, 
        description=f"{params['ag_pair']}",
        tags={"mlflow.runName": run_name},
    ):

        ag_pos, ag_neg = params["ag_pair"]
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
            "num_samples_closed": num_samples_closed,
            "num_samples_open": num_samples_open,
            "test_hash_key_0": params["test_hash_key_0"],
            "test_hash_keys": params["test_hash_keys"],
            "rep": params["replicate"],
        })

        if df_test_open_exclusive_shared is None:
            df_train_val, df_test_closed_exclusive, df_test_open_exclusive = preprocessing.openset_datasplit_from_global_stable(
                df_global,
                ags_open,
                closedset_antigens=list(params["ag_pair"]),
                sample_closed=num_samples_closed,
                farmhash_mod_10_test_mask=params["test_hash_keys"],
            )
        else:
            df_test_open_exclusive = df_test_open_exclusive_shared
            try:
                df_train_val, df_test_closed_exclusive, _ = preprocessing.openset_datasplit_from_global_stable(
                    df_global,
                    openset_antigens=None,
                    closedset_antigens=list(params["ag_pair"]),
                    sample_closed=num_samples_closed,
                    farmhash_mod_10_test_mask=params["test_hash_keys"],
                )
            except OverflowError as e:
                print(f"{df_global.shape=} | {num_samples_closed=} | {params['ag_pair']=}")
                print(e)
                raise


        (   
            _,
            _,
            _,
            train_loader,
            test_loader,
            open_loader,
            ) = preprocessing.preprocess_data_for_pytorch_binary(
                df_train_val=df_train_val,
                df_test_closed=df_test_closed_exclusive,
                ag_pos=ag_pos,
                scale_onehot=True,
                batch_size=batch_size,
                df_test_open=df_test_open_exclusive,
                sample_train=None,
            )
        mlflow.log_params({
            "N_train": len(train_loader.dataset),
            "N_closed": len(test_loader.dataset),
            "N_open": len(open_loader.dataset),
        })

        ### 
        torch.manual_seed(params["replicate"])
        model = ml.SN10().to("cpu")
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
        
        utils.mlflow_log_params_online_metrics(online_metrics)

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
            }
        )

        metadata={
            "ag_pos": ag_pos,
            "ag_neg": ag_neg,  # ok to be a list
            "N_train": len(train_loader.dataset),
            "N_closed": len(test_loader.dataset),
            "N_open": len(open_loader.dataset),
        }
        
        fig_abs_logit_distr, ax_abs_logit_distr = visualisations.plot_abs_logit_distr(
                eval_metrics["open"], 
                metadata=metadata,
            )
        mlflow.log_figure(fig_abs_logit_distr, "fig_abs_logit_distr.png")
            
        fig_roc, _ = visualisations.plot_roc_open_and_closed_testsets(eval_metrics, metadata=metadata)
        mlflow.log_figure(fig_roc, "fig_roc.png")

        fig_pr, _ = visualisations.plot_pr_open_and_closed_testsets(eval_metrics, metadata=metadata)
        mlflow.log_figure(fig_pr, "fig_pr.png")

        if save_model:
            mlflow.pytorch.log_model(model, "pytorch_model")


if __name__ == "__main__":

    utils.nco_seed()
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    experiment = mlflow.set_experiment(experiment_id=experiment_id)

    print(f"Starting. Num runs: {num_runs()}")
    dataset = datasets.AbsolutDataset3()
    df_global = preprocessing.convert_wide_to_global(dataset.df_wide)  # global format of all binding data
    ags_c, ags_o = datasets.AbsolutDataset3.split_open_closed(dataset.antigens, num_open=40, seed=config.SEED)
    print(f"Preprocessed df_global.")

    builder = pipelines.NDB1_Assymetry_from_Absolut_Builder(ags_c, dataset)
    builder.step_01_select_random_pairs(NUM_RANDOM_PAIRS)
    builder.step_02_convert_to_global_format()
    print(f"Builder finished.")

    if TEST:
        params = {
            "ag_pair": tuple(ags_c[:2]),
            "test_hash_key": 7,
            "replicate": 2,
        }

        multiprocessing_wrapper_script_06c(
            params,
            df_global,
            ags_open=ags_o,
            num_samples_closed=num_samples_closed,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            optimizer_type=optimizer_type,
            momentum=momentum,
            weight_decay=weight_decay,
            save_model=save_model,
            experiment_id=experiment_id,
            run_name=run_name,
        )

    else:

        run_params = []
        for ag_pair in builder.ag_pairs:
            for test_hash_key_0 in range(NUM_TRAIN_TEST_SPLITS):
                for rep in range(NUM_REPLICATES_PER_SPLIT):
                    test_hash_keys = [test_hash_key_0 + i for i in range(NUM_TEST_KEYS)]
                    params = {
                        "ag_pair": ag_pair,
                        "test_hash_key_0": test_hash_key_0,
                        "test_hash_keys": test_hash_keys,
                        "replicate": rep,
                    }
                    run_params.append(params)
                    params = params.copy()
                    params["ag_pair"] = ag_pair[::-1]
                    run_params.append(params)

        # build a common df_test_open_exclusive
        builder.ag_pairs
        _, _, df_test_open_exclusive_shared = preprocessing.openset_datasplit_from_global_stable(
            df_global,
            ags_o,
            closedset_antigens=None,
            sample_closed=None,
        )
        df_test_open_exclusive_shared = df_test_open_exclusive_shared.sample(
            n=num_samples_open, 
            random_state=config.SEED,
            )
        print(f"Built df_test_open_exclusive_shared.")

        for i in range(0, len(run_params), num_processes):
            params_batch = run_params[i:i+num_processes]
            with multiprocessing.Pool(processes=num_processes) as pool:
                pool.starmap(
                    multiprocessing_wrapper_script_06c,
                    [
                        (
                            params,
                            df_global,
                            ags_o,
                            num_samples_closed,
                            df_test_open_exclusive_shared,
                            batch_size,
                            epochs,
                            learning_rate,
                            optimizer_type,
                            momentum,
                            weight_decay,
                            save_model,
                            experiment_id,
                            run_name,
                        )
                        for params in params_batch
                    ]
                )
