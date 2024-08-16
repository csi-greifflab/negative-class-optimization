"""
Script to compute attributions, neuron activations and other features for tasks.
Based on:
- script_14_frozen_transfer_performance.py
- notebook 07d_Activations.ipynb
"""

import json
import logging
import multiprocessing
import shutil
import time
from itertools import permutations
from pathlib import Path
from typing import List

import torch
import torch.optim as optim
from docopt import docopt

from NegativeClassOptimization import config, datasets, ml, pipelines
from NegativeClassOptimization.ml import load_model_from_state_dict

# from script_14b_epi_frozen_transfer_performance import \
    # get_test_dataset as get_test_dataset_epi  # now done through pipelines


docopt_doc = """Compute attributions.

Usage:
    script_15_compute_attributions.py <analysis_name> <input_dir>
    script_15_compute_attributions.py <analysis_name> <input_dir> --experimental
    script_15_compute_attributions.py <analysis_name> <input_dir> --epitopes_only <epitopes_test_set>

Options:
    -h --help   Show help.
"""

arguments = docopt(docopt_doc, version="NCO")

TEST = False
TEST_TASKLIST = False

DIR_EXISTS_HANDLE = "overwrite"  # "raise" or "skip" or "overwrite" or "ignore"

# One of the below make true, or all false

if arguments["--experimental"]:
    EXPERIMENTAL_DATA_ONLY = True
else:
    EXPERIMENTAL_DATA_ONLY = False

if arguments["--epitopes_only"]:
    EPITOPES_ONLY = True
else:
    EPITOPES_ONLY = False
# EPITOPES_ONLY = True
# EPITOPES_TEST_SET = "PositiveSet_Epitope"  # see script_14b for meaning
EPITOPES_TEST_SET = arguments["<epitopes_test_set>"]
SIMILAR_ANTIGENS_ONLY = False
DISSIMILAR_ANTIGENS_ONLY = False

analysis_name = arguments["<analysis_name>"]
# analysis_name = "v2.0-3-epipos"  # v2.0-2 most of the time, for the epitopes v2.0-3-epi

data_dir = Path(arguments["<input_dir>"])
# data_dir = Path("data/Frozen_MiniAbsolut_ML")  # "data/Frozen_MiniAbsolut_ML" "data/Frozen_MiniAbsolut_ML_shuffled/"

task_types = [
    datasets.ClassificationTaskType.HIGH_VS_95LOW,
    datasets.ClassificationTaskType.HIGH_VS_LOOSER,
    datasets.ClassificationTaskType.ONE_VS_ONE,
    datasets.ClassificationTaskType.ONE_VS_NINE,
]

task_split_seed_filter = ((42,), (0,))  # split, seed. Set to None for all.

# Define attributor templates, which are used to generate attributors for each task.
attributor_templates = [
    {
        "name": f"DeepLIFT_LOCAL_{analysis_name}",
        "type": "deep_lift",
        "baseline_type": "shuffle",
        "num_shuffles": 10,
        "compute_on": "logits",
        "multiply_by_inputs": True,
    },
    # {
    #     "name": f"DeepLIFT_GLOBAL_R10_{analysis_name}",
    #     "type": "deep_lift",
    #     "baseline_type": "shuffle",
    #     "num_shuffles": 10,
    #     "compute_on": "logits",
    #     "multiply_by_inputs": False,
    # },
]
loader = datasets.FrozenMiniAbsolutMLLoader(data_dir=data_dir)
num_processes = 20

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(process)d %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("data/logs/15_compute_attributions.log"),
        logging.StreamHandler(),
    ],
)


def task_generator(task_types=task_types, loader=loader):
    """
    Generate tasks for which to compute attributions.
    """
    seed_split_ids = datasets.FrozenMiniAbsolutMLLoader.generate_seed_split_ids()
    for ag_1, ag_2 in permutations(config.ANTIGENS, r=2):
        for seed_id, split_id in seed_split_ids:
            for task_type in task_types:
                if task_type == datasets.ClassificationTaskType.ONE_VS_ONE:
                    task = datasets.ClassificationTask(
                        task_type=task_type,
                        ag_pos=ag_1,
                        ag_neg=ag_2,
                        seed_id=seed_id,
                        split_id=split_id,
                    )
                else:
                    task = datasets.ClassificationTask(
                        task_type=task_type,
                        ag_pos=ag_1,
                        ag_neg="auto",
                        seed_id=seed_id,
                        split_id=split_id,
                    )
                yield task


def task_generator_for_experimental():
    seed_split_ids = datasets.FrozenMiniAbsolutMLLoader.generate_seed_split_ids()
    for ag in ["HR2P", "HR2B"]:
        for seed_id, split_id in seed_split_ids:
            for task_type in [
                datasets.ClassificationTaskType.HIGH_VS_95LOW,
                datasets.ClassificationTaskType.HIGH_VS_LOOSER,
            ]:
                task = datasets.ClassificationTask(
                    task_type=task_type,
                    ag_pos=ag,
                    ag_neg="auto",
                    seed_id=seed_id,
                    split_id=split_id,
                )
                yield task


def task_generator_for_experimental_randomized():
    seed_split_ids = datasets.FrozenMiniAbsolutMLLoader.generate_seed_split_ids()
    for ag_1, ag_2 in [("HR2P", "HR2PSR"), ("HR2P", "HR2PIR")]:
        for seed_id, split_id in seed_split_ids:
            for task_type in [
                datasets.ClassificationTaskType.ONE_VS_ONE,
                ]:
                task = datasets.ClassificationTask(
                    task_type=task_type,
                    ag_pos=ag_1,
                    ag_neg=ag_2,
                    seed_id=seed_id,
                    split_id=split_id,
                )
                yield task


def task_generator_for_epitopes(task_types=task_types):
    """
    Generate tasks for which to compute attributions.
    """
    seed_split_ids = datasets.FrozenMiniAbsolutMLLoader.generate_seed_split_ids()
    
    for task_type in task_types:
        for ag_1 in config.ANTIGEN_EPITOPES:
            for ag_2 in config.ANTIGENS + config.ANTIGEN_EPITOPES:

                if ag_1.split("E1")[0] == ag_2:
                    continue

                if ag_1 == ag_2:
                    continue

                for seed_id, split_id in seed_split_ids:
                    
                    if not (seed_id == 0 and split_id == 42):
                        continue
                
                    if task_type == datasets.ClassificationTaskType.ONE_VS_ONE:
                        task = datasets.ClassificationTask(
                            task_type=task_type,
                            ag_pos=ag_1,
                            ag_neg=ag_2,
                            seed_id=seed_id,
                            split_id=split_id,
                        )
                    else:
                        task = datasets.ClassificationTask(
                            task_type=task_type,
                            ag_pos=ag_1,
                            ag_neg="auto",
                            seed_id=seed_id,
                            split_id=split_id,
                        )
                    yield task


def task_generator_for_similar_or_dissimilar(similar=True, task_types=task_types, loader=loader):
    """
    Generate tasks for which to compute attributions.
    """
    seed_split_ids = datasets.FrozenMiniAbsolutMLLoader.generate_seed_split_ids()
    
    if similar:
        ag_list = [ag + "SIM" for ag in config.ANTIGENS]
    else:
        ag_list = [ag + "DIF" for ag in config.ANTIGENS]

    for task_type in task_types:
        for ag_1 in ag_list:
            for ag_2 in ag_list:

                if ag_1.split("E1")[0] == ag_2:
                    continue

                for seed_id, split_id in seed_split_ids:
                    
                    if not (seed_id == 0 and split_id == 42):
                        continue
                
                    if task_type == datasets.ClassificationTaskType.ONE_VS_ONE:
                        task = datasets.ClassificationTask(
                            task_type=task_type,
                            ag_pos=ag_1,
                            ag_neg=ag_2,
                            seed_id=seed_id,
                            split_id=split_id,
                        )
                    else:
                        task = datasets.ClassificationTask(
                            task_type=task_type,
                            ag_pos=ag_1,
                            ag_neg="auto",
                            seed_id=seed_id,
                            split_id=split_id,
                        )
                    yield task



def compute_attributions(task, save=True):
    # Get logger per process
    logger = logging.getLogger()
    logger.info(f"Task: {repr(task)}")

    # Load task.
    loader.load(task)

    # For experimental data, the state dict is loaded,
    # not the model. We adjust it here.
    if task.model is None:
        assert task.state_dict is not None
        model = load_model_from_state_dict(task.state_dict)
        task.model = model

    model = task.model  # type: ignore
    
    if EPITOPES_ONLY:
        test_dataset = pipelines.get_test_dataset_for_epitope_analysis(task, test_set=EPITOPES_TEST_SET)
    else:
        test_dataset = task.test_dataset  # type: ignore
    
    logger.info(f"Loaded task.")

    # import pdb
    # pdb.set_trace()

    if save:
        # Build output dir.
        # Check analysis_name is not already used.
        task_basepath = loader.infer_task_basepath(task)
        output_dir = task_basepath / "attributions"
        if not output_dir.exists():
            logger.info(f"Creating output dir {output_dir}.")
            output_dir.mkdir()
        elif output_dir.exists():
            logger.info(f"Output dir already exists: {output_dir}")
        
        output_dir = output_dir / analysis_name
        if output_dir.exists():
            if DIR_EXISTS_HANDLE == "raise":
                logger.error(
                    f"Output dir {output_dir} already exists (analysis_name conflict?)"
                )
                raise ValueError(
                    f"Output dir {output_dir} already exists (analysis_name conflict?)."
                )
            elif DIR_EXISTS_HANDLE == "skip":
                logger.info(f"Output dir {output_dir} already exists. Skipping.")
                return
            elif DIR_EXISTS_HANDLE == "overwrite":
                # Remove and recreate the dir.
                # shutil.rmtree(output_dir)
                output_dir.mkdir(exist_ok=True)
            elif DIR_EXISTS_HANDLE == "ignore":
                pass
        else:
            logger.info(f"Creating output dir {output_dir}.")
            Path(output_dir).mkdir()

    if type(model) == torch.optim.swa_utils.AveragedModel:
        # Unwrap the SWA model. We need a module class,
        # that has updated weights, but still has other
        # module funcs, such as forward_logits.
        # Note: swa_model.module has same weights as swa_model.state_dict().
        model_for_attr = model.module
    else:
        model_for_attr = model

    attributors = [
        ml.Attributor(model=model_for_attr, **attributor_template)  # type: ignore
        for attributor_template in attributor_templates
    ]

    # Compute attributions.
    t_start = time.time()
    logger.info(f"Computing attributions for task {task}.")
    df = ml.compute_and_collect_model_predictions_and_attributions(
        test_dataset if not TEST else test_dataset.iloc[:10],
        model,
        attributors,
        N=None,  # type: ignore
    )
    t_end = time.time()
    logger.info(f"Execution time: {t_end - t_start:.2f} seconds.")

    # Drop the "enc" column, which is a list of encoded sequences.
    df = df.drop(["enc"], axis=1)

    # Make json-serializable.
    records: List[dict] = df.to_dict(orient="records")
    records_serializable: List[dict] = []
    for record in records:
        record_serializable = record.copy()
        # Remove attributions dict key, which are not json-serializable.
        record_serializable.pop("attributions")
        # Add attributions dict keys as top-level keys and json-serialize the values.
        record_serializable.update(
            {
                attributor_name: attributions["attributions"].tolist()
                for attributor_name, attributions in record["attributions"].items()
            }
        )
        record_serializable["exec_time"] = t_end - t_start

        records_serializable.append(record_serializable)

    # Compute neurons' z (before ReLU).
    # Note: this is not the same as the neuron's activation, but
    # it can be used to compute the neuron's activation easily.
    # See notebook 07d_Activations.ipynb.
    z_records = []
    slides: List[str] = test_dataset["Slide"].tolist()
    for slide in slides:
        z: torch.Tensor = ml.get_activations_on_slide(
            slide, model_for_attr, return_z=True  # type: ignore
        )
        z = z.tolist()  # type: ignore
        record = {
            "slide": slide,
            "z": z,
        }
        z_records.append(record)

    if save:
        logger.info(f"Saving results to {output_dir}.")

        # Save the results.
        # The output_dir should have been generated previously, same function.
        assert output_dir.exists(), f"Output dir {output_dir} does not exist."

        output_file = output_dir / f"attribution_records.json"
        with open(output_file, "w") as f:
            json.dump(records_serializable, f)

        # Save the attributor templates.
        output_file = output_dir / f"attributor_templates.json"
        with open(output_file, "w") as f:
            json.dump(attributor_templates, f)

        # Save the z records.
        output_file = output_dir / ".." / f"z_records.json"
        with open(output_file, "w") as f:
            json.dump(z_records, f)

        # Save model linear_1, linear_2 and respective biases weights.
        output_file = output_dir / f"model_weights.json"
        with open(output_file, "w") as f:
            json.dump(
                {
                    "linear_1.weight": model_for_attr.linear_1.weight.tolist(),  # type: ignore
                    "linear_1.bias": model_for_attr.linear_1.bias.tolist(),  # type: ignore
                    "linear_2.weight": model_for_attr.linear_2.weight.tolist(),  # type: ignore
                    "linear_2.bias": model_for_attr.linear_2.bias.tolist(),  # type: ignore
                },
                f,
            )

    if TEST:
        print(records_serializable)


if __name__ == "__main__":
    if EXPERIMENTAL_DATA_ONLY:
        # Generate only the tasks from the experimental data
        # task_data = list(task_generator_for_experimental())
        task_data = list(task_generator_for_experimental_randomized())
    elif EPITOPES_ONLY:
        task_data = list(task_generator_for_epitopes())
    elif SIMILAR_ANTIGENS_ONLY:
        task_data = list(task_generator_for_similar_or_dissimilar(similar=True))
    elif DISSIMILAR_ANTIGENS_ONLY:
        task_data = list(task_generator_for_similar_or_dissimilar(similar=False))
    else:
        # Generate all the tasks from Absolut
        task_data = list(task_generator())

    if task_split_seed_filter is not None:
        task_data = [
            task
            for task in task_data
            if task.split_id in task_split_seed_filter[0]
            and task.seed_id in task_split_seed_filter[1]
        ]
    print(len(task_data))
    if TEST:
        task = task_data[2]
        compute_attributions(task, save=False)
    elif TEST_TASKLIST:
        for task in task_data:
            print(task)
    else:
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.starmap(
                compute_attributions,
                [(task,) for task in task_data],
            )
