"""
Script to compute attributions, neuron activations and other features for tasks.
Based on:
- script_14_frozen_transfer_performance.py
- 
- notebook 07d_Activations.ipynb
"""

import json
import logging
import multiprocessing
import time
from itertools import permutations
from pathlib import Path
from typing import List

import pandas as pd
import torch

from NegativeClassOptimization import config, datasets, ml, preprocessing

DIR_EXISTS_HANDLE = "skip"  # "raise" or "skip"
TEST = False

analysis_name = "v2.0-2_mutants"
data_dir = Path("data/Frozen_MiniAbsolut_ML/")

# Define attributor templates, which are used to generate attributors for each task.
attributor_templates = [
    {
        "name": f"DeepLIFT_GLOBAL_R10_{analysis_name}",
        "type": "deep_lift",
        "baseline_type": "shuffle",
        "num_shuffles": 10,
        "compute_on": "logits",
        "multiply_by_inputs": False,
    },
]
loader = datasets.FrozenMiniAbsolutMLLoader(data_dir=data_dir)
num_processes = 20

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(process)d %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("data/logs/15b_compute_attributions_mutated.log"),
        logging.StreamHandler(),
    ],
)


def task_generator():
    """
    Generate tasks for which to compute attributions.
    """
    for path in Path("data/mutants/mutant_igs/").glob("*.csv"):
        if "shuffled" in str(path):
            continue
        str_describing_task = path.name.split("_mut_igs")[0]

        if "_HIGH_VS_95LOW" in str_describing_task:
            task_type = datasets.ClassificationTaskType.HIGH_VS_95LOW
            ag_pos = str_describing_task.split("_HIGH_VS_95LOW")[0]
            ag_neg = "auto"
        elif "_HIGH_VS_LOOSER" in str_describing_task:
            task_type = datasets.ClassificationTaskType.HIGH_VS_LOOSER
            ag_pos = str_describing_task.split("_HIGH_VS_LOOSER")[0]
            ag_neg = "auto"
        elif "_ONE_VS_NINE" in str_describing_task:
            task_type = datasets.ClassificationTaskType.ONE_VS_NINE
            ag_pos = str_describing_task.split("_ONE_VS_NINE")[0]
            ag_neg = "auto"
        elif "__vs__" in str_describing_task:
            # ONE_VS_ONE
            task_type = datasets.ClassificationTaskType.ONE_VS_ONE
            ag_pos = str_describing_task.split("__vs__")[0]
            ag_neg = str_describing_task.split("__vs__")[1]

        task = datasets.ClassificationTask(
            task_type=task_type,  # type:ignore
            ag_pos=ag_pos,  # type:ignore
            ag_neg=ag_neg,  # type:ignore
            seed_id=0,  # type: ignore
            split_id=42,  # type: ignore
        )

        task.mutated_path = path

        yield task


def compute_attributions(task, save=True):
    # Get logger per process
    logger = logging.getLogger()
    logger.info(f"Task: {repr(task)}")

    # Load mutant dataset.
    mutated_path = task.mutated_path
    df = pd.read_csv(mutated_path)

    # Load task's model.
    loader.load(task)
    model = task.model  # type: ignore
    logger.info(f"Loaded task.")

    if save:
        # Build output dir.
        # Check analysis_name is not already used.
        task_basepath = f"data/mutants/mutant_attributions_{analysis_name}/"

        if not Path(task_basepath).exists():
            Path(task_basepath).mkdir()
        # else:
        #     if DIR_EXISTS_HANDLE == "raise":
        #         logger.error(
        #             f"Output dir {task_basepath} already exists (analysis_name conflict?)"
        #         )
        #         raise ValueError(
        #             f"Output dir {task_basepath} already exists (analysis_name conflict?)."
        #         )
        #     elif DIR_EXISTS_HANDLE == "skip":
        #         logger.info(f"Output dir {task_basepath} already exists. Skipping.")
        #         return

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
    # Adapted for the computation of attributions for mutants.
    t_start = time.time()
    assert len(attributors) == 1, "Only one attributor supported for now."
    attributor = attributors[0]
    records = []
    for i, row in df.iterrows():
        slide = row["Slide"]
        enc = torch.tensor(preprocessing.onehot_encode(slide)).float().reshape((1, -1))
        attributions, baseline = attributor.attribute(enc, return_baseline=True)

        original_slide = row["original Slide"]
        original_enc = (
            torch.tensor(preprocessing.onehot_encode(original_slide))
            .float()
            .reshape((1, -1))
        )
        original_attributions, original_baseline = attributor.attribute(
            original_enc, return_baseline=True
        )

        record = {
            "slide": slide,
            "enc": enc.tolist(),
            "original_slide": original_slide,
            "original_enc": original_enc.tolist(),
            "attributions": attributions.tolist(),
            "baseline": baseline,
            "original_attributions": original_attributions.tolist(),
            "original_baseline": original_baseline,
        }
        records.append(record)

    t_end = time.time()
    logger.info(f"Execution time: {t_end - t_start:.2f} seconds.")

    # Save records
    if save:
        logger.info(f"Saving results to {task_basepath}.")

        # Save the results based on mutated path name.
        res_path = (
            task_basepath + mutated_path.name.split(".csv")[0] + "_attributions.json"
        )
        with open(res_path, "w") as f:
            json.dump(records, f)

    return records

    ### Dropped this part, because it is not needed for the analysis.
    # # Compute neurons' z (before ReLU).
    # # Note: this is not the same as the neuron's activation, but
    # # it can be used to compute the neuron's activation easily.
    # # See notebook 07d_Activations.ipynb.
    # z_records = []
    # slides: List[str] = test_dataset["Slide"].tolist()
    # for slide in slides:
    #     z: torch.Tensor = ml.get_activations_on_slide(
    #         slide, model_for_attr, return_z=True  # type: ignore
    #     )
    #     z = z.tolist()  # type: ignore
    #     record = {
    #         "slide": slide,
    #         "z": z,
    #     }
    #     z_records.append(record)

    # if save:
    #     logger.info(f"Saving results to {output_dir}.")

    #     # Save the results.
    #     output_dir.mkdir()
    #     output_file = output_dir / f"attribution_records.json"
    #     with open(output_file, "w") as f:
    #         json.dump(records_serializable, f)

    #     # Save the attributor templates.
    #     output_file = output_dir / f"attributor_templates.json"
    #     with open(output_file, "w") as f:
    #         json.dump(attributor_templates, f)

    #     # Save the z records.
    #     output_file = output_dir / ".." / f"z_records.json"
    #     with open(output_file, "w") as f:
    #         json.dump(z_records, f)

    #     # Save model linear_1, linear_2 and respective biases weights.
    #     output_file = output_dir / f"model_weights.json"
    #     with open(output_file, "w") as f:
    #         json.dump(
    #             {
    #                 "linear_1.weight": model_for_attr.linear_1.weight.tolist(),  # type: ignore
    #                 "linear_1.bias": model_for_attr.linear_1.bias.tolist(),  # type: ignore
    #                 "linear_2.weight": model_for_attr.linear_2.weight.tolist(),  # type: ignore
    #                 "linear_2.bias": model_for_attr.linear_2.bias.tolist(),  # type: ignore
    #             },
    #             f,
    #         )


if __name__ == "__main__":
    task_data = list(task_generator())
    print(len(task_data))
    if TEST:
        task = task_data[0]
        records = compute_attributions(task, save=False)
    else:
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.starmap(
                compute_attributions,
                [(task,) for task in task_data],
            )
