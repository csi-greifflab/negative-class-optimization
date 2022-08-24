"""
This script runs workflow 05 for all the antigen pairs.
"""

from itertools import combinations
from pathlib import Path
from typing import List
import yaml
import tqdm

import NegativeClassOptimization.config as config
from script_05_SN10_closedset_binary_interpretability import process_data_and_train_model, run_attribution_workflow


# Get parameters
with open(config.PARAMS_PATH, "r") as fh:
    params = yaml.safe_load(fh)

learning_rate = params["05_SN10_closedset_binary_interpretability"]["learning_rate"]
epochs = params["05_SN10_closedset_binary_interpretability"]["epochs"]

out_dir = Path("data/SN10_closedset_binary_interpretability_all")
data_path = config.DATA_SLACK_1_GLOBAL



if __name__ == "__main__":

    out_dir.mkdir(exist_ok=True)

    antigens = config.ANTIGENS
    for (ag_pos, ag_neg) in tqdm.tqdm(combinations(antigens, 2)):

        out_dir_i = out_dir / f"{ag_pos}_vs_{ag_neg}"
        out_dir_i.mkdir(exist_ok=True)

        attributions_savepath = out_dir_i / f"attributions_{ag_pos}_vs_{ag_neg}.tsv"
        agg_attributions_savepath = out_dir_i / \
            f"aggregated_attributions_{ag_pos}_vs_{ag_neg}.tsv"
        attributions_ag_pos_fig_path = out_dir_i / \
            f"aggregated_attributions_{ag_pos}.png"
        attributions_ag_neg_fig_path = out_dir_i / \
            f"aggregated_attributions_{ag_neg}.png"

        test_data, model = process_data_and_train_model(
            ag_pos,
            ag_neg,
            learning_rate,
            epochs,
            data_path
        )

        run_attribution_workflow(
            ag_pos,
            ag_neg,
            attributions_savepath,
            agg_attributions_savepath,
            attributions_ag_neg_fig_path,
            test_data,
            model
        )
