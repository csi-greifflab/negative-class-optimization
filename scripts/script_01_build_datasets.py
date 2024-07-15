"""
Build datasets.
"""

import json
import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from docopt import docopt

import NegativeClassOptimization.config as config
import NegativeClassOptimization.datasets as datasets
import NegativeClassOptimization.preprocessing as preprocessing
import NegativeClassOptimization.utils as utils
from NegativeClassOptimization.utils import (save_train_test_rest,
                                             split_to_train_test_rest_dfs)

dataset_path: Path = config.DATA_SLACK_1_RAW_DIR


docopt_doc = """Build datasets.

Usage:
    script_01_build_datasets.py download_absolut
    script_01_build_datasets.py unzip_rawbindingsmurine
    script_01_build_datasets.py miniabsolut <N_train> <N_test>
    script_01_build_datasets.py frozen_results
    script_01_build_datasets.py adapt_attributions_for_linear
    script_01_build_datasets.py add_freq_based_attributions


Options:
    -h --help   Show help.
"""

logging.basicConfig(level=logging.DEBUG)


def get_closed_open_antigens(ds3):
    num_closed_ags = config.NUM_CLOSED_ANTIGENS_ABSOLUT_DATASET3
    ags_shuffled = utils.shuffle_antigens(ds3.antigens)
    ags_closed = ags_shuffled[:num_closed_ags]
    ags_open = ags_shuffled[num_closed_ags:]
    return ags_closed, ags_open


def process_downstream_and_save(out_dir, ags_open, df_wide):
    df_global = preprocessing.convert_wide_to_global(df_wide)

    (
        df_train_val,
        df_test_closed_exclusive,
        df_test_open_exclusive,
    ) = preprocessing.openset_datasplit_from_global_stable(
        df_global=df_global,
        openset_antigens=ags_open,
    )

    ag_counts = df_train_val["Antigen"].value_counts()
    represented_antigens = ag_counts.loc[ag_counts > 1000].index.tolist()
    df_train_val = df_train_val.loc[
        df_train_val["Antigen"].isin(represented_antigens)
    ].copy()
    df_test_closed_exclusive = df_test_closed_exclusive.loc[
        df_test_closed_exclusive["Antigen"].isin(represented_antigens)
    ].copy()

    # dfs = {
    #         "df_train_val": df_train_val,
    #         "df_test_closed_exclusive": df_test_closed_exclusive,
    #         "df_test_open_exclusive": df_test_open_exclusive,
    #     }

    metadata = {
        "df_train_val__shape": df_train_val.shape,
        "df_test_closed_exclusive__shape": df_test_closed_exclusive.shape,
        "df_test_open_exclusive__shape": df_test_open_exclusive.shape,
        "ags_closed": represented_antigens,
    }

    df_train_val.to_csv(out_dir / "df_train_val.tsv", sep="\t")
    df_test_closed_exclusive.to_csv(out_dir / "df_test_closed_exclusive.tsv", sep="\t")
    df_test_open_exclusive.to_csv(out_dir / "df_test_open_exclusive.tsv", sep="\t")

    with open(out_dir / "build_metadata.json", "w+") as fh:
        json.dump(metadata, fh)


if __name__ == "__main__":

    arguments = docopt(docopt_doc, version="NCO")

    if arguments["download_absolut"]:
        utils.download_absolut(antigens_only=True)

    elif arguments["unzip_rawbindingsmurine"]:
        input_dir = Path("data/Absolut/data/RawBindingsMurine")
        output_dir = Path("data/Absolut/data/RawBindingsMurine/unzipped")
        output_dir.mkdir(exist_ok=True, parents=True)

        print(f"Unzipping all files in {input_dir} to {output_dir}...")
        for file in input_dir.glob("*.zip"):
            if file.stem.split("_")[0] not in config.ANTIGENS:
                print(f"Skipping {file}, not in Mini-Absolut")
                continue
            print(f"Unzipping {file} to {output_dir / file.stem}")
            utils.unzip_file(file, output_dir / file.stem)

    elif arguments["miniabsolut"]:

        N_train =  int(arguments['<N_train>'])
        N_test =  int(arguments['<N_test>'])

        make_splits_l = [False, True, True, True, True, True, True]
        seed_l = [None, 0, 1, 2, 3, 4, None]

        if not Path(config.DATA_MINIABSOLUT_SPLITS).exists():
            Path(config.DATA_MINIABSOLUT_SPLITS).mkdir(exist_ok=True, parents=True)

        for make_splits, seed in zip(make_splits_l, seed_l):

            if make_splits:
                base_p = Path(config.DATA_MINIABSOLUT_SPLITS) / f"MiniAbsolut_Seed{seed}"
                base_p.mkdir(exist_ok=True, parents=False)
            else:
                base_p = config.DATA_MINIABSOLUT
                base_p.mkdir(exist_ok=True, parents=False)

            # Load the Absolut binding data for the antigens in Mini-Absolut.
            dfs = []
            for ag in config.ANTIGENS:
                # Load based on config.DATA_SLACK_1_RAWBINDINGS_PERCLASS_MURINE
                df = utils.load_binding_per_ag(ag)
                df["Antigen"] = ag
                dfs.append(df)
            df = pd.concat(dfs, axis=0)

            # Get mascotte
            df_m = df.loc[df["Source"] == "mascotte"].copy()

            # Get mascotte without duplicates
            df_m_nodup = df_m.loc[~df_m["Slide"].duplicated(keep=False)].copy()

            for ag in config.ANTIGENS:
                print(f"Processing {ag}...")

                ag_dir = base_p / ag
                ag_dir.mkdir(exist_ok=True, parents=False)

                # Get the mascotte data for the antigen.
                df_ag = df_m_nodup.loc[df_m_nodup["Antigen"] == ag].copy()
                df_ag = df_ag.loc[df_ag["Source"] == "mascotte"].copy()
                df_ag = df_ag.sample(frac=1).reset_index(drop=True)  # shuffle
                print(f"mascotte: {df_ag.shape}")
                df_train, df_test, df_rest = split_to_train_test_rest_dfs(
                    N_train,
                    N_test,
                    df_ag,
                    random_state=seed,
                )
                save_train_test_rest(
                    "high", N_train, N_test, ag_dir, df_train, df_test, df_rest
                )

                # Get looserX data for the antigen.
                df_ag = df.loc[df["Antigen"] == ag].copy()

                df_ag = df_ag.loc[
                    ~df_ag["Slide"].isin(
                        df_m_nodup.loc[df_m_nodup["Antigen"] == ag]["Slide"]
                    )
                ].copy()

                df_ag.sort_values(
                    by="Energy", ascending=False, inplace=True
                )  # make ascending == True ?
                df_ag = df_ag.loc[~df_ag.duplicated(subset=["Slide"], keep="first")].copy()
                df_ag = df_ag.loc[df_ag["Source"] == "looserX"].copy()
                df_ag = df_ag.sample(frac=1).reset_index(drop=True)  # shuffle
                print(f"looserX: {df_ag.shape}")
                df_train, df_test, df_rest = split_to_train_test_rest_dfs(
                    N_train,
                    N_test,
                    df_ag,
                    random_state=seed,
                )
                save_train_test_rest(
                    "looserX", N_train, N_test, ag_dir, df_train, df_test, df_rest
                )

                # Get 95low data for the antigen.
                df_ag = df.loc[df["Antigen"] == ag].copy()

                df_ag = df_ag.loc[
                    ~df_ag["Slide"].isin(
                        df_m_nodup.loc[df_m_nodup["Antigen"] == ag]["Slide"]
                    )
                ].copy()

                df_ag.sort_values(
                    by="Energy", ascending=False, inplace=True
                )  # make ascending == True ? Keep Slide with lowest energy?
                df_ag = df_ag.loc[~df_ag.duplicated(subset=["Slide"], keep="first")].copy()

                # Exclude 95low intersection with looserX.
                energy_5p_cutoff = df_ag[df_ag["Source"] == "looserX"]["Energy"].max()
                df_ag = df_ag.loc[df_ag["Energy"] >= energy_5p_cutoff].copy()  # make ">" ?

                df_ag = df_ag.sample(frac=1).reset_index(drop=True)  # shuffle
                print(f"95low: {df_ag.shape}")
                df_train, df_test, df_rest = split_to_train_test_rest_dfs(
                    N_train,
                    N_test,
                    df_ag,
                    random_state=seed,
                )
                save_train_test_rest(
                    "95low", N_train, N_test, ag_dir, df_train, df_test, df_rest
                )

    elif arguments["frozen_results"]:
        # Save the frozen results in a convenient format for sharing and further analysis.
        # For each tasks of interest: high vs looser, high vs 95low, 1v1 and 1v9.

        ### GOTO notebook 02b

        pass

    elif arguments["adapt_attributions_for_linear"]:
        # Adapt the attributions for the linear model,
        # in a similar way as for the SN10, to achieve
        # compatibility with the Energy-logit code.

        ## 2) Linear models with shuffled weights
        ## We generate here (by differential commenting :))
        ## the attributions for the linear models with shuffled weights.
        ## Check the commented sections below for the code. They are
        ## tagged with #SHUFFLED WEIGHTS.

        linear_dir = Path(config.DATA_LINEAR_ML)

        # Glob directories from Path
        task_type_dirs = list(linear_dir.glob("*"))

        for task_type_dir in task_type_dirs:
            # Get seed directories
            seed_dirs = list(task_type_dir.glob("*"))
            for seed_dir in seed_dirs:
                # Get split directories
                split_dirs = list(seed_dir.glob("*"))
                for split_dir in split_dirs:
                    # Get task directories
                    task_dirs = list(split_dir.glob("*"))
                    for task_dir in task_dirs:
                        print("Processing:", task_dir)
                        # Load pytorch model
                        model = torch.load(task_dir / "swa_model/data/model.pth")
                        # Get linear weights
                        weights = (
                            model.state_dict()["module.linear.weight"].numpy().tolist()
                        )  # Shape 1x220
                        bias: float = float(
                            model.state_dict()["module.linear.bias"].numpy()[0]
                        )

                        ## #SHUFFLED WEIGHTS
                        ## Input shape 1x220, shuffle operates on first axis
                        ## of array. That is why the operation is performed
                        ## on weights[0].
                        # np.random.shuffle(weights[0])

                        attributions_dir = task_dir / "attributions"
                        attributions_dir.mkdir(exist_ok=True, parents=False)
                        attributions_dir = attributions_dir / "v0.1.2-3"
                        ## #SHUFFLED WEIGHTS: comment above line and uncomment below
                        # attributions_dir = (
                        #     attributions_dir / "v0.1.2-3_shuffled_weights"
                        # )
                        attributions_dir.mkdir(exist_ok=True, parents=False)

                        # Get the test dataset
                        test_dataset_path = list(task_dir.glob("*test_dataset*"))[0]
                        df_test = pd.read_csv(test_dataset_path, sep="\t")

                        # Compute on the test dataset
                        # and record the logits and the coefficients / "attributions"
                        # (which are constant for logistic regression).
                        records = []
                        for i, row in df_test.iterrows():
                            # Get logits from torch LogisticRegression model
                            slide = row["Slide"]
                            slide_onehot = preprocessing.onehot_encode(slide)
                            logits = model.module.forward_logits(
                                torch.tensor(slide_onehot).type_as(
                                    model.module.linear.weight
                                )
                            )
                            records.append(
                                {
                                    "slide": slide,
                                    "logits": logits.detach().numpy().tolist()[0],
                                    "weights": weights,
                                    "bias": bias,
                                    "y_true": row["y"],
                                }
                            )
                        # Save json
                        with open(
                            attributions_dir / "attribution_records.json", "w"
                        ) as f:
                            json.dump(records, f)

    elif arguments["add_freq_based_attributions"]:
        
        # Adapted code based on "adapt_attributions_for_linear"

        linear_dir = Path(config.DATA_ML)
        # Glob directories from Path
        task_type_dirs = list(linear_dir.glob("*"))

        for task_type_dir in task_type_dirs:
            # Get seed directories
            seed_dirs = list(task_type_dir.glob("*"))
            for seed_dir in seed_dirs:
                # Get split directories
                split_dirs = list(seed_dir.glob("*"))
                for split_dir in split_dirs:
                    # Get task directories
                    task_dirs = list(split_dir.glob("*"))
                    for task_dir in task_dirs:
                        print("Processing:", task_dir)
                        
                        attributions_dir = task_dir / "attributions/v2.0-2"
                        attribution_records_path = attributions_dir / "attribution_records.json"
                        if not attribution_records_path.exists():
                            continue

                        with open(attribution_records_path, 'r') as f:
                            attr_records = json.load(f)

                        all_slides = [attr['slide'] for attr in attr_records]
                        pos_slides = [attr['slide'] for attr in attr_records if attr['y_true'] == 1]
                        neg_slides = [attr['slide'] for attr in attr_records if attr['y_true'] == 0]

                        try:
                            ohs_freq, ohs_freq_rel = preprocessing.compute_frequencies_and_relative(all_slides)
                            ohs_freq_pos, ohs_freq_rel_pos = preprocessing.compute_frequencies_and_relative(pos_slides)
                            ohs_freq_neg, ohs_freq_rel_neg = preprocessing.compute_frequencies_and_relative(neg_slides)
                        except ValueError:
                            print("Error computing frequencies for:", task_dir)
                            continue

                        freqs, rel_freqs = preprocessing.extract_frequences_as_features(all_slides, ohs_freq, ohs_freq_rel)
                        freqs_pos, rel_freqs_pos = preprocessing.extract_frequences_as_features(all_slides, ohs_freq_pos, ohs_freq_rel_pos)
                        freqs_neg, rel_freqs_neg = preprocessing.extract_frequences_as_features(all_slides, ohs_freq_neg, ohs_freq_rel_neg)

                        for i, attr in enumerate(attr_records):
                            attr['freq'] = freqs[i].tolist()
                            attr['rel_freq'] = rel_freqs[i].tolist()
                            attr['freq_pos'] = freqs_pos[i].tolist()
                            attr['rel_freq_pos'] = rel_freqs_pos[i].tolist()
                            attr['freq_neg'] = freqs_neg[i].tolist()
                            attr['rel_freq_neg'] = rel_freqs_neg[i].tolist()

                        # Save the updated records
                        augmented_path = attributions_dir / "attribution_records_augmented.json"
                        with open(augmented_path, "w") as f:
                            json.dump(attr_records, f)
