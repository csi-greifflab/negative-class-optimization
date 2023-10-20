"""
Script to compute the energy contributions of each aminoacid from a Slide.
Uses the new feature of Absolut! developed by Philippe.
Make sure to save as environment variable the path to the Absolut NoLib executable. (ABSOLUTNOLIB_PATH)
"""

import itertools
import logging
import math
import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from NegativeClassOptimization import config, ml, preprocessing, utils

# Default
# PATH = config.DATA_MINIABSOLUT
# PATH = Path("data/MiniAbsolut_Splits/MiniAbsolut_Seed4")
PATH = Path("data/MiniAbsolut")

# Load from the env var
ABSOLUTNOLIB_PATH = os.environ["ABSOLUTNOLIB_PATH"]

if __name__ == "__main__":
    # 1) For each antigen, copy each tsv with slides to a local folder
    #    and change format, so that Absolut can process it.
    #    This is done only for *test* files.
    filepaths = {}  # list of filepaths to be used in the next step
    for ag_path in PATH.glob("*"):
        antigen_name = ag_path.stem
        filepaths[antigen_name] = []
        # Create a local folder for results
        energy_contributions_dir = ag_path / "energy_contributions"
        energy_contributions_dir.mkdir(exist_ok=True)
        logging.info(f"Created {energy_contributions_dir}")
        for test_slide_fp in itertools.chain(
            ag_path.glob("*train_15000.tsv"),
            ag_path.glob("*test_5000.tsv"),
            ag_path.glob("*rest.tsv"),
        ):
            # Copy test file to a local folder
            test_slide_fp_for_energy_contributions = (
                energy_contributions_dir / f"{test_slide_fp.stem}_absolut.tsv"
            )
            test_slide_fp_for_energy_contributions.write_text(test_slide_fp.read_text())
            logging.info(
                f"Copied {test_slide_fp} to {test_slide_fp_for_energy_contributions}"
            )
            ## Change format of test file to be compatible with Absolut
            # 1.1) Get Antigen name to later add a 1st row e.g. #Antigen 1ADQ_A
            antigen_name_row = f"#Antigen {antigen_name}"
            # 1.2) Read the file as a dataframe. Remove the index column. Rename columns.
            test_slide_df = pd.read_csv(
                test_slide_fp_for_energy_contributions, sep="\t", index_col=0, header=1
            )
            test_slide_df = test_slide_df.iloc[:, :-2]
            test_slide_df.columns = [
                "ID",
                "CDR3Sequence",
                "BestForThisCDR?('true'/'false')",
                "Slide",
                "Energy",
                "Structure",
            ]
            test_slide_df["BestForThisCDR?('true'/'false')"] = test_slide_df[
                "BestForThisCDR?('true'/'false')"
            ].replace({True: "true", False: "false"})
            # 1.3) Save the file
            test_slide_fp_for_energy_contributions.write_text(
                antigen_name_row + "\n" + test_slide_df.to_csv(sep="\t", index=False)
            )
            logging.info(
                f"Changed format of {test_slide_fp_for_energy_contributions} to be compatible with Absolut"
            )
            filepaths[antigen_name].append(test_slide_fp_for_energy_contributions)

    ## 2) Run Absolut! on each file.
    ##    For each of the file in the list above, run Absolut! and save the results the same folder
    ##    > $ABSOLUTNOLIB_PATH getFeatures <ANTIGEN> <INPUT_PATH> <OUTPUT_PATH> 1 true
    for antigen in filepaths.keys():
        for fp in filepaths[antigen]:
            logging.info(f"Running Absolut! on {fp}")

            # Convert antigen name to the full antigen name,
            # which is expected by Absolut.
            antigen_fullname = utils.convert_antigen_name_to_full_name(antigen)
            subprocess.run(
                [
                    ABSOLUTNOLIB_PATH,
                    "getFeatures",
                    antigen_fullname,
                    str(fp),
                    str(fp.parent) + f"/{fp.stem}_energy_contributions.tsv",
                    "1",
                    "true",
                ]
            )
