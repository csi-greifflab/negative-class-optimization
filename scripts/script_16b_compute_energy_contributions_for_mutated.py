"""
Script to compute the energy contributions of each aminoacid from a Slide.
Uses the new feature of Absolut! developed by Philippe.
Make sure to save as environment variable the path to the Absolut NoLib executable. (ABSOLUTNOLIB_PATH)
"""

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

# Load from the env var
ABSOLUTNOLIB_PATH = os.environ["ABSOLUTNOLIB_PATH"]

if __name__ == "__main__":
    # 1) For each antigen, copy each tsv with slides to a local folder
    #    and change format, so that Absolut can process it.
    #    This is done only for *test* files.
    filepaths = {}  # list of filepaths to be used in the next step

    for path in Path("data/mutants/mutant_igs/").glob("*.csv"):
        if "shuffled" in str(path):
            continue
        str_describing_task = path.name.split("_mut_igs")[0]

        if "__vs__" in str_describing_task:
            # ONE_VS_ONE: compute once for positive and once for negative antigen.
            antigen = str_describing_task.split("__vs__")[
                0
            ]  # run with both 0 and 1, to get both.
        else:
            antigen = str_describing_task[:4]

        ## Change format of test file to be compatible with Absolut
        # 1.1) Get Antigen name to later add a 1st row e.g. #Antigen 1ADQ_A
        antigen_name_row = f"#Antigen {antigen}"
        # 1.2) Read the file as a dataframe. Remove the index column. Rename columns.
        df = pd.read_csv(path)

        df = df.iloc[:, :6]
        df.columns = [
            "ID",
            "CDR3Sequence",
            "BestForThisCDR?('true'/'false')",
            "Slide",
            "Energy",
            "Structure",
        ]
        df["BestForThisCDR?('true'/'false')"] = df[
            "BestForThisCDR?('true'/'false')"
        ].replace({True: "true", False: "false"})
        df.head()

        # 1.3) Save the file
        new_path = (
            path.parent.parent
            / "mutant_energy_contributions"
            / f"{path.stem}_energy_contributions_input.tsv"
        )
        new_path.write_text(antigen_name_row + "\n" + df.to_csv(sep="\t", index=False))
        logging.info(
            f"Changed format of {path} in new {new_path} to be compatible with Absolut"
        )
        if antigen not in filepaths.keys():
            filepaths[antigen] = []
        filepaths[antigen].append(new_path)

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
                    str(fp.parent) + f"/{fp.stem[:-5]}for_{antigen_fullname}.tsv",
                    "1",
                    "true",
                ]
            )
