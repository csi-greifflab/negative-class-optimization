"""
Build datasets.
"""

from pathlib import Path
from itertools import combinations
from docopt import docopt
import logging
import pandas as pd
import sys
sys.path.append('/nfs/scistore08/kondrgrp/aminnega/negative-class-optimization/src/NegativeClassOptimization')

import NegativeClassOptimization.utils as utils
import NegativeClassOptimization.config as config
import NegativeClassOptimization.datasets as datasets


dataset_path: Path = config.DATA_SLACK_1

docopt_doc = """Build datasets.

Usage:
    script_01_build_datasets.py global
    script_01_build_datasets.py pairwise
    script_01_build_datasets.py 1_vs_all

Options:
    -h --help   Show help.

"""

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":

    arguments = docopt(docopt_doc, version='Naval Fate 2.0')

    if arguments["global"]:
        logging.info("Building the global dataset.")
        global_datasets_dir: Path = config.GLOBAL_DATASETS_DIR
        dataset_name = dataset_path.name
        df_global = utils.build_global_dataset(dataset_path)
        df_global.to_csv(global_datasets_dir / f"{dataset_name}_global.tsv", sep='\t')
    else:
        df_global = pd.read_csv(config.DATA_SLACK_1_GLOBAL, sep='\t')
    
    antigens = sorted(df_global["Antigen"].unique().tolist())
    
    if arguments["pairwise"]:
        logging.info("Building pairwise datasets")
        for ag1, ag2 in combinations(antigens, 2):
            logging.info(f"Building pairwise dataset: {ag1} vs {ag2}")
            datasets.generate_pairwise_dataset(
                df_global,
                ag1 = ag1,
                ag2 = ag2,
                read_if_exists = False
            )
    elif arguments["1_vs_all"]:     
        logging.info("Building 1_vs_all datasets")   
        for ag in antigens:
            logging.info(f"Building 1_vs_all dataset: {ag}")
            datasets.generate_1_vs_all_dataset(df_global, ag)
