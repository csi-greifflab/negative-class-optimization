"""
Build datasets.
"""

from pathlib import Path
from itertools import combinations
from docopt import docopt
import logging
import pandas as pd

import NegativeClassOptimization.utils as utils
import NegativeClassOptimization.config as config
import NegativeClassOptimization.datasets as datasets


dataset_path: Path = config.DATA_SLACK_1_RAW_DIR

docopt_doc = """Build datasets.

Usage:
    script_01_build_datasets.py global
    script_01_build_datasets.py processed
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
        dataset_name = dataset_path.name
        df_global = utils.build_global_dataset(dataset_path, remove_ag_slide_duplicates=True)
        config.DATA_SLACK_1_GLOBAL.parent.mkdir(exist_ok=True)
        df_global.to_csv(config.DATA_SLACK_1_GLOBAL, sep='\t')
    else:
        df_global = pd.read_csv(
            config.DATA_SLACK_1_GLOBAL, 
            sep='\t', 
            dtype={"Antigen": str})
    
    print(df_global.Antigen.unique())
    antigens = sorted(df_global["Antigen"].unique().tolist())

    if arguments["pairwise"]:
        logging.info("Building pairwise datasets")
        for ag1, ag2 in combinations(antigens, 2):
            logging.info(f"Building pairwise dataset: {ag1} vs {ag2}")
            datasets.generate_pairwise_dataframe(
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
    elif arguments["processed"]:
        logging.info("Building openset_exclusive dataset")
        
        out_dir = Path(config.DATA_SLACK_1_PROCESSED_DIR)
        out_dir.mkdir(exist_ok=True)
        
        mask_ = df_global["Antigen"].isin(config.ANTIGENS_OPENSET)
        df_closed = df_global.loc[~mask_].copy()
        df_open = df_global.loc[mask_].copy()
        df_test_open_exclusive = df_open.loc[~df_open["Slide"].isin(df_closed["Slide"])].copy()
        
        df_train_val_closed = None
        df_test_closed_exclusive = None
        df_test_open_exclusive.to_csv(out_dir / "df_test_open_exclusive.tsv", sep='\t')

