"""
Build datasets.
"""

import json
from pathlib import Path
from itertools import combinations
from docopt import docopt
import logging
import pandas as pd

import NegativeClassOptimization.utils as utils
import NegativeClassOptimization.config as config
import NegativeClassOptimization.datasets as datasets
import NegativeClassOptimization.preprocessing as preprocessing


dataset_path: Path = config.DATA_SLACK_1_RAW_DIR

docopt_doc = """Build datasets.

Usage:
    script_01_build_datasets.py global
    script_01_build_datasets.py add_farmhash_mod_10_to_global
    script_01_build_datasets.py processed
    script_01_build_datasets.py pairwise
    script_01_build_datasets.py 1_vs_all
    script_01_build_datasets.py download_absolut
    script_01_build_datasets.py absolut_processed_multiclass
    script_01_build_datasets.py absolut_processed_multilabel

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
    df_train_val = df_train_val.loc[df_train_val["Antigen"].isin(represented_antigens)].copy()
    df_test_closed_exclusive = df_test_closed_exclusive.loc[df_test_closed_exclusive["Antigen"].isin(represented_antigens)].copy()

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

    df_train_val.to_csv(out_dir / "df_train_val.tsv", sep='\t')
    df_test_closed_exclusive.to_csv(out_dir / "df_test_closed_exclusive.tsv", sep='\t')
    df_test_open_exclusive.to_csv(out_dir / "df_test_open_exclusive.tsv", sep='\t')

    with open(out_dir / "build_metadata.json", "w+") as fh:
        json.dump(metadata, fh)


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
    
    antigens = sorted(df_global["Antigen"].unique().tolist())

    if arguments["add_farmhash_mod_10_to_global"]:
        df_global["Slide_farmhash_mod_10"] = (
            df_global["Slide"].apply(
                lambda x: preprocessing.farmhash_mod_10(x)
            )
        )

        dir_ = config.DATA_SLACK_1_GLOBAL.parent
        basename = config.DATA_SLACK_1_GLOBAL.stem
        df_global.to_csv(
            dir_ / f"{basename}_farmhashed.tsv", 
            sep='\t')

    elif arguments["pairwise"]:
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
        
        df_train_val, df_test_closed_exclusive, df_test_open_exclusive = (
            preprocessing.openset_datasplit_from_global_stable(df_global)
        )

        df_train_val.to_csv(out_dir / "df_train_val.tsv", sep='\t')
        df_test_closed_exclusive.to_csv(out_dir / "df_test_closed_exclusive.tsv", sep='\t')
        df_test_open_exclusive.to_csv(out_dir / "df_test_open_exclusive.tsv", sep='\t')

        meta = {
            "df_train_val__shape": df_train_val.shape,
            "df_test_closed_exclusive__shape": df_test_closed_exclusive.shape,
            "df_test_open_exclusive__shape": df_test_open_exclusive.shape,
        }
        with open(out_dir / "build_metadata.json", "w+") as fh:
            json.dump(meta, fh)

    elif arguments["download_absolut"]:
        utils.download_absolut()
    
    elif arguments["absolut_processed_multiclass"] or arguments["absolut_processed_multilabel"]:
        
        ds3 = datasets.AbsolutDataset3()
        ags_closed, ags_open = get_closed_open_antigens(ds3)
        
        df_wide = ds3.df_wide
        mask_c = (df_wide[ags_closed].sum(axis=1) >= 1) & (df_wide[ags_open].sum(axis=1) == 0)
        mask_o = (df_wide[ags_closed].sum(axis=1) == 0) & (df_wide[ags_open].sum(axis=1) >= 1)
        if arguments["absolut_processed_multiclass"]:
            # Filter for unimodal binding and exclusive open set and closed sets.
            mask_unimodal = df_wide.sum(axis=1) == 1
            df_wide = df_wide.loc[(mask_unimodal & mask_c) | (mask_unimodal & mask_o)].copy()
            out_dir = config.DATA_ABSOLUT_PROCESSED_MULTICLASS_DIR
        elif arguments["absolut_processed_multilabel"]:
            df_wide = df_wide.loc[(mask_c) | (mask_o)].copy()
            out_dir = config.DATA_ABSOLUT_PROCESSED_MULTILABEL_DIR

        process_downstream_and_save(out_dir, ags_open, df_wide)
