from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import NegativeClassOptimization.config as config


TEST = False
TEST_PARAMS = {
    "TEST_DATA_DIR": "../data/CompAIRR/test",
    "SAMPLE_SIZE": 1000,
    "AG1": "1ADQ",
    "AG2": "3VRL",
}


def convert_internal_to_AIRR_format(
    df: pd.DataFrame
    ) -> pd.DataFrame:
    """Convert internal global dataset format to AIRR format.
    https://github.com/uio-bmi/compairr#input-files

    Args:
        df (pd.DataFrame): dataframe in internal "global" format.

    Returns:
        pd.DataFrame: dataframe in AIRR format.
    """    
    column_map = {
        "Antigen": "repertoire_id",
        "UID": "sequence_id",
        "Slide": "junction_aa",
    }
    df_AIRR = df[list(column_map.keys())].copy()
    df_AIRR = df_AIRR.rename(columns=column_map)
    return df_AIRR


if __name__ == "__main__":
    
    np.random.seed(config.SEED)

    df = pd.read_csv(config.DATA_SLACK_1_GLOBAL, sep='\t')
    
    if not TEST:
        dir_path = Path("../data/CompAIRR")
        dir_path.mkdir(exist_ok=True)

        df.drop_duplicates(["Antigen", "Slide"], inplace=True)

        df_1 = convert_internal_to_AIRR_format(df)
        df_2 = df_1.copy()

        # df_1["repertoire_id"] = df_1["repertoire_id"].str + "_1"
        # df_2["repertoire_id"] = df_2["repertoire_id"].str + "_2"
        df_1.to_csv(dir_path / "AIRR_1.tsv", sep='\t', index=False)
        df_2.to_csv(dir_path / "AIRR_2.tsv", sep='\t', index=False)
        
    else:
        test_data_dir = Path(TEST_PARAMS["TEST_DATA_DIR"])
        test_data_dir.mkdir(exist_ok=True)
        
        df_1 = convert_internal_to_AIRR_format(
            df.loc[df["Antigen"] == TEST_PARAMS["AG1"]].sample(n=TEST_PARAMS["SAMPLE_SIZE"])
        )
        df_2 = convert_internal_to_AIRR_format(
            df.loc[df["Antigen"] == TEST_PARAMS["AG2"]].sample(n=TEST_PARAMS["SAMPLE_SIZE"])
        )

        df_1.to_csv(test_data_dir / "AG1_AIRR.tsv", sep='\t', index=False)
        df_2.to_csv(test_data_dir / "AG2_AIRR.tsv", sep='\t', index=False)
