"""
Build all the pairwise datasets.
"""

from itertools import combinations
import pandas as pd

import config
import datasets


if __name__ == "__main__":

    df_global = pd.read_csv(config.DATA_SLACK_1_GLOBAL, sep='\t')
    antigens = sorted(df_global["Antigen"].unique().tolist())

    for ag1, ag2 in combinations(antigens, 2):
        datasets.generate_pairwise_dataset(
            df_global,
            ag1 = ag1,
            ag2 = ag2,
            read_if_exists = False
        )