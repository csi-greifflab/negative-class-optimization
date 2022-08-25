# import abc
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, List
import warnings
import pandas as pd
import numpy as np
# from tinydb import TinyDB, Query

import torch
from torch.utils.data import Dataset

import sys
sys.path.append('/nfs/scistore08/kondrgrp/aminnega/negative-class-optimization/src/NegativeClassOptimization')


import NegativeClassOptimization.utils as utils
import NegativeClassOptimization.config as config


def trim_short_CDR3(df) -> pd.DataFrame:
    LEN_TH = 25
    cdr3_len_counts = df["CDR3"].str.len().value_counts()
    small_lengths = (
        cdr3_len_counts
        .loc[cdr3_len_counts < LEN_TH]
        .index.to_list()
    )
    small_len_mask = df["CDR3"].str.len().isin(small_lengths)
    warnings.warn(
        "Removal of sequences with len<25 can be not enough" 
        "for onehot encoding!"
        f"\nRemoving rare CDR3 lengths: {sum(small_len_mask)}"
        f" rows from {df.shape}"
    )
    df = df.loc[~small_len_mask]
    return df


def generate_pairwise_dataset(
    df_global: pd.DataFrame,
    ag1: str,
    ag2: str,
    N: Optional[int] = None,
    base_data_path: Path = config.DATA_BASE_PATH,
    seed = config.SEED,
    read_if_exists = True,
    Slide=True,
    save_datasets=True
    ):

    if Slide:
        filepath = base_data_path / "pairwise_wo_dupl" / f"pairwise_dataset_{ag1}_{ag2}.tsv"
    elif N:
        filepath = base_data_path / "pairwise" / f"pairwise_dataset_{ag1}_{ag2}_{N}.tsv"
    else:
        filepath = base_data_path / "pairwise" / f"pairwise_dataset_{ag1}_{ag2}.tsv"

    if read_if_exists and filepath.exists():
        df = pd.read_csv(filepath, sep='\t')
        return df

    df_ag1 = df_global.loc[df_global["Antigen"] == ag1].copy()
    df_ag2 = df_global.loc[df_global["Antigen"] == ag2].copy()
    
    if Slide:
        df_ag1.drop_duplicates(subset='Slide', keep="last", inplace=True)
        df_ag2.drop_duplicates(subset='Slide', keep="last", inplace=True)
        inters = set(df_ag1.Slide)&set(df_ag2.Slide)
        df_ag2 = df_ag2[df_ag2['Slide'].isin(inters) == False]
        df_ag1 = df_ag1[df_ag1['Slide'].isin(inters) == False]
    else:
        inters = set(df_ag1.CDR3)&set(df_ag2.CDR3)
        df_ag2 = df_ag2[df_ag2['CDR3'].isin(inters) == False]
        df_ag1 = df_ag1[df_ag1['CDR3'].isin(inters) == False]   

    if N:
        np.random.seed(seed)
        df_ag1 = df_ag1.sample(N // 2)
        df_ag2 = df_ag2.sample(N // 2)

    df_ag1["binder"] = True
    df_ag2["binder"] = False

    df = pd.concat([df_ag1, df_ag2], axis=0)

    #df = trim_short_CDR3(df)
    df = df.sample(frac=1, random_state=config.SEED)
    if save_datasets:
        df.to_csv(filepath, sep='\t')

    return df


def generate_1_vs_all_dataset(
    df_global: pd.DataFrame,
    ag: str,
    base_data_path: Path = config.DATA_BASE_PATH,
    seed = config.SEED,
    ) -> pd.DataFrame:
    df = df_global.copy()
    df["binder"] = df["Antigen"] == ag
    df = trim_short_CDR3(df)
    
    filepath = base_data_path / "1_vs_all" / f"{ag}_vs_all_dataset.tsv"
    df.to_csv(filepath, sep='\t')
    
    return df


class BinaryDataset(Dataset):
    """Pytorch dataset for modelling antigen binding binary classifiers.
    """

    def __init__(self, df):
        self.df = df

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        return (
            torch.tensor(self.df.loc[idx, "X"]).reshape(
                (1, -1)).type(torch.float),
            torch.tensor(self.df.loc[idx, "y"]).reshape((1)).type(torch.float),
        )


class MulticlassDataset(Dataset):
    """Pytorch dataset for modelling antigen binding multiclass classifiers.
    """

    def __init__(self, df):
        self.df = df

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        return (
            torch.tensor(self.df.loc[idx, "X"]).reshape(
                (1, -1)).type(torch.float),
            torch.tensor(self.df.loc[idx, "y"]).reshape((1)).type(torch.uint8),
        )    
