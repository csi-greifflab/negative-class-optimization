# import abc
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, List
import warnings
import pandas as pd
import numpy as np
# from tinydb import TinyDB, Query
import sys
sys.path.append('/nfs/scistore08/kondrgrp/aminnega/negative-class-optimization/src/NegativeClassOptimization')


import NegativeClassOptimization.utils as utils
import NegativeClassOptimization.config as config


class BindingDatasetsType(Enum):
    BASE = 0
    RANDOM = 1
    POSITIVE = 2
    NEGATIVE = 3
    COMPLETE = 4


class BaseBindingDataset:

    DF_VALIDATION_COLS = ["CDR3", "UID", "Antigen"]

    def __init__(
        self, 
        name: str, 
        dstype: BindingDatasetsType, 
        df: pd.DataFrame) -> None:
        self.name = name
        self.dstype = dstype
        self.df = df.copy()
        
        if not self.validate_df():
            raise ValueError("Failed to init BaseBindingDataset")
    
    def validate_df(self) -> bool:
        return all(map(
                lambda validation_col: validation_col in self.df.columns,
                BaseBindingDataset.DF_VALIDATION_COLS
        ))


class RandomBindingDataset(BaseBindingDataset):

    def __init__(self, name: str, num_seq: int):

        dstype = BindingDatasetsType.RANDOM
        df = utils.build_random_dataset(num_seq)
        df["Dataset"] = name

        assert df["Antigen"].unique()[0] == "random"
        super().__init__(name, dstype, df)


class PositiveBindingDataset(BaseBindingDataset):

    def __init__(
        self,
        name: str,
        df: pd.DataFrame,
        antigen: str) -> None:
        
        self.antigen = antigen

        dstype = BindingDatasetsType.POSITIVE
        df_ = df.copy()
        df_["Dataset"] = name
        super().__init__(name, dstype, df_)

        if not self.validate_positive_dataset():
            raise ValueError("Failed to init PositiveBindingDataset")

    def validate_positive_dataset(self):
        return self.df["Antigen"].unique()[0] == self.antigen


class NegativeBindingDataset(BaseBindingDataset):
    
    def __init__(
        self,
        name: str,
        positive_datasets: Optional[List[PositiveBindingDataset]] = None,
        random_datasets: Optional[List[RandomBindingDataset]] = None
        ) -> None:
        
        if (
            (positive_datasets is None and random_datasets is None) 
            or (len(positive_datasets) == len(random_datasets) == 0)
            ):
            raise ValueError("Datasets not provided or empty")

        self.positive_datasets = positive_datasets
        self.random_datasets = random_datasets
        df = pd.concat(map(lambda ds: ds.df, positive_datasets + random_datasets), axis=0)
        df = df[BaseBindingDataset.DF_VALIDATION_COLS + ["Dataset"]]

        dstype = BindingDatasetsType.NEGATIVE
        super().__init__(name, dstype, df)


class CompleteBindingDataset(BaseBindingDataset):

    # DB_PATH = config.COMPLETE_DATASETS_DB_PATH

    def __init__(
        self,
        name: str,
        positive_dataset: PositiveBindingDataset,
        negative_dataset: NegativeBindingDataset,
        ) -> None:
        
        self.positive_dataset = positive_dataset
        self.negative_dataset = negative_dataset


        df_pos = positive_dataset.df.copy()
        df_pos["binder"] = True
        df_neg = negative_dataset.df.copy()
        df_neg["binder"] = False
        df = pd.concat([df_pos, df_neg], axis=0)

        dstype = BindingDatasetsType.COMPLETE
        super().__init__(name, dstype, df)
    
    def save_df(self, fp: Path) -> None:
        self.df.to_csv(fp, sep='\t')
    
    # def record_dataset(self, fp, metadata: dict) -> None:
    #     self.save_df(fp)
        
    #     db = TinyDB(config.COMPLETE_DATASETS_DB_PATH)
    #     record = {
    #         "filepath": fp,
    #         **metadata,
    #     }
    #     db.insert(record)


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

# @dataclass(frozen=True)
class BuildCmd:
    pass

class DatasetBuilder:
    pass