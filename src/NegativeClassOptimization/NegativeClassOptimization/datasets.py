# import abc
import warnings
from pathlib import Path
from typing import Optional, List
from itertools import combinations, product

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import NegativeClassOptimization.config as config


def trim_short_CDR3(df) -> pd.DataFrame:
    """Legacy function for removing short CDR3 - was relevant when
    working with ImmuneML.
    # TODO: Remove.
    """    
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


def generate_pairwise_dataframe(
    df_global: pd.DataFrame,
    ag1: str,
    ag2: str,
    N: Optional[int] = None,
    base_data_path: Path = config.DATA_BASE_PATH,
    seed = config.SEED,
    read_if_exists = True,
    Slide=True,
    save_datasets=True
    ) -> pd.DataFrame:
    """Generate pairwise dataframe from a global dataframe.

    Args:
        df_global (pd.DataFrame): _description_
        ag1 (str): _description_
        ag2 (str): _description_
        N (Optional[int], optional): _description_. Defaults to None.
        base_data_path (Path, optional): _description_. Defaults to config.DATA_BASE_PATH.
        seed (_type_, optional): _description_. Defaults to config.SEED.
        read_if_exists (bool, optional): _description_. Defaults to True.
        Slide (bool, optional): _description_. Defaults to True.
        save_datasets (bool, optional): _description_. Defaults to True.

    Returns:
        pd.DataFrame: _description_
    """    

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
    """Generate a 1_vs_all dataframe.

    Args:
        df_global (pd.DataFrame): _description_
        ag (str): _description_
        base_data_path (Path, optional): _description_. Defaults to config.DATA_BASE_PATH.
        seed (_type_, optional): _description_. Defaults to config.SEED.

    Returns:
        pd.DataFrame: _description_
    """    
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
        self.process_y_tensor = lambda t: t.reshape((1)).type(torch.float)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        return (
            torch.tensor(self.df.loc[idx, "X"]).reshape(
                (1, -1)).type(torch.float),
            self.process_y_tensor(torch.tensor(self.df.loc[idx, "y"])),
        )


class MulticlassDataset(BinaryDataset):
    """Pytorch dataset for modelling antigen binding multiclass classifiers.
    """

    def __init__(self, df):
        super().__init__(df)
        self.process_y_tensor = lambda t: t.reshape(-1).type(torch.long)


def construct_dataset_atoms(
    antigens: List[str]
    ) -> List[List[str]]:
    atoms = []
    for i in range(len(antigens)):
        size = i + 1
        atoms += sorted(combinations(antigens, r=size))
    return atoms


def construct_dataset_atom_combinations(
    antigens: List[str] = config.ANTIGENS_CLOSEDSET,
    atoms: Optional[List[List[str]]] = None
    ) -> List[List[str]]:
    """Construct ag set pairs to be used in defining datasets.

    Args:
        antigens (List[str]): antigen support to use for building. Defaults to config.ANTIGENS_CLOSEDSET.
        atoms (List[List[str]], optional): atoms for building pairs. Defaults to None.

    Returns:
        _type_: _description_
    """    
    
    if atoms is None:
        atoms = construct_dataset_atoms(antigens)
    valid_combinations = []
    
    for ag_pos_atom, ag_neg_atom in product(atoms, atoms):
        if len(set(ag_pos_atom).intersection(set(ag_neg_atom))) > 0:
            continue
        if len(ag_pos_atom) + len(ag_neg_atom) > len(antigens):
            continue

        valid_combinations.append((list(ag_pos_atom), list(ag_neg_atom)))

    return valid_combinations
