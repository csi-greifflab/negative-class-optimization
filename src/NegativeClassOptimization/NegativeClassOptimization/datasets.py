# import abc
from dataclasses import dataclass
import json
import os
import warnings
from pathlib import Path
from typing import Optional, List
from itertools import combinations, product

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import NegativeClassOptimization.config as config
from NegativeClassOptimization import utils


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

    def _get_indexes(self):
        return self.df.index.values


class MulticlassDataset(BinaryDataset):
    """Pytorch dataset for modelling antigen binding multiclass classifiers.
    """

    def __init__(self, df):
        super().__init__(df)
        self.process_y_tensor = lambda t: t.reshape(-1).type(torch.long)


class MultilabelDataset(BinaryDataset):

    def __init__(self, df):
        super().__init__(df)
        self.process_y_tensor = lambda t: t.reshape((1, -1)).type(torch.long)


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


class AbsolutDataset3:
    """Dataset for modelling antigen binding classifiers."""

    def __init__(self):
        self.df = AbsolutDataset3.get_binding_matrix()
        self.antigens = AbsolutDataset3.get_antigens()
        self.df_wide = AbsolutDataset3.convert_to_wide_format(self.df, self.antigens)

        # self._build_splits()

    # def _build_splits(self):
    #     self.splits_open_closed = {}
    #     for seed in range(1, 5):
    #         self.splits_open_closed[seed] = AbsolutDataset3.split_open_closed(
    #             self.antigens, 
    #             num_open=40, 
    #             seed=seed
    #             )
    
    # def get_closed_open_antigen_split(self, seed=1):
        # return self.splits_open_closed[seed]

    @staticmethod
    def split_open_closed(antigens: List[str], num_open: int, seed: int):
        shuffled = utils.shuffle_antigens(antigens, seed=seed)
        num_closed = len(antigens) - num_open
        return shuffled[:num_closed], shuffled[num_closed:]

    @staticmethod
    def convert_to_wide_format(df, antigens: Optional[List[str]] = None):
        """Converts Absolut Dataset 3 format to wide format.
        """    
        if antigens is None:
            antigens = AbsolutDataset3.get_antigens()
        df_wide = pd.DataFrame.from_records(
            data=df["binding_profile"].apply(lambda x: {antigens[i]: int(x[i]) for i in range(len(antigens))}).to_list(),
        )
        assert all(df_wide.sum(axis=1) == df["num_binding_ags"])

        df_wide.index = df["Slide"]
        return df_wide

    @staticmethod
    def get_antigens(path = config.DATA_ABSOLUT_DATASET3_AGLIST):
        with open(path, "r") as f:
            antigens = f.read().splitlines()
        return antigens
    
    @staticmethod
    def get_closed_antigens(processed_path = config.DATA_ABSOLUT_PROCESSED_MULTICLASS_DIR):
        dfs = utils.load_processed_dataframes(processed_path)
        df_train_val = dfs["train_val"]
        return sorted(set(df_train_val["Antigen"]))

    @staticmethod
    def get_binding_matrix(path = config.DATA_ABSOLUT_DATASET3_BINDINGMTX):
        df = pd.read_csv(path, sep='\t', header=None)
        df.columns = ["Slide", "num_binding_ags", "binding_profile"]
        return df


### Deprecated
# @dataclass
# class Task:
#     """Task for modelling antigen binding classifiers.
#     Fetches data from MLFlow backend (on local system) 
#     and stores it in the local filesystem for sharing.

#     This class shares a lot with `ClassificationTaskType`.
#     It is not a good design. However, this class is used
#     to interact with mlflow, while the other represents
#     the more abstract task, that doesn't depend on mlflow.

#     This class can also probably be removed.

#     Returns:
#         _type_: _description_
#     """
#     ag_pos: str
#     ag_neg: str
#     shuffle_antigen_labels: str = "False"
#     run_name: Optional[str] = None

#     def __post_init__(self):
#         if isinstance(self.shuffle_antigen_labels, bool):
#             self.shuffle_antigen_labels = str(self.shuffle_antigen_labels)

    # def get_exp_and_run_ids(self) -> str:
    #     api = utils.MLFlowTaskAPI()
    #     exp_id, run_id = api.get_experiment_and_run(self.__dict__, run_name=self.run_name)
    #     return exp_id, run_id
    
    # def get_paths_to_data(self) -> List[Path]:
    #     exp_id, run_id = self.get_exp_and_run_ids()

    #     artifacts_path, dataset_hash, df_train_path, df_test_path, metrics_path, model_path, swa_model_path = Task.compile_paths(exp_id, run_id)


    #     self.artifacts_path = artifacts_path
    #     self.dataset_hash = dataset_hash
    #     self.df_train_path = df_train_path
    #     self.df_test_path = df_test_path
    #     self.metrics_path = metrics_path
    #     self.model_path = model_path
    #     self.swa_model_path = swa_model_path
    
    #     self.exp_id = exp_id
    #     self.run_id = run_id

    
    # def copy_files_to_dir(self, dest_dir: Path):
    #     self.get_paths_to_data()

    #     dest_dir = Path(dest_dir) / f"{self.ag_pos}__vs__{self.ag_neg}"
    #     dest_dir.mkdir(exist_ok=True, parents=True)
        
    #     with open(dest_dir / "task.json", "w") as f:
    #         d = self.__dict__.copy()
            
    #         # Make Paths serializable
    #         for k, v in d.items():
    #             if isinstance(v, Path):
    #                 d[k] = str(v)
            
    #         json.dump(d, f, indent=4)

    #     list_of_paths = [self.df_train_path, self.df_test_path, self.metrics_path, self.model_path, self.swa_model_path]

    #     self.copy_pathlist_to_dest(dest_dir, list_of_paths)


    # @staticmethod
    # def compile_paths(exp_id, run_id) -> List[Path]:
    #     artifacts_path = config.DATA_BASE_PATH / Path(f"nco_mlflow_runs/ftp/artifacts_store/{exp_id}/{run_id}/artifacts/")

    #     # This is a hack to correct for a bug in folder/file namiang
    #     glob_list = list((artifacts_path / "dataset/train_dataset.tsv").glob("*tsv"))
    #     dataset_hash = glob_list[0].stem.split("_")[0]
    #     df_train_path = artifacts_path / f"dataset/train_dataset.tsv/{dataset_hash}_train_dataset.tsv"
    #     df_test_path = artifacts_path / f"dataset/test_dataset.tsv/{dataset_hash}_test_dataset.tsv"

    #     metrics_path = artifacts_path / "eval_metrics.json"
    #     model_path = artifacts_path / f"models/trained_model"
    #     swa_model_path = artifacts_path / f"models/swa_model"
    #     return [artifacts_path, dataset_hash, df_train_path, df_test_path, metrics_path, model_path, swa_model_path]


    # @staticmethod
    # def copy_pathlist_to_dest(dest_dir: Path, list_of_paths: List[Path]):
    #     for path in list_of_paths:
    #         dest_path = dest_dir / path.name
    #         if dest_path.exists():
    #             warnings.warn(f"File {dest_path} already exists. Skipping copy.")
    #         else:
    #             os.system(f"cp -r {path} {dest_path}")