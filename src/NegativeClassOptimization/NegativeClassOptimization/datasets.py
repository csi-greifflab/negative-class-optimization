# import abc
import json
import os
import warnings
from dataclasses import dataclass
from enum import Enum
from itertools import combinations, product
from pathlib import Path
from typing import List, Optional

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
    small_lengths = cdr3_len_counts.loc[cdr3_len_counts < LEN_TH].index.to_list()
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
    seed=config.SEED,
    read_if_exists=True,
    Slide=True,
    save_datasets=True,
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
        filepath = (
            base_data_path / "pairwise_wo_dupl" / f"pairwise_dataset_{ag1}_{ag2}.tsv"
        )
    elif N:
        filepath = base_data_path / "pairwise" / f"pairwise_dataset_{ag1}_{ag2}_{N}.tsv"
    else:
        filepath = base_data_path / "pairwise" / f"pairwise_dataset_{ag1}_{ag2}.tsv"

    if read_if_exists and filepath.exists():
        df = pd.read_csv(filepath, sep="\t")
        return df

    df_ag1 = df_global.loc[df_global["Antigen"] == ag1].copy()
    df_ag2 = df_global.loc[df_global["Antigen"] == ag2].copy()

    if Slide:
        df_ag1.drop_duplicates(subset="Slide", keep="last", inplace=True)
        df_ag2.drop_duplicates(subset="Slide", keep="last", inplace=True)
        inters = set(df_ag1.Slide) & set(df_ag2.Slide)
        df_ag2 = df_ag2[df_ag2["Slide"].isin(inters) == False]
        df_ag1 = df_ag1[df_ag1["Slide"].isin(inters) == False]
    else:
        inters = set(df_ag1.CDR3) & set(df_ag2.CDR3)
        df_ag2 = df_ag2[df_ag2["CDR3"].isin(inters) == False]
        df_ag1 = df_ag1[df_ag1["CDR3"].isin(inters) == False]

    if N:
        np.random.seed(seed)
        df_ag1 = df_ag1.sample(N // 2)
        df_ag2 = df_ag2.sample(N // 2)

    df_ag1["binder"] = True
    df_ag2["binder"] = False

    df = pd.concat([df_ag1, df_ag2], axis=0)

    df = df.sample(frac=1, random_state=config.SEED)
    if save_datasets:
        df.to_csv(filepath, sep="\t")

    return df


def generate_1_vs_all_dataset(
    df_global: pd.DataFrame,
    ag: str,
    base_data_path: Path = config.DATA_BASE_PATH,
    seed=config.SEED,
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
    df.to_csv(filepath, sep="\t")

    return df


class BinaryDataset(Dataset):
    """Pytorch dataset for modelling antigen binding binary classifiers."""

    def __init__(self, df):
        self.df = df
        self.process_y_tensor = lambda t: t.reshape((1)).type(torch.float)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        return (
            torch.tensor(self.df.loc[idx, "X"]).reshape((1, -1)).type(torch.float),
            self.process_y_tensor(torch.tensor(self.df.loc[idx, "y"])),
        )

    def _get_indexes(self):
        return self.df.index.values


class MulticlassDataset(BinaryDataset):
    """Pytorch dataset for modelling antigen binding multiclass classifiers."""

    def __init__(self, df):
        super().__init__(df)
        self.process_y_tensor = lambda t: t.reshape(-1).type(torch.long)


class MultilabelDataset(BinaryDataset):
    def __init__(self, df):
        super().__init__(df)
        self.process_y_tensor = lambda t: t.reshape((1, -1)).type(torch.long)


def construct_dataset_atoms(antigens: List[str]) -> List[List[str]]:
    atoms = []
    for i in range(len(antigens)):
        size = i + 1
        atoms += sorted(combinations(antigens, r=size))
    return atoms


def construct_dataset_atom_combinations(
    antigens: List[str] = config.ANTIGENS_CLOSEDSET,
    atoms: Optional[List[List[str]]] = None,
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

    @staticmethod
    def split_open_closed(antigens: List[str], num_open: int, seed: int):
        shuffled = utils.shuffle_antigens(antigens, seed=seed)
        num_closed = len(antigens) - num_open
        return shuffled[:num_closed], shuffled[num_closed:]

    @staticmethod
    def convert_to_wide_format(df, antigens: Optional[List[str]] = None):
        """Converts Absolut Dataset 3 format to wide format."""
        if antigens is None:
            antigens = AbsolutDataset3.get_antigens()
        df_wide = pd.DataFrame.from_records(
            data=df["binding_profile"]
            .apply(lambda x: {antigens[i]: int(x[i]) for i in range(len(antigens))})  # type: ignore
            .to_list(),
        )
        assert all(df_wide.sum(axis=1) == df["num_binding_ags"])

        df_wide.index = df["Slide"]
        return df_wide

    @staticmethod
    def get_antigens(path=config.DATA_ABSOLUT_DATASET3_AGLIST):
        with open(path, "r") as f:
            antigens = f.read().splitlines()
        return antigens

    @staticmethod
    def get_closed_antigens(
        processed_path=config.DATA_ABSOLUT_PROCESSED_MULTICLASS_DIR,
    ):
        dfs = utils.load_processed_dataframes(processed_path)
        df_train_val = dfs["train_val"]
        return sorted(set(df_train_val["Antigen"]))

    @staticmethod
    def get_binding_matrix(path=config.DATA_ABSOLUT_DATASET3_BINDINGMTX):
        df = pd.read_csv(path, sep="\t", header=None)
        df.columns = ["Slide", "num_binding_ags", "binding_profile"]
        return df


class ClassificationTaskType(Enum):
    ONE_VS_ONE = 1
    ONE_VS_NINE = 2
    HIGH_VS_95LOW = 3
    HIGH_VS_LOOSER = 4

    @classmethod  # https://stackoverflow.com/questions/41359490/python-static-method-inside-private-inner-enum-class
    def from_str(cls, s: str) -> "ClassificationTaskType":
        if s == "1v1":
            return ClassificationTaskType.ONE_VS_ONE
        elif s == "1v9":
            return ClassificationTaskType.ONE_VS_NINE
        elif s == "high_vs_95low":
            return ClassificationTaskType.HIGH_VS_95LOW
        elif s == "high_vs_looser":
            return ClassificationTaskType.HIGH_VS_LOOSER
        else:
            raise ValueError(f"Unrecognized ClassificationTaskType: {s}")

    def to_str(self) -> str:
        if self == ClassificationTaskType.ONE_VS_ONE:
            return "1v1"
        elif self == ClassificationTaskType.ONE_VS_NINE:
            return "1v9"
        elif self == ClassificationTaskType.HIGH_VS_95LOW:
            return "high_vs_95low"
        elif self == ClassificationTaskType.HIGH_VS_LOOSER:
            return "high_vs_looser"
        else:
            raise ValueError(f"Unrecognized ClassificationTaskType: {self}")


class ClassificationTask:
    """Task for modelling antigen binding classifiers.

    ## Example
    # t = ClassificationTask(
    #     task_type=ClassificationTaskType.ONE_VS_ONE,
    #     ag_pos="A",
    #     ag_neg="B",
    #     seed_id=0,
    #     split_id=0,
    # )

    # t_inv = ClassificationTask.init_from_str(str(t))
    # str(t_inv) == str(t)
    """

    def __init__(
        self,
        task_type: ClassificationTaskType,
        ag_pos: str,
        seed_id: int,
        split_id: int,
        ag_neg: str = "auto",
    ):
        # Validate task type
        if task_type == ClassificationTaskType.ONE_VS_ONE and ag_neg == "auto":
            raise ValueError("ag_neg must be specified for ONE_VS_ONE task type")
        elif task_type != ClassificationTaskType.ONE_VS_ONE and ag_neg != "auto":
            raise ValueError(
                "ag_neg must be set to 'auto' / not set for non-ONE_VS_ONE task type"
            )

        # Validate antigens
        ClassificationTask.validate_antigen(ag_pos)
        if ag_neg != "auto":
            ClassificationTask.validate_antigen(ag_neg)

        self.task_type = task_type
        self.ag_pos = ag_pos
        self.seed_id = seed_id
        self.split_id = split_id
        self.ag_neg = ag_neg

        self.basepath = None
        self.model = None
        self.test_dataset = None

    @staticmethod
    def validate_antigen(ag: str):
        """
        Validates an antigen name.
        """
        # if "_" in ag:
        #     raise ValueError(f"Invalid antigen name: {ag}")
        # elif len(ag) < 4 or len(ag) > 6:
        #     raise ValueError(f"Invalid antigen name: {ag}")
        pass

    def get_nco_ag_pos(self):
        """
        Returns the antigen name for NCO.
        """
        if self.task_type == ClassificationTaskType.ONE_VS_ONE:
            return self.ag_pos
        elif self.task_type == ClassificationTaskType.ONE_VS_NINE:
            return self.ag_pos
        elif self.task_type in [
            ClassificationTaskType.HIGH_VS_95LOW,
            ClassificationTaskType.HIGH_VS_LOOSER,
        ]:
            return f"{self.ag_pos}_high"
        else:
            raise ValueError(f"Invalid task type: {self.task_type}")

    def get_nco_ag_neg(self):
        """
        Returns the antigen name for NCO.
        """
        if self.task_type == ClassificationTaskType.ONE_VS_ONE:
            return self.ag_neg
        elif self.task_type == ClassificationTaskType.ONE_VS_NINE:
            return "9"
        elif self.task_type == ClassificationTaskType.HIGH_VS_95LOW:
            return f"{self.ag_pos}_95low"
        elif self.task_type == ClassificationTaskType.HIGH_VS_LOOSER:
            return f"{self.ag_pos}_looser"
        else:
            raise ValueError(f"Invalid task type: {self.task_type}")

    def __str__(self):
        return f"{self.task_type.name}__{self.ag_pos}__{self.ag_neg}__{self.seed_id}__{self.split_id}"

    def __repr__(self):
        return str(self)

    @staticmethod
    def init_from_str(task_str: str) -> "ClassificationTask":
        """
        Returns a Task object from a string.
        """
        task_type_name, ag_pos, ag_neg, seed_id, split_id = task_str.split("__")
        return ClassificationTask(
            task_type=ClassificationTaskType[task_type_name],
            ag_pos=ag_pos,
            ag_neg=ag_neg,
            seed_id=int(seed_id),
            split_id=int(split_id),
        )

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, __value: object) -> bool:
        return str(self) == str(__value)


class FrozenMiniAbsolutMLLoader:
    """
    Loads the frozen MiniAbsolut dataset.

    ## Example use case
    # loader = FrozenMiniAbsolutMLLoader(data_dir=Path("../data/Frozen_MiniAbsolut_ML/"))
    # task = ClassificationTask(
    #     task_type=ClassificationTaskType.ONE_VS_ONE,
    #     ag_pos="1ADQ",
    #     ag_neg="3VRL",
    #     seed_id=0,
    #     split_id=0,
    # )
    # loader.load(task)
    # print(task.model, task.test_dataset)
    """

    def __init__(
        self,
        data_dir: Path,
    ):
        self.data_dir = data_dir

    def load(
        self,
        task: ClassificationTask,
        load_model=True,
        load_test_dataset=True,
        load_train_dataset=False,
        attributions_toload=None,
        attribution_records_toload="attribution_records.json",
        load_zscores=False,
    ):
        """
        Loads the frozen MiniAbsolut dataset.
        """

        basepath = self.infer_task_basepath(task)
        task.basepath = basepath

        if load_model:
            model_path = basepath / "swa_model/data/model.pth"
            model_path_alt = (
                basepath / "swa_model/swa_model.pt"
            )  # Path for models on exp data.
            if model_path.exists():
                task.model = torch.load(model_path)
            elif model_path_alt.exists():
                # Path for models on exp data.
                task.state_dict = torch.load(model_path_alt)  # type: ignore
                task.model = None
            else:
                raise ValueError(f"Model not found at {model_path}")
        if load_test_dataset:
            try:
                tsv_name_split = list(basepath.glob("*tsv"))[0].name.split("_")
                if len(tsv_name_split) <= 2:
                    hash_val = None
                else:
                    hash_val = list(basepath.glob("*tsv"))[0].name.split("_")[0]
            except:
                print(basepath, list(basepath.glob("*tsv")))
                raise
            if hash_val is None:
                test_dataset_path = basepath / "test_dataset.tsv"
            else:
                test_dataset_path = basepath / f"{hash_val}_test_dataset.tsv"
            task.test_dataset = pd.read_csv(test_dataset_path, sep="\t")  # type: ignore
        if load_train_dataset:
            try:
                hash_val = list(basepath.glob("*tsv"))[0].name.split("_")[0]
            except:
                print(basepath, list(basepath.glob("*tsv")))
                raise
            train_dataset_path = basepath / f"{hash_val}_train_dataset.tsv"
            task.train_dataset = pd.read_csv(train_dataset_path, sep="\t")  # type: ignore
        if attributions_toload is not None:
            attr_dir = basepath / "attributions" / attributions_toload
            attr_records = attr_dir / attribution_records_toload
            assert (
                attr_records.exists()
            ), f"Attribution records not found at {attr_records}"
            with open(attr_records, "r") as f:
                attr_records = json.load(f)
            task.attributions = attr_records  # type: ignore
        if load_zscores:
            zscores_path = basepath / "attributions/z_records.json"
            assert zscores_path.exists(), f"Zscores not found at {zscores_path}"
            with open(zscores_path, "r") as f:
                z_records = json.load(f)
            task.z_records = z_records  # type: ignore

        return task

    def infer_task_basepath(self, task):
        task_type_str = task.task_type.to_str()

        # Infer positive and negative antigens
        ag_pos = task.get_nco_ag_pos()
        ag_neg = task.get_nco_ag_neg()

        task.basepath = (  # type: ignore
            self.data_dir
            / task_type_str
            / f"seed_{task.seed_id}"
            / f"split_{task.split_id}"
            / f"{ag_pos}__vs__{ag_neg}"
        )
        return task.basepath

    @staticmethod
    def generate_seed_split_ids(standard_split_only=False):
        """
        Generate valid seed_id and split_id combinations,
        used in FrozenMiniAbsolutML dataset.
        """
        if standard_split_only:
            return [(0, 42)]
        else:
            seed_split_ids = []
            for seed in [0, 1, 2, 3]:
                split_id_default = 42
                seed_split_ids.append((seed, split_id_default))
            for split_id in [0, 1, 2, 3, 4]:
                seed_id_default = 0
                seed_split_ids.append((seed_id_default, split_id))
            return seed_split_ids
