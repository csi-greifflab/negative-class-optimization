# import abc
from enum import Enum
from pathlib import Path
from typing import Optional, List
import pandas as pd
from tinydb import TinyDB, Query
import utils
import config


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

    DB_PATH = config.COMPLETE_DATASETS_DB_PATH

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
    
    def record_dataset(self, fp, metadata: dict) -> None:
        self.save_df(fp)
        
        db = TinyDB(config.COMPLETE_DATASETS_DB_PATH)
        record = {
            "filepath": fp,
            **metadata,
        }
        db.insert(record)