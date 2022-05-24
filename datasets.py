# import abc
from enum import Enum
from typing import Optional, List
import pandas as pd


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


class RandomBindingDataset:
    pass
    # raise NotImplementedError


class PositiveBindingDataset(BaseBindingDataset):

    def __init__(
        self,
        name: str,
        df: pd.DataFrame,
        antigen: str) -> None:

        dstype = BindingDatasetsType.POSITIVE
        self.super().__init__(name, dstype, df)

        self.antigen = antigen
        
        if not self.validate_df():
            raise ValueError("Failed to init PositiveBindingDataset")


    def validate_df(self):
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

        dstype = BindingDatasetsType.NEGATIVE
        self.super().__init__(name, dstype, df)


class CompleteBindingDataset(BaseBindingDataset):

    def __init__(
        self,
        name: str,
        positive_dataset: PositiveBindingDataset,
        negative_dataset: NegativeBindingDataset,
        ) -> None:
        
        self.positive_dataset = positive_dataset
        self.negative_dataset = negative_dataset

        dstype = BindingDatasetsType.COMPLETE
        df = pd.concat([positive_dataset.df, negative_dataset.df], axis=0)
        self.super().__init__(name, dstype, df)