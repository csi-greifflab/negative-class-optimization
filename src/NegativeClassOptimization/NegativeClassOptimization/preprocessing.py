from typing import List
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
import NegativeClassOptimization.config as config


def get_one_hot_aa_encoder(aminoacids: List[str] = config.SLIDE_AMINOACIDS):
    """Get a OneHotEncoder fitted to the aminoacids characters used in the `Slide`s of `700k dataset`.

    Args:
        aminoacids (List[str], optional): default list of amino acids used in `Slide`. Defaults to config.SLIDE_AMINOACIDS.

    Returns:
        OneHotEncoder: fitted, ready to transform sequences.
    """    
    aa_as_array = np.array(aminoacids)
    encoder = OneHotEncoder(sparse=False).fit(aa_as_array.reshape(-1, 1))
    return encoder


def onehot_encode(
    string: str, 
    encoder = get_one_hot_aa_encoder()
    ) -> np.array:
    """Encode a string to onehot numpy array with shape (-1).

    Args:
        string (str)
        encoder (_type_, optional): fitted encoder. Defaults to get_one_hot_aa_encoder().

    Returns:
        np.array
    """    
    string_as_array = np.array(list(string))
    string_as_onehot = encoder.transform(string_as_array.reshape(-1, 1))
    return string_as_onehot.reshape(-1)


def onehot_encode_df(
    df, 
    encoder = get_one_hot_aa_encoder()
    )-> np.array:
    """Encode a dataframe with `Slide` as onehot

    Args:
        df: must have `Slide` column.
        encoder: see `onehot_encode`.
        scale (bool): scale column-wise?

    Returns:
        np.array
    """      
    df = df.copy()
    df["Slide_onehot"] = df["Slide"].apply(lambda s: onehot_encode(s, encoder=encoder))
    return df


def get_antigen_label_encoder() -> LabelEncoder:
    label_encoder = LabelEncoder()
    label_encoder.fit(config.ANTIGENS_CLOSEDSET)
    return label_encoder


def remove_duplicates_for_binary(df: pd.DataFrame, ag_pos: List[str]) -> pd.DataFrame:
    """Remove `Slide` duplicates for datasets for binary classifiers.
    An important step in preparing data training and evaluation. 
    This function handles this for the binary problems: NDB1, NDBK, NDM1,
    NDMK).

    Args:
        df (pd.DataFrame): typical dataframe used in the project
        pos_ag (str): the antigen assuming the positive dataset role

    Returns:
        pd.DataFrame: df with 2 columns suitable for modelling: `Slide` and `binds_a_pos_ag`.
    """

    def infer_antigen_from_duplicate_list(
        antigens: List[str], 
        pos_antigens: List[str],
        ):
        for pos_ag in pos_antigens:
            if pos_ag in antigens:
                return 1    
        return 0

    df = df.groupby("Slide").apply(
        lambda df_: infer_antigen_from_duplicate_list(
            df_["Antigen"].unique().tolist(), pos_antigens=ag_pos
        )
    )
    df = pd.DataFrame(data=df, columns=["binds_a_pos_ag"])
    df = df.reset_index()
    return df


def remove_duplicates_for_multiclass(df: pd.DataFrame) -> pd.DataFrame:
    """Remove all Slides that are duplicated. A temporary solution for
    dealing with duplicates in `Slide`.

    Args:
        df (pd.DataFrame)

    Returns:
        pd.DataFrame
    """    
    df = df.loc[~df["Slide"].duplicated(keep=False)].copy()
    return df
