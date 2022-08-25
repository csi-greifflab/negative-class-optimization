from typing import List
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from torch.utils.data import Dataset, DataLoader

import NegativeClassOptimization.config as config
import NegativeClassOptimization.datasets as datasets


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
    dealing with duplicates in `Slide`. A better solution is multilabel
    classification.

    Args:
        df (pd.DataFrame)

    Returns:
        pd.DataFrame
    """    
    df = df.loc[~df["Slide"].duplicated(keep=False)].copy()
    return df


def preprocess_data_for_pytorch_binary(
    df,
    ag_pos: List[str],
    batch_size = 64,
    train_frac = 0.8,
    scale_onehot = True,
    df_openset = None,
):
    """Get train, test and openset pytorch Datasets and DataLoaders.

    Args:
        df (pd.DataFrame): dataframe in typical global format.
        ag_pos (List[str]): list of antigens labeled as positive.
        batch_size (int, optional): Defaults to 64.
        train_frac (float, optional): Defaults to 0.8.
        scale_onehot
        df_openset

    Returns:
        tuple: (train_data, test_data, train_loader, test_loader).
    """

    if not scale_onehot:
        raise NotImplementedError("Bug expected.")

    has_openset = df_openset is not None
    if has_openset:
        if df_openset["Slide"].isin(df["Slide"]).sum() != 0:
            raise ValueError(
                "There are slides in the open set from the closed set."
            )

    df = remove_duplicates_for_binary(df, ag_pos)
    df = onehot_encode_df(df)

    df = df.sample(frac=1).reset_index(drop=True)  # shuffle

    split_idx = int(df.shape[0] * train_frac)
    df_train = df.loc[:split_idx].copy().reset_index(drop=True)
    df_test = df.loc[split_idx:].copy().reset_index(drop=True)

    if scale_onehot:

        arr_from_series = lambda s: np.stack(s, axis=0)

        train_onehot_stack = arr_from_series(df_train["Slide_onehot"])
        test_onehot_stack = arr_from_series(df_test["Slide_onehot"])
        scaler = StandardScaler()
        scaler.fit(train_onehot_stack)
        df_train["Slide_onehot"] = scaler.transform(train_onehot_stack).tolist()
        df_test["Slide_onehot"] = scaler.transform(test_onehot_stack).tolist()

    df_train["X"] = df_train["Slide_onehot"]
    df_train["y"] = df_train["binds_a_pos_ag"]
    df_test["X"] = df_test["Slide_onehot"]
    df_test["y"] = df_test["binds_a_pos_ag"]

    train_data = datasets.BinaryDataset(df_train)
    test_data = datasets.BinaryDataset(df_test)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    if has_openset:
        df_openset = onehot_encode_df(df_openset)
        if scale_onehot:
            openset_onehot_stack = arr_from_series(df_openset["Slide_onehot"])
            df_openset["Slide_onehot"] = scaler.transform(openset_onehot_stack).tolist()
        df_openset["X"] = df_openset["Slide_onehot"]
        df_openset["y"] = 0
        openset_data = datasets.BinaryDataset(df_openset)
        openset_loader = DataLoader(openset_data, batch_size=batch_size, shuffle=False)
        return (train_data, test_data, openset_data, train_loader, test_loader, openset_loader)
    else:
        return (train_data, test_data, train_loader, test_loader)


def preprocess_data_for_pytorch_multiclass(
    df,
    batch_size = 64,
    train_frac = 0.8,
    scale_onehot = True,
):
    """Get train and test pytorch Datasets and DataLoaders.

    Args:
        df (pd.DataFrame): dataframe in typical global format.add()
        ag_pos (List[str]): list of antigens labeled as positive.add()
        batch_size (int, optional): Defaults to 64.
        train_frac (float, optional): Defaults to 0.8.

    Returns:
        tuple: (train_data, test_data, train_loader, test_loader).
    """

    df = remove_duplicates_for_multiclass(df)
    df = onehot_encode_df(df)

    df = df.sample(frac=1).reset_index(drop=True)  # shuffle

    split_idx = int(df.shape[0] * train_frac)
    df_train = df.loc[:split_idx].copy().reset_index(drop=True)
    df_test = df.loc[split_idx:].copy().reset_index(drop=True)

    if scale_onehot:
        train_onehot_stack = np.stack(df_train["Slide_onehot"], axis=0)
        test_onehot_stack = np.stack(df_test["Slide_onehot"], axis=0)
        scaler = StandardScaler()
        scaler.fit(train_onehot_stack)
        df_train["Slide_onehot"] = scaler.transform(train_onehot_stack).tolist()
        df_test["Slide_onehot"] = scaler.transform(test_onehot_stack).tolist()

    label_encoder = get_antigen_label_encoder()
    df_train["X"] = df_train["Slide_onehot"]
    df_train["y"] = label_encoder.transform(df_train["Antigen"])
    df_test["X"] = df_test["Slide_onehot"]
    df_test["y"] = label_encoder.transform(df_test["Antigen"])

    train_data = datasets.MulticlassDataset(df_train)
    test_data = datasets.MulticlassDataset(df_test)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return (train_data, test_data, train_loader, test_loader)
