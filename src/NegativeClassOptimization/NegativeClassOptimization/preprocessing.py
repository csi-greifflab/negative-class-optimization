"""
Preprocessing and transforms.
"""
import logging
from typing import List, Tuple

import farmhash
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

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
    df_train_val,
    df_test_closed,
    ag_pos: List[str],
    batch_size = 64,
    scale_onehot = True,
    df_test_open = None,
    sample_train = None,
):
    """Get train, test and openset pytorch Datasets and DataLoaders.

    Args:
        df_train_val (pd.DataFrame): dataframe in typical global format.
        ag_pos (List[str]): list of antigens labeled as positive.
        batch_size (int, optional): Defaults to 64.
        train_frac (float, optional): Defaults to 0.8.
        scale_onehot
        df_test_closed
        df_test_open
        sample_train

    Returns:
        tuple: (train_data, test_data, train_loader, test_loader).
    """

    # TODO: check references and test.

    if not scale_onehot:
        raise NotImplementedError()

    has_openset = df_test_open is not None
    if has_openset:
        if df_test_open["Slide"].isin(df_train_val["Slide"]).sum() != 0:
            raise ValueError(
                "There are slides in the test open set from the train_val set."
            )

    has_closedset = df_test_closed is not None
    if has_closedset:
        if df_test_closed["Slide"].isin(df_train_val["Slide"]).sum() != 0:
            raise ValueError(
                "There are slides in the test closed set from the train_val set."
            )

    df_train_val = remove_duplicates_for_binary(df_train_val, ag_pos)
    df_train_val = onehot_encode_df(df_train_val)
    
    if sample_train:
        df_train_val = sample_train_val(df_train_val, sample_train)

    df_test_closed = remove_duplicates_for_binary(df_test_closed, ag_pos)
    df_test_closed = onehot_encode_df(df_test_closed)

    if scale_onehot:

        arr_from_series = lambda s: np.stack(s, axis=0)

        train_onehot_stack = arr_from_series(df_train_val["Slide_onehot"])
        test_onehot_stack = arr_from_series(df_test_closed["Slide_onehot"])
        scaler = StandardScaler()
        scaler.fit(train_onehot_stack)
        df_train_val["Slide_onehot"] = scaler.transform(train_onehot_stack).tolist()
        df_test_closed["Slide_onehot"] = scaler.transform(test_onehot_stack).tolist()

    df_train_val["X"] = df_train_val["Slide_onehot"]
    df_train_val["y"] = df_train_val["binds_a_pos_ag"]
    df_test_closed["X"] = df_test_closed["Slide_onehot"]
    df_test_closed["y"] = df_test_closed["binds_a_pos_ag"]

    train_data = datasets.BinaryDataset(df_train_val)
    test_data = datasets.BinaryDataset(df_test_closed)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    if has_openset:
        df_test_open = onehot_encode_df(df_test_open)
        if scale_onehot:
            openset_onehot_stack = arr_from_series(df_test_open["Slide_onehot"])
            df_test_open["Slide_onehot"] = scaler.transform(openset_onehot_stack).tolist()
        df_test_open["X"] = df_test_open["Slide_onehot"]
        df_test_open["y"] = 0
        openset_data = datasets.BinaryDataset(df_test_open)
        openset_loader = DataLoader(openset_data, batch_size=batch_size, shuffle=False)
        return (train_data, test_data, openset_data, train_loader, test_loader, openset_loader)
    else:
        return (train_data, test_data, train_loader, test_loader)


def sample_train_val(df_train_val, sample_train, num_buckets = 16384):
    """Deterministic sampling of train_val based on hashing.

    Args:
        df_train_val (_type_): _description_
        sample_train (_type_): _description_
        num_buckets (int, optional): _description_. Defaults to 16384.

    Raises:
        OverflowError: _description_

    Returns:
        _type_: _description_
    """
    logger = logging.info()
    nrows = df_train_val.shape[0]
    try:
        if sample_train <= nrows:
            # deterministic split
            slide_hash_colname = f"Slide_farmhash_mod_{num_buckets}"
            df_train_val[slide_hash_colname] = list(map(
                    lambda s: farmhash.hash64(s) % num_buckets,
                    df_train_val["Slide"]
                ))
            sampling_frac = sample_train / nrows
            num_buckets_to_sample = np.round(sampling_frac * num_buckets)
            df_train_val = (
                    df_train_val
                    .loc[
                        df_train_val[slide_hash_colname] <= num_buckets_to_sample
                    ].copy()
                )
            logger.info(
                f"Sampling df_train_val (nrows={nrows})"
                f" and sample_train={sample_train} => "
                f"{df_train_val.shape[0]}"
                )
        else:
            raise OverflowError(f"sample_train={sample_train} > train_val nrows={nrows}.")
    except OverflowError as error:
        logger.exception(error)
        raise
    logger.warning("Resetting the index of df_train_val.")
    return df_train_val.reset_index(drop=True)  # not resetting index yields index error in Dataset and DataLoader.


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

    raise NotImplementedError("Not evaluated.")

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


def farmhash_mod_10(seq: str) -> int:
    return farmhash.hash64(seq) % 10


def openset_datasplit_from_global_stable(
    df_global: pd.DataFrame, 
    openset_antigens: List[str] = config.ANTIGENS_OPENSET,
    farmhash_mod_10_test_mask: int = config.FARMHASH_MOD_10_TEST_MASK,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """From a global dataset get a train_val, an exclusive closed test
    and an exclusive open test dataset.

    Args:
        df_global (pd.DataFrame)
        openset_antigens (List[str], optional): Defaults to config.ANTIGENS_OPENSET.

    Returns:
        df_train_val, df_test_closed_exclusive, df_test_open_exclusive
    """
    mask_ = df_global["Antigen"].isin(openset_antigens)
    df_closed = df_global.loc[~mask_].copy()
    df_open = df_global.loc[mask_].copy()
    df_test_open_exclusive = df_open.loc[~df_open["Slide"].isin(df_closed["Slide"])].copy()
        
    df_closed["Slide_farmhash_mod_10"] = list(map(
            farmhash_mod_10,
            df_closed["Slide"]
        ))
    test_mask = df_closed["Slide_farmhash_mod_10"] == farmhash_mod_10_test_mask
    df_train_val = df_closed.loc[~test_mask].copy()
    df_test_closed_exclusive = df_closed.loc[test_mask].copy()
    df_test_closed_exclusive = df_test_closed_exclusive.loc[
            ~df_test_closed_exclusive["Slide"]
            .isin(df_train_val["Slide"])
        ]
    return df_train_val, df_test_closed_exclusive, df_test_open_exclusive
