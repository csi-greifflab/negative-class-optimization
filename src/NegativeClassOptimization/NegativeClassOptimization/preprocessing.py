"""
Preprocessing and transforms.
"""
import logging
from typing import List, Optional, Tuple, Union

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

        train_onehot_stack = arr_from_list_series(df_train_val["Slide_onehot"])
        test_onehot_stack = arr_from_list_series(df_test_closed["Slide_onehot"])

        scaler = StandardScaler()
        scaler.fit(train_onehot_stack)
        df_train_val["Slide_onehot"] = scaler.transform(train_onehot_stack).tolist()
        df_test_closed["Slide_onehot"] = scaler.transform(test_onehot_stack).tolist()
    else:
        scaler = None

    df_train_val["X"] = df_train_val["Slide_onehot"]
    df_train_val["y"] = df_train_val["binds_a_pos_ag"]
    df_test_closed["X"] = df_test_closed["Slide_onehot"]
    df_test_closed["y"] = df_test_closed["binds_a_pos_ag"]

    train_data = datasets.BinaryDataset(df_train_val)
    test_data = datasets.BinaryDataset(df_test_closed)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    if has_openset:
        
        openset_data, openset_loader = construct_open_dataset_loader(
            df_test_open, 
            batch_size, 
            scaler=scaler,
            )

        return (train_data, test_data, openset_data, train_loader, test_loader, openset_loader)
    else:
        return (train_data, test_data, train_loader, test_loader)


def construct_open_dataset_loader(
    df_test_open, 
    batch_size, 
    scaler=None
    ):
    df_test_open = df_test_open.reset_index(drop=True)
    df_test_open = onehot_encode_df(df_test_open)
    if scaler is not None:
        openset_onehot_stack = arr_from_list_series(df_test_open["Slide_onehot"])
        df_test_open["Slide_onehot"] = scaler.transform(openset_onehot_stack).tolist()
    df_test_open["X"] = df_test_open["Slide_onehot"]
    df_test_open["y"] = 0
    openset_data = datasets.BinaryDataset(df_test_open)
    openset_loader = DataLoader(openset_data, batch_size=batch_size, shuffle=True)
    return openset_data, openset_loader


def arr_from_list_series(s: pd.Series): 
    """Convert to 2D array from series of lists."""    
    return np.stack(s, axis=0)


def sample_df_deterministically(df, sample, num_buckets = 16384) -> pd.DataFrame:
    """Wrapper to generalize deterministic sampling of a dataframe.

    Args:
        df (_type_): dataframe to sample from.
        sample (_type_): number of samples to take.
        num_buckets (int, optional): Defaults to 16384.

    Returns:
        pd.DataFrame: deterministically sampled dataframe.
    """    
    assert "Slide" in df.columns, "df must have a column named `Slide`."
    return sample_train_val(
        df_train_val = df, 
        sample_train = sample, 
        num_buckets = num_buckets,
        )


def sample_train_val(df_train_val, sample_train, num_buckets = 2*16384):
    """Deterministic sampling of train_val based on hashing.

    Args:
        df_train_val (_type_): _description_
        sample_train (_type_): _description_
        num_buckets (int, optional): _description_. Defaults to 2*16384.

    Raises:
        OverflowError: _description_

    Returns:
        _type_: _description_
    """
    logger = logging.getLogger()
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
                f"Sampling df_df (nrows={nrows})"
                f" and sample={sample_train} => "
                f"{df_train_val.shape[0]}"
                )
        else:
            print(f"{df_train_val.shape=} | {sample_train=}")
            raise OverflowError()
    except OverflowError as error:
        logger.exception(error)
        raise
    logger.warning("Resetting the index of df_train_val.")
    return df_train_val.reset_index(drop=True)  # not resetting index can yield index error in Dataset and DataLoader.


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


def preprocess_df_for_multiclass(
    df,
    ags: List[str],
    scaler = None,
    encoder = None,
    sample = None,
    sample_per_ag = None,
    sample_per_ag_seed = config.SEED,
    ):
    
    df = df.loc[df["Antigen"].isin(ags)].copy()

    df = remove_duplicates_for_multiclass(df)    
    
    if sample_per_ag is not None:
        try:
            df = df.groupby("Antigen").sample(sample_per_ag, random_state=sample_per_ag_seed)
        except ValueError as e:
            print(e)
            print(df["Antigen"].value_counts())
            raise
    elif sample is not None:
        df = sample_train_val(df, sample)

    df = onehot_encode_df(df)

    arr = arr_from_list_series(df["Slide_onehot"])
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(arr)
    df["X"] = scaler.transform(arr).tolist()

    if encoder is None:
        antigens = df["Antigen"].unique().tolist()
        encoder = LabelEncoder().fit(antigens)

    df["y"] = encoder.transform(df["Antigen"])
    df = df[["X", "y"]]
    return df, scaler, encoder


def farmhash_mod_10(seq: str) -> int:
    return farmhash.hash64(seq) % 10


def openset_datasplit_from_global_stable(
    df_global: pd.DataFrame,
    openset_antigens: List[str] = config.ANTIGENS_OPENSET,
    farmhash_mod_10_test_mask: Union[int, List[int]] = config.FARMHASH_MOD_10_TEST_MASK,
    closedset_antigens: Optional[List[str]] = None,
    sample_closed: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """From a global dataset get a train_val, an exclusive closed test
    and an exclusive open test dataset.

    Args:
        df_global (pd.DataFrame)
        openset_antigens (List[str], optional): Defaults to config.ANTIGENS_OPENSET.

    Returns:
        df_train_val, df_test_closed_exclusive, df_test_open_exclusive
    """
    assert set(["Antigen", "Slide", "Slide_farmhash_mod_10"]).issubset(set(df_global.columns)), (
        "df_global must have columns Antigen, Slide and Slide_farmhash_mod_10."
    )
    
    df_global.reset_index(inplace=True)

    if openset_antigens is not None:
        mask_ = df_global["Antigen"].isin(openset_antigens)
    else:
        mask_ = pd.Series([False for i in range(df_global.shape[0])])

    if closedset_antigens is None:
        df_closed = df_global.loc[~mask_].copy()
    else:
        df_closed = df_global.loc[(~mask_) & (df_global["Antigen"].isin(closedset_antigens))].copy()
    
    if openset_antigens is not None:
        df_open = df_global.loc[mask_].copy()
        df_test_open_exclusive = df_open.loc[~df_open["Slide"].isin(df_closed["Slide"])].copy()
    else:
        df_test_open_exclusive = None
    
    if sample_closed is not None:
        df_closed = sample_train_val(df_closed, sample_closed)
        

    df_closed["Slide_farmhash_mod_10"] = list(map(
            farmhash_mod_10,
            df_closed["Slide"]
        ))
    
    if type(farmhash_mod_10_test_mask) == int:
        test_mask = df_closed["Slide_farmhash_mod_10"] == farmhash_mod_10_test_mask
    elif type(farmhash_mod_10_test_mask) == list:
        test_mask = df_closed["Slide_farmhash_mod_10"].isin(farmhash_mod_10_test_mask)
    else:
        raise ValueError()
    
    df_train_val = df_closed.loc[~test_mask].copy()
    df_test_closed_exclusive = df_closed.loc[test_mask].copy()
    df_test_closed_exclusive = df_test_closed_exclusive.loc[
            ~df_test_closed_exclusive["Slide"]
            .isin(df_train_val["Slide"])
        ]
    return df_train_val, df_test_closed_exclusive, df_test_open_exclusive


def convert_wide_to_global(df_wide):
    """Convert wide format Absolut binding data to global format (one row per antigen),
    which was used until now.

    Args:
        df_wide (_type_): index is `Slide` sequence, columns are antigens, 
        values are 0/1 if antigen is bound or not.

    Returns:
        _type_: global format `Slide` binding data.
    """    
    df = df_wide.copy()
    df.reset_index(inplace=True)
    df = df.melt(id_vars=["Slide"], var_name="Antigen", value_name="Binding")
    df = df.loc[df["Binding"] == 1]
    df = df.drop(columns=["Binding"])
    df["Slide_farmhash_mod_10"] = df["Slide"].apply(farmhash_mod_10)
    return df
