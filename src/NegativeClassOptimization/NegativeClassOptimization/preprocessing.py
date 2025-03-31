"""
Preprocessing and transforms.
"""
import logging
import re
import string
import warnings
from typing import List, Optional, Tuple, Union

#import farmhash #temp fix
import numpy as np
import pandas as pd
from sklearn.preprocessing import (LabelEncoder, MultiLabelBinarizer,
                                   OneHotEncoder, StandardScaler)
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
    encoder = OneHotEncoder(sparse_output=False).fit(aa_as_array.reshape(-1, 1))
    return encoder


def onehot_encode(string: str, encoder=get_one_hot_aa_encoder()) -> np.array:
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
    encoder=get_one_hot_aa_encoder(),
) -> pd.DataFrame:
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


def get_no_degree_paratope(agregatesABParatope: str) -> str:
    """Represent paratope in simple format without degree of freedom.

    Args:
        seqAB (str): Ab sequence encoding used in Task4_Merged_Slice_ParaEpi.txt
        motifAB (str): Ab motif encoding used in Task4_Merged_Slice_ParaEpi.txt

    Returns:
        str: simplified no-degree representation.
    """

    simple_paratope = agregatesABParatope
    simple_paratope = simple_paratope.replace("*", "")
    simple_paratope = simple_paratope.replace("--", "-")

    simple_paratope_no_deg = ""
    for char in simple_paratope:
        if char.isdigit():
            pass
        else:
            simple_paratope_no_deg += char

    return simple_paratope_no_deg


def onehot_encode_nodeg_paratope(paratope: str) -> np.ndarray:
    """One hot encode no degree, simple paratope.

    Args:
        paratope (str): simple paratope string obtained from get_no_degree_paratope.

    Returns:
        np.ndarray: one hot encoded paratope (1 x L*20).
    """
    encodings = []
    for char in paratope:
        if char == "-":
            enc = np.zeros(20)
        else:
            enc = onehot_encode(char)
        encodings.append(enc)

    return np.stack(encodings, axis=0).reshape(1, -1)


def onehot_encode_deg_paratope(deg_paratope: str) -> np.ndarray:
    """One hot encode degree, complex paratope (e.g. A2I3--L2D1W3Y1F4D1V4W3*).

    Args:
        paratope (str): e.g. A2I3--L2D1W3Y1F4D1V4W3*.

    Returns:
        np.ndarray: encoded paratope (1 x L*20).
    """

    paratope = deg_paratope
    paratope = paratope.replace("*", "")

    encodings = []
    for i in range(0, len(paratope), 2):
        char, degree = paratope[i], paratope[i + 1]

        if char == "-":
            assert degree == "-"
            enc = np.zeros(20)
        else:
            enc = onehot_encode(char) * int(degree)
        encodings.append(enc)

    return np.stack(encodings, axis=0).reshape(1, -1)


def get_antigen_label_encoder() -> LabelEncoder:
    label_encoder = LabelEncoder()
    label_encoder.fit(config.ANTIGENS_CLOSEDSET)
    return label_encoder


def load_embedder(embedder_type: str):
    """Load embedders from bio_embeddings.

    Args:
        embedder_type (str)

    Raises:
        NotImplementedError
        ValueError

    Returns:
        Embedder
    """
    if embedder_type == "ESM1b":
        from bio_embeddings.embed import ESM1bEmbedder

        return ESM1bEmbedder(
            model_file=str(config.DATA_EMB_DIR / "esm1b/esm1b_t33_650M_UR50S.pt")
        )
    elif embedder_type == "ProtTransT5XLU50":
        from bio_embeddings.embed import ProtTransT5XLU50Embedder

        return ProtTransT5XLU50Embedder(
            model_directory=str(config.DATA_EMB_DIR / "prottrans_t5_xl_u50")
        )
    elif embedder_type == "Half_ProtTransT5XLU50":
        raise NotImplementedError(
            "Half_ProtTransT5XLU50 not implemented yet, "
            "need to research the difference from ProtTransT5XLU50."
        )
    else:
        raise ValueError(f"Unknown embedder type: {embedder_type}.")


def get_vsh8_embedding_matrix():
    """Get the VSH8 matrix, from the paper :))."""
    s_raw = "Ala A   0.15    1.11    1.35    0.92    0.02    0.91    0.36    0.48 Arg R  1.47        1.45        1.24        1.27        1.55        1.47        1.30        0.83 Asn N  0.99        0.00  0.37        0.69  0.55        0.85        0.73  0.80 Asp D  1.15        0.67  0.41  0.01  2.68        1.31        0.03        0.56 Cys C             0.18  1.67  0.46  0.21        0.00        1.20  1.61  0.19 Gln Q  0.96        0.12        0.18        0.16        0.09        0.42  0.20  0.41 Glu E  1.18        0.40        0.10        0.36  2.16  0.17        0.91        0.02 Gly G  0.20  1.53  2.63        2.28  0.53  1.18        2.01  1.34 His H  0.43  0.25        0.37        0.19        0.51        1.28        0.93        0.65 Ile I               1.27  0.14        0.30  1.80        0.30  1.61  0.16  0.13 Leu L             1.36        0.07        0.26  0.80        0.22  1.37        0.08  0.62 Lys K  1.17        0.70        0.70        0.80        1.64        0.67        1.63        0.13 Met M             1.01  0.53        0.43        0.00        0.23        0.10  0.86  0.68 Phe F              1.52        0.61        0.96  0.16        0.25        0.28  1.33  0.20 Pro P              0.22  0.17  0.50        0.05  0.01  1.34  0.19        3.56 Ser S  0.67  0.86  1.07  0.41  0.32        0.27  0.64        0.11 Thr T  0.34  0.51  0.55  1.06  0.06  0.01  0.79        0.39 Trp W             1.50        2.06        1.79        0.75        0.75  0.13  1.01  0.85 Tyr Y             0.61        1.60        1.17        0.73        0.53        0.25  0.96  0.52 Val V             0.76  0.92  0.17  1.91        0.22  1.40  0.24  0.03"

    floats = re.findall(r"[-+]?\d*\.\d+|\d+", s_raw)
    floats = [float(i) for i in floats]

    aminoacids = re.findall(r" [A-Z] ", s_raw)
    aminoacids = [s[1] for s in aminoacids]

    vshe8 = {}
    for i, aa in enumerate(aminoacids):
        vshe8[aa] = floats[i * 8 : i * 8 + 8]

    df = pd.DataFrame.from_dict(
        vshe8, orient="index", columns=[f"VSHE_{i}" for i in range(1, 9)]
    )

    return df


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

    if type(ag_pos[0]) != str:
        raise TypeError(f"ag_pos must be a list of strings: {ag_pos=}")

    def infer_antigen_from_duplicate_list(
        antigens: List[str],
        pos_antigens: List[str],
    ):
        for pos_ag in pos_antigens:
            if pos_ag in antigens:
                return 1
        return 0

    df_f = df.groupby("Slide").apply(
        lambda df_: infer_antigen_from_duplicate_list(
            df_["Antigen"].unique().tolist(), pos_antigens=ag_pos
        )
    )
    df_f = pd.DataFrame(data=df_f, columns=["binds_a_pos_ag"])
    df_f = df_f.reset_index()

    if "embeddings" in df.columns:
        return df_f.merge(df[["Slide", "embeddings"]], on=["Slide"], how="left")
    else:
        return df_f


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
    batch_size=64,
    scale_X=True,
    encoder_type: str = "onehot",
    df_test_open=None,
    sample_train=None,
    use_embeddings=False,
):
    """Get train, test and openset pytorch Datasets and DataLoaders.

    Args:
        df_train_val (pd.DataFrame): dataframe in typical global format.
        ag_pos (List[str]): list of antigens labeled as positive.
        batch_size (int, optional): Defaults to 64.
        train_frac (float, optional): Defaults to 0.8.
        scale_X
        df_test_closed
        df_test_open
        sample_train

    Returns:
        tuple: (train_data, test_data, train_loader, test_loader).
    """

    # TODO: check references and test.

    if not isinstance(ag_pos, list):
        raise TypeError("ag_pos must be a list.")

    if not scale_X:
        warnings.warn("Not scaling X.")

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
    df_test_closed = remove_duplicates_for_binary(df_test_closed, ag_pos)

    if sample_train is not None:
        df_train_val = sample_train_val(df_train_val, sample_train)

    if use_embeddings:
        df_train_val["X"] = df_train_val["embeddings"]
        df_test_closed["X"] = df_test_closed["embeddings"]
    else:
        if encoder_type == "onehot":
            encoder_func = onehot_encode_df
            encoder_colname = "Slide_onehot"
        elif encoder_type == "ProtTransT5XLU50":
            raise NotImplementedError()
        elif encoder_type == "ESM1b":
            raise NotImplementedError()
        else:
            raise ValueError(f"encoder_type {encoder_type} not recognized.")

        df_train_val = encoder_func(df_train_val)
        df_train_val["X"] = df_train_val[encoder_colname]
        df_test_closed = encoder_func(df_test_closed)
        df_test_closed["X"] = df_test_closed[encoder_colname]

    df_train_val["y"] = df_train_val["binds_a_pos_ag"]
    df_test_closed["y"] = df_test_closed["binds_a_pos_ag"]

    if scale_X:
        train_as_mat = arr_from_list_series(df_train_val["X"])
        test_as_mat = arr_from_list_series(df_test_closed["X"])

        scaler = StandardScaler()
        scaler.fit(train_as_mat)
        df_train_val["X"] = scaler.transform(train_as_mat).tolist()
        df_test_closed["X"] = scaler.transform(test_as_mat).tolist()
    else:
        scaler = None

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

        return (
            train_data,
            test_data,
            openset_data,
            train_loader,
            test_loader,
            openset_loader,
        )
    else:
        return (train_data, test_data, train_loader, test_loader)


def construct_open_dataset_loader(df_test_open, batch_size, scaler=None):
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


def sample_df_deterministically(df, sample, num_buckets=16384) -> pd.DataFrame:
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
        df_train_val=df,
        sample_train=sample,
        num_buckets=num_buckets,
    )


def sample_train_val(df_train_val, sample_train, num_buckets=2 * 16384):
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
            df_train_val[slide_hash_colname] = list(
                map(
                    lambda s: farmhash.hash64(s) % num_buckets,
                    df_train_val["Slide"].values.reshape(-1).tolist(),
                )
            )
            sampling_frac = sample_train / nrows
            num_buckets_to_sample = np.round(sampling_frac * num_buckets)
            df_train_val = df_train_val.loc[
                df_train_val[slide_hash_colname] <= num_buckets_to_sample
            ].copy()
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
    except TypeError as error:
        logger.exception(error)
        print(df_train_val.head(2))
        raise
    logger.warning("Resetting the index of df_train_val.")
    return df_train_val.reset_index(
        drop=True
    )  # not resetting index can yield index error in Dataset and DataLoader.


def preprocess_data_for_pytorch_multiclass(
    df,
    batch_size=64,
    train_frac=0.8,
    scale_onehot=True,
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
    scaler=None,
    encoder=None,
    sample=None,
    sample_per_ag=None,
    sample_per_ag_seed=config.SEED,
):
    df = df.loc[df["Antigen"].isin(ags)].copy()
    df = remove_duplicates_for_multiclass(df)

    if sample_per_ag is not None:
        try:
            df = df.groupby("Antigen").sample(
                sample_per_ag, random_state=sample_per_ag_seed
            )
        except ValueError as e:
            print(e)
            print(df["Antigen"].value_counts())
            raise
    elif sample is not None:
        df = sample_train_val(df, sample)

    df, scaler = preprocess_X(
        df, ags, scaler, sample, sample_per_ag, sample_per_ag_seed
    )

    if encoder is None:
        antigens = df["Antigen"].unique().tolist()
        encoder = LabelEncoder().fit(antigens)

    df["y"] = encoder.transform(df["Antigen"])
    df = df[["X", "y"]]
    return df, scaler, encoder


def preprocess_df_for_multilabel(
    df,
    ags: List[str],
    scaler=None,
    encoder=None,
    sample=None,
    sample_per_ag=None,
    sample_per_ag_seed=config.SEED,
):
    df = df.loc[df["Antigen"].isin(ags)].copy()

    if sample_per_ag is not None:
        try:
            df = df.groupby("Antigen").sample(
                sample_per_ag, random_state=sample_per_ag_seed
            )
        except ValueError as e:
            print(e)
            print(df["Antigen"].value_counts())
            raise
    elif sample is not None:
        df = sample_train_val(df, sample)

    df = (
        df.groupby("Slide", as_index=False)[["Antigen", "Slide_farmhash_mod_10"]]
        .apply(
            lambda df_: (sorted(df_["Antigen"]), df_["Slide_farmhash_mod_10"].iloc[0])
        )
        .reset_index()
    )
    df.columns = [col[0] for col in df.columns]

    df, scaler = preprocess_X(
        df,
        scaler,
    )

    multilabel_list: List[List[str]] = df["Antigen"].values.reshape(-1).tolist()
    if encoder is None:
        encoder = MultiLabelBinarizer().fit(multilabel_list)
    # df['y'] = list(map(lambda x: encoder.transform(x), multilabel_list))
    # df['y'] = encoder.transform(multilabel_list)
    df["y"] = [arr for arr in encoder.transform(multilabel_list)]

    return df, scaler, encoder


def preprocess_X(
    df,
    scaler=None,
):
    df = onehot_encode_df(df)

    arr = arr_from_list_series(df["Slide_onehot"])
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(arr)
    scaled_arr = scaler.transform(arr)

    df["X"] = scaled_arr.tolist()
    return df, scaler


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
    assert set(["Antigen", "Slide", "Slide_farmhash_mod_10"]).issubset(
        set(df_global.columns)
    ), "df_global must have columns Antigen, Slide and Slide_farmhash_mod_10."

    df_global.reset_index(inplace=True)

    if openset_antigens is not None:
        mask_ = df_global["Antigen"].isin(openset_antigens)
    else:
        mask_ = pd.Series([False for i in range(df_global.shape[0])])

    if closedset_antigens is None:
        df_closed = df_global.loc[~mask_].copy()
    else:
        df_closed = df_global.loc[
            (~mask_) & (df_global["Antigen"].isin(closedset_antigens))
        ].copy()

    if openset_antigens is not None:
        df_open = df_global.loc[mask_].copy()
        df_test_open_exclusive = df_open.loc[
            ~df_open["Slide"].isin(df_closed["Slide"])
        ].copy()
    else:
        df_test_open_exclusive = None

    if sample_closed is not None:
        df_closed = sample_train_val(df_closed, sample_closed)

    if "Slide_farmhash_mod_10" not in df_closed.columns:
        df_closed["Slide_farmhash_mod_10"] = list(
            map(farmhash_mod_10, df_closed["Slide"])
        )

    if type(farmhash_mod_10_test_mask) == int:
        test_mask = df_closed["Slide_farmhash_mod_10"] == farmhash_mod_10_test_mask
    elif type(farmhash_mod_10_test_mask) == list:
        test_mask = df_closed["Slide_farmhash_mod_10"].isin(farmhash_mod_10_test_mask)
    else:
        raise ValueError()
    df_train_val = df_closed.loc[~test_mask].copy()
    df_test_closed_exclusive = df_closed.loc[test_mask].copy()
    df_test_closed_exclusive = df_test_closed_exclusive.loc[
        ~df_test_closed_exclusive["Slide"].isin(df_train_val["Slide"])
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


def compute_frequencies_and_relative(slides):
    ohs = []
    for slide in slides:
        ohs.append(onehot_encode(slide))

    ohs = np.array(ohs)
    ohs_freq = np.sum(ohs, axis=0) / len(ohs)
    ohs_freq_m = ohs_freq.reshape(11, 20)
    ohs_freq_m_sd = np.std(ohs_freq_m, axis=1)
    ohs_freq_rel_m = ohs_freq_m / np.array([ohs_freq_m_sd for _ in range(20)]).T
    ohs_freq_rel = ohs_freq_rel_m.reshape(220)
    return ohs_freq,ohs_freq_rel


def extract_frequences_as_features(slides, ohs_freq, ohs_freq_rel):
    ## Compute freq per slide
    freqs = []
    for slide in slides:
        freqs.append(ohs_freq[onehot_encode(slide) == 1])

    ## Compute relative freq per slide
    rel_freqs = []
    for slide in slides:
        freq_rel = ohs_freq_rel[onehot_encode(slide) == 1]
        rel_freqs.append(freq_rel)
    
    return freqs, rel_freqs