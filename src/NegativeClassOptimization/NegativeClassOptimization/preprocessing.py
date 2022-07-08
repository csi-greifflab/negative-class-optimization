from typing import List
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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
    encoder = get_one_hot_aa_encoder(),
    scale = True) -> np.array:
    """Encode a dataframe with `Slide` as onehot

    Args:
        df: must have `Slide` column.
        encoder: see `onehot_encode`.
        scale (bool): scale column-wise?

    Returns:
        np.array: _description_
    """      
    df["Slide_onehot"] = df["Slide"].apply(lambda s: onehot_encode(s, encoder=encoder))
    slide_onehot_array = np.stack(df["Slide_onehot"], axis=0)
    if scale:
        scaled_slide_onehot_array = StandardScaler().fit_transform(slide_onehot_array)
        return scaled_slide_onehot_array
    else:
        return slide_onehot_array