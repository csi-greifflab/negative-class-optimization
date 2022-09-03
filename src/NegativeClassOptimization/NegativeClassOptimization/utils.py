"""
# TODO: Trim many unnecessary functions. They were useful initially to validate
and check the initial dataset files.
"""

from dataclasses import dataclass
from pathlib import Path
import uuid
from typing import Optional, List
import numpy as np
import pandas as pd


import NegativeClassOptimization.config as config


def summarize_data_files(path: Path) -> pd.DataFrame:
    """Function to summarize the data files obtained in
    the `Slack` format. This is the file structure of the
    data that we first received from Slack.

    Args:
        path (Path): _description_

    Returns:
        pd.DataFrame: _description_
    """    
    filepaths = path.glob("*")
    records = []
    for filepath in filepaths:
        fname = filepath.name
        ftype = fname.split(".")[-1]
        
        if ftype == "csv":
            datatype = "corpus"
        elif ftype == "txt":
            datatype = "features"
        else:
            continue
        
        if fname.split("_")[0] != "outputFeaturesFile":
            antigen = fname.split("_")[0]
        else:
            antigen = None
        

        records.append({
            "filepath": filepath,
            "filename": fname,
            "filetype": ftype,
            "antigen": antigen,
            "datatype": datatype,
        })
    return pd.DataFrame.from_records(records)


@dataclass
class AntigenData:
    corpus: Path
    features: Path

    df_c: Optional[pd.DataFrame] = None
    df_f: Optional[pd.DataFrame] = None

    def __init__(self, antigen: str, base_path: Path, load=True):
        self.antigen = antigen
        self.base_path = base_path
        self.corpus = base_path / f"{antigen}_top_70000_corpus.csv"
        self.features = base_path / f"{antigen}_outputFeaturesFile.txt"
        if load:
            self.validate()
    
    def read_corpus(self) -> pd.DataFrame:
        self.df_c = pd.read_csv(self.corpus)
        self.df_c["UID"] = self.antigen + "_" + self.df_c["ID_slide_Variant"]
        return self.df_c

    def read_features(self) -> pd.DataFrame:
        self.df_f = pd.read_csv(self.features, sep='\t', header=1)
        self.df_f["UID"] = self.antigen + "_" + self.df_c["ID_slide_Variant"]
        return self.df_f

    def validate(self) -> bool:
        df_c = self.read_corpus()
        df_f = self.read_features()
        same_ids = (
            set(df_c["ID_slide_Variant"]) 
            == set(df_f["ID_slide_Variant"])
        )
        all_are_best = all(df_c['Best'].unique() == True)
        ids_unique = (
            df_c["ID_slide_Variant"].unique().shape[0] 
            == df_c.shape[0]
        ) 
        return same_ids and all_are_best and ids_unique


def antigens_from_dataset_path(dataset_path: Path) -> List[str]:
    return (
        summarize_data_files(dataset_path)
        ["antigen"].unique().tolist()
    )


def build_global_dataset(
    dataset_path: Path,
    remove_ag_slide_duplicates = True,
    ):
    antigens: List[str] = antigens_from_dataset_path(dataset_path)

    dfs = []
    for antigen in antigens:
        ag_data = AntigenData(antigen, Path(dataset_path))
        df_component = ag_data.df_c
        df_component["Antigen"] = ag_data.antigen
        dfs.append(ag_data.df_c)

    df_global = pd.concat(dfs, axis=0)

    # Remove duplicated Slide that bind the same Antigen
    if remove_ag_slide_duplicates:
        df_global = df_global.groupby("Antigen").apply(
            lambda df_: df_
            .sort_values(["Slide", "Energy"], ascending=True)
            .drop_duplicates("Slide", keep="first")
            ).reset_index(drop=True)

    return df_global


def build_random_dataset(
    num_seq: int, 
    cdr3_len_distr: dict = config.GLOBAL_CDR3_LEN_DISTR,
    alphabet: list = config.AMINOACID_ALPHABET,
    seed=config.SEED,
    ) -> pd.DataFrame:
    
    np.random.seed(seed)
    cdr3_records = []
    for _ in range(num_seq):
        random_size = np.random.choice(
            list(cdr3_len_distr.keys()),
            size=1,
            p=list(cdr3_len_distr.values())
        )
        random_sequence = "".join(
            np.random.choice(
                alphabet,
                size=random_size
            )
        )
        cdr3_records.append({
            "CDR3": random_sequence,
            "UID": "random_" + str(uuid.uuid4())[:8]
        })
    df = pd.DataFrame.from_records(cdr3_records)
    df = df.drop_duplicates(["UID"])
    df["Antigen"] = "random"
    return df