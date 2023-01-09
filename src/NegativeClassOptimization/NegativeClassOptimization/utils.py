"""
# TODO: Trim many unnecessary functions. They were useful initially to validate
and check the initial dataset files.
"""

from dataclasses import dataclass
from itertools import chain
from multiprocessing.sharedctypes import Value
from pathlib import Path
import random
import re
import uuid
from typing import Optional, List
import numpy as np
import pandas as pd
import torch
import mlflow
import requests

import NegativeClassOptimization.config as config


def nco_seed(seed: int = config.SEED):
    """Seed for the project.
    https://pytorch.org/docs/stable/notes/randomness.html

    Args:
        seed (int, optional): Defaults to config.SEED.
    """    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def shuffle_antigens(antigens: List[str], seed: int = config.SEED) -> List[str]:
    """Shuffle antigens.

    Args:
        antigens (List[str]): antigens to shuffle.
        seed (int, optional): Defaults to config.SEED.

    Returns:
        List[str]: shuffled antigens.
    """    
    from numpy.random import default_rng
    assert seed in {config.SEED}, "Only the default seed is supported."
    rng = default_rng(seed=seed)
    antigens_shuffled = antigens[:]
    rng.shuffle(antigens_shuffled)
    return antigens_shuffled


def generate_ag_set_chain(
    antigens,
    ag_set_sizes: List[int],
    seed_ban_list: Optional[List[List[str]]] = None,
    ) -> List[List[str]]:
    """Generate antigen set chains.

    Args:
        antigens (_type_): _description_
        ag_set_sizes (List[int]): _description_
        seed_ban_list (Optional[List[List[str]]], optional): List of seed sets to exclude. Defaults to None.

    Returns:
        List[List[str]]: _description_
    """    
    ag_sets_chain = []
    for size in ag_set_sizes:
        if len(ag_sets_chain) == 0:
            while True:
                ag_set = sorted(np.random.choice(antigens, size=size, replace=False))
                if (seed_ban_list is None) or (ag_set not in seed_ban_list):
                    break
        else:
            last_set = ag_sets_chain[-1]
            increment = size - len(last_set)
            ag_set = sorted(np.random.choice(list(set(antigens) - set(last_set)), size=increment, replace=False))
            ag_set = sorted(last_set + ag_set)    
        ag_sets_chain.append(ag_set)
    
    assert list(map(len, ag_sets_chain)) == ag_set_sizes, "ag_set_sizes not correct"
    return ag_sets_chain


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


def load_global_dataframe(path = config.DATA_SLACK_1_GLOBAL):
    return pd.read_csv(path, sep='\t', dtype={"Antigen": str}).iloc[:, 1:]


def load_processed_dataframes(
    dir_path = config.DATA_SLACK_1_PROCESSED_DIR,
    sample: Optional[int] = None,
    ) -> dict:
    """Loads processed dataframes for ml runs.

    Args:
        dir_path (_type_, optional): Defaults to config.DATA_SLACK_1_PROCESSED_DIR.
        sample (Optional[int], optional): samples train_val, test closed and test open. Defaults to None.

    Returns:
        dict: _description_
    """

    if sample is None:
        load_df = lambda fname: (
            pd.read_csv(dir_path / fname, sep='\t', dtype={"Antigen": str})
        )
    else:
        load_df = lambda fname: (
            pd.read_csv(dir_path / fname, sep='\t', dtype={"Antigen": str})
            .sample(frac=1)
            .sample(sample)
        )
    return {
        "train_val": load_df("df_train_val.tsv"),
        "test_closed_exclusive": load_df("df_test_closed_exclusive.tsv"),
        "test_open_exclusive": load_df("df_test_open_exclusive.tsv"),
    }


def mlflow_log_params_online_metrics(online_metrics: dict) -> None:
    for i, epoch_metrics in enumerate(online_metrics):
        epoch = i+1
        try:
            mlflow.log_metrics(
                    process_epoch_metrics(epoch_metrics),
                    step=epoch
                )
        except TypeError as e:
            print(f"TypeError: {e}")
            print(epoch_metrics)


def process_epoch_metrics(epoch_metrics: dict) -> dict:
    metrics = {}
    metrics["train_loss"] = epoch_metrics["train_losses"][-1]
    metrics_iter = chain(
        epoch_metrics["test_metrics"].items(),
        epoch_metrics["open_metrics"].items(),
    )
    for metric, val in metrics_iter:
        if type(val) != np.ndarray:
            metrics[metric] = val
        
    return metrics


def download_absolut(
    out_dir: Path = config.DATA_ABSOLUT_DIR,
    doi_csv: Path = config.DATA_ABSOLUT_DOI,
) -> None:
    """Download Absolut raw data (https://ns9999k.webs.sigma2.no/10.11582_2021.00063).

    Args:
        out_dir (Path, optional): output directory. Defaults to config.DATA_ABSOLUT_DIR.
        doi_csv (Path, optional): path to csv with filepaths (as obtained from Norwegian database website). Defaults to config.DATA_ABSOLUT_DOI.
    """
    from urllib import request

    df = pd.read_csv(doi_csv, header=None)
    url_paths: List[str] = df.iloc[:, 1].to_list()
    
    URL = "https://ns9999k.webs.sigma2.no/10.11582_2021.00063"
    for url_path in url_paths:
        print(f"Processing {url_path=}")
        url = URL + url_path[1:]
        filepath = url_path.split("AbsolutOnline/")[1]
        filepath = Path(out_dir) / filepath
        
        if not filepath.parent.exists():
            filepath.parent.mkdir(parents=True)
        request.urlretrieve(url, filename=str(filepath))


def extract_antigens_from_string(liststring: str) -> List[str]:
    regex_pattern = r"[A-Z0-9]+"
    return re.findall(
        regex_pattern,
        liststring
    )


def num_trainable_params(model) -> int:
    """Get number of trainable parameters in pytorch model.

    Args:
        model (nn.Module): pytorch model

    Returns:
        int: number of trainable parameters.
    """    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    return num_params


class MlflowAPI:
    """Class to interact with mlflow API.
    """    

    def __init__(self):
        self.URL = "http://10.40.3.22:5000/api/2.0/mlflow/runs/search"
        self.response = None


    def mlflow_request(self, experiment_id: str, run_name: Optional[str] = None):
        self.response = requests.post(
            self.URL,
                json={
                    "experiment_ids": [experiment_id],
                    "filter": f'tags."mlflow.runName" = "{run_name}" and attributes.status = "FINISHED"',
                },
            ).json()
        return self.response


    def build_mlflow_results_df(self):
        
        assert self.response is not None

        def build_record_from_mlflow_record(mlflow_record_data: dict) -> dict:
            record = {}
            for mlflow_record in (*mlflow_record_data["params"], *mlflow_record_data["metrics"], *mlflow_record_data["tags"]):
                record[mlflow_record["key"]] = mlflow_record["value"]
            return record
        
        mlflow_records = [self.response["runs"][idx]["data"] for idx in range(len(self.response["runs"]))]
        df = pd.DataFrame.from_records(
            map(
                build_record_from_mlflow_record,
                mlflow_records,
                )
        )
        return df
