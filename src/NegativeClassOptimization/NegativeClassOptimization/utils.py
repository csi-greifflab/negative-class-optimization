"""
# TODO: Trim many unnecessary functions. They were useful initially to validate
and check the initial dataset files.
"""

import json
import random
import re
import uuid
import zipfile
from dataclasses import dataclass
from itertools import chain
from multiprocessing.sharedctypes import Value
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import requests
import torch
from Bio import motifs
from Bio.Seq import Seq
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

import mlflow
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

        records.append(
            {
                "filepath": filepath,
                "filename": fname,
                "filetype": ftype,
                "antigen": antigen,
                "datatype": datatype,
            }
        )
    return pd.DataFrame.from_records(records)


def unzip_file(path, output_path):
    """Unzip a file to a given path."""
    with zipfile.ZipFile(path, "r") as zip_ref:
        zip_ref.extractall(output_path)


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
        assert self.df_c is not None, "Read corpus first."
        self.df_f = pd.read_csv(self.features, sep="\t", header=1)
        self.df_f["UID"] = self.antigen + "_" + self.df_c["ID_slide_Variant"]
        return self.df_f

    def validate(self) -> bool:
        df_c = self.read_corpus()
        df_f = self.read_features()
        same_ids = set(df_c["ID_slide_Variant"]) == set(df_f["ID_slide_Variant"])
        all_are_best = all(df_c["Best"].unique() == True)
        ids_unique = df_c["ID_slide_Variant"].unique().shape[0] == df_c.shape[0]
        return same_ids and all_are_best and ids_unique


def antigens_from_dataset_path(dataset_path: Path) -> List[str]:
    return summarize_data_files(dataset_path)["antigen"].unique().tolist()


def convert_antigen_name_to_full_name(antigen_name: str) -> str:
    """Convert antigen name to full name."""
    convertor = {
        "3VRL": "3VRL_C",
        "1NSN": "1NSN_S",
        "3RAJ": "3RAJ_A",
        "5E94": "5E94_G",
        "1H0D": "1H0D_C",
        "1WEJ": "1WEJ_F",
        "1ADQ": "1ADQ_A",
        "1FBI": "1FBI_X",
        "2YPV": "2YPV_A",
        "1OB1": "1OB1_C",
    }
    return convertor[antigen_name]


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
            ag_set = sorted(
                np.random.choice(
                    list(set(antigens) - set(last_set)), size=increment, replace=False
                )
            )
            ag_set = sorted(last_set + ag_set)
        ag_sets_chain.append(ag_set)

    assert list(map(len, ag_sets_chain)) == ag_set_sizes, "ag_set_sizes not correct"
    return ag_sets_chain


def build_global_dataset(
    dataset_path: Path,
    remove_ag_slide_duplicates=True,
):
    antigens: List[str] = antigens_from_dataset_path(dataset_path)

    dfs = []
    for antigen in antigens:
        ag_data = AntigenData(antigen, Path(dataset_path))
        assert ag_data.df_c is not None, "Read corpus first."
        df_component = ag_data.df_c
        df_component["Antigen"] = ag_data.antigen
        dfs.append(ag_data.df_c)

    df_global = pd.concat(dfs, axis=0)

    # Remove duplicated Slide that bind the same Antigen
    if remove_ag_slide_duplicates:
        df_global = (
            df_global.groupby("Antigen")
            .apply(
                lambda df_: df_.sort_values(
                    ["Slide", "Energy"], ascending=True
                ).drop_duplicates("Slide", keep="first")
            )
            .reset_index(drop=True)
        )

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
            list(cdr3_len_distr.keys()), size=1, p=list(cdr3_len_distr.values())
        )
        random_sequence = "".join(np.random.choice(alphabet, size=random_size))
        cdr3_records.append(
            {"CDR3": random_sequence, "UID": "random_" + str(uuid.uuid4())[:8]}
        )
    df = pd.DataFrame.from_records(cdr3_records)
    df = df.drop_duplicates(["UID"])
    df["Antigen"] = "random"
    return df


def load_global_dataframe(path=config.DATA_SLACK_1_GLOBAL):
    dir_ = path.parent
    basename = path.stem
    farmhashed_path = dir_ / f"{basename}_farmhashed.tsv"
    if (farmhashed_path).exists():
        return pd.read_csv(farmhashed_path, sep="\t", dtype={"Antigen": str})
    else:
        return pd.read_csv(path, sep="\t", dtype={"Antigen": str}).iloc[:, 1:]


def load_processed_dataframes(
    dir_path=config.DATA_SLACK_1_PROCESSED_DIR,
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
            pd.read_csv(dir_path / fname, sep="\t", dtype={"Antigen": str})
        )
    else:
        load_df = lambda fname: (
            pd.read_csv(dir_path / fname, sep="\t", dtype={"Antigen": str})
            .sample(frac=1)
            .sample(sample)
        )
    return {
        "train_val": load_df("df_train_val.tsv"),
        "test_closed_exclusive": load_df("df_test_closed_exclusive.tsv"),
        "test_open_exclusive": load_df("df_test_open_exclusive.tsv"),
    }


def load_1v1_binary_dataset(
    ag_pos="3VRL",
    ag_neg: Union[str, List[str]] = "1ADQ",
    num_samples: Optional[int] = 20000,
    drop_duplicates=True,
    with_paratopes=False,
):
    if isinstance(ag_neg, str):
        ag_neg = [ag_neg]

    df = load_global_dataframe()
    df = df.loc[df["Antigen"].isin([ag_pos, *ag_neg])].copy()

    if with_paratopes:
        df_para = load_paratopes()
        df_para["Antigen"] = df_para["Label"].str.split("_").str[0]
        df_para = df_para.loc[df_para["Antigen"].isin([ag_pos, *ag_neg])].copy()

        # Merge on Slide and Antigen, since there are multiple paratopes per slide.
        df = pd.merge(
            df,
            df_para,
            on=("Slide", "Antigen"),
            how="left",
        )
        df = df.iloc[:, 2:]

    if drop_duplicates:
        df = df.drop_duplicates(["Slide"])

    if num_samples is not None:
        df = df.sample(n=num_samples, random_state=42)
    df = df.sample(frac=1, random_state=42)

    return df


def load_paratopes(
    path=config.DATA_SLACK_1_PARATOPES,
) -> pd.DataFrame:
    df_para = pd.read_csv(path, sep="\t", dtype={"Label": str})

    df_para = df_para[
        [
            "Slide",
            "Label",
            "hotspot_ID",
            "agregatesAGEpitope",
            "agregatesABParatope",
        ]
    ].copy()

    df_para["Antigen"] = df_para["Label"].str.split("_").str[0]
    return df_para


def load_raw_bindings_murine(ag, base_path=config.DATA_SLACK_1_RAWBINDINGSMURINE):
    """
    **DEPRECATED**

    Use `load_binding_per_ag`.

    I tried to use the murine binding data, `Absolut/data/RawBindingsMurine`,
    but it proved not useful. I didn't find interesections with paratopes and,
    overall, it's source is unclear. We use `Absolut/data/RawBindingsPerClassMurine/`
    instead.
    """
    raise DeprecationWarning("DEPRECATED.")

    ag_full = [
        ag_i.name for ag_i in list(base_path.glob("*")) if ag_i.stem.split("_")[0] == ag
    ][0]
    ag_dir = base_path / f"{ag_full}/{ag_full}"
    num_files = len(list(ag_dir.glob("*Process*.txt")))

    df = pd.DataFrame()
    for i in range(1, num_files + 1):
        df_ = pd.read_csv(
            Path(ag_dir / f"{ag_full}FinalBindings_Process_{i}_Of_{num_files}.txt"),
            header=1,
            sep="\t",
        )
        df_.sort_values("Energy", ascending=True, inplace=True)
        df_.drop_duplicates(subset="Slide", keep="first", inplace=True)
        df = pd.concat([df, df_], axis=0)
    return df


def build_binding_binary_dataset(ag, dataset_type: str, df=None, seed=config.SEED):
    """
    **DEPRECATED**

    Check `load_raw_bindings_murine` documentation.
    Use `build_binding_dataset_per_ag`.
    """

    raise DeprecationWarning("DEPRECATED.")

    if df is None:
        df = load_raw_bindings_murine(ag)

    perc_1 = df["Energy"].quantile(0.01)
    perc_5 = df["Energy"].quantile(0.05)

    df_pos = df.loc[df["Energy"] <= perc_1].sample(50000, random_state=seed)
    df_pos["Antigen"] = f"{ag}_high"

    if dataset_type == "high_looser":
        df_neg = df.loc[(perc_1 < df["Energy"]) & (df["Energy"] <= perc_5)].sample(
            50000, random_state=seed
        )
        df_neg["Antigen"] = f"{ag}_looser"
    elif dataset_type == "high_95low":
        df_neg = df.loc[df["Energy"] > perc_5].sample(50000, random_state=seed)
        df_neg["Antigen"] = f"{ag}_95low"
    else:
        raise ValueError(f"Invalid dataset_type: {dataset_type}")

    df_final = pd.concat([df_pos, df_neg], axis=0)
    df_final = df_final.sample(frac=1).reset_index(drop=True)  # shuffle

    return df_final


def load_binding_per_ag(
    ag: str, base_path=config.DATA_SLACK_1_RAWBINDINGS_PERCLASS_MURINE
):
    """Load binding data for a given antigen."""

    validate_ag_miniabsolut(ag)

    # Get the full name (Absolut) of the antigen.
    ag_full = resolve_full_ag_name(ag, base_path)

    ag_dir = base_path / f"{ag_full}Analyses"

    mascotte_path = ag_dir / f"{ag_full}_MascotteSlices.txt.zip"
    losser_exc_path = ag_dir / f"{ag_full}_LooserExclusiveSlices.txt.zip"
    nonmascotte_path = ag_dir / f"{ag_full}_500kNonMascotte.txt.zip"

    reader = lambda path: pd.read_csv(path, compression="zip", sep="\t", header=1)
    df_mascotte = reader(mascotte_path)
    df_looser_exc = reader(losser_exc_path)
    df_nonmascotte = reader(nonmascotte_path)

    df_mascotte["Source"] = "mascotte"
    df_looser_exc["Source"] = "looserX"
    df_nonmascotte["Source"] = "nonmascotte"
    df = pd.concat([df_mascotte, df_looser_exc, df_nonmascotte], axis=0)

    return df


def resolve_full_ag_name(ag, base_path=config.DATA_SLACK_1_RAWBINDINGS_PERCLASS_MURINE):
    """Resolve the full name of an antigen, i.e. name in full Absolut dataset."""
    ag_full: str = [
        path_i.name.split("Analyses")[0]
        for path_i in list(base_path.glob("*"))
        if path_i.stem.split("_")[0] == ag
    ][0]

    return ag_full


def build_binding_dataset_per_ag(
    ag,
    dataset_type: str,
    df=None,
    seed=config.SEED,
    num_slides: int = 80000,
):
    """Build a binary dataset for a given antigen."""

    validate_ag_miniabsolut(ag)

    if df is None:
        df = load_binding_per_ag(ag)

    # Duplicated slides exist. We keep the slide with the lowest energy.
    df.sort_values("Energy", ascending=True, inplace=True)
    df.drop_duplicates(subset="Slide", keep="first", inplace=True)

    try:
        df_pos = df.loc[df["Source"] == "mascotte"].sample(
            num_slides // 2, random_state=seed
        )
        df_pos["Antigen"] = f"{ag}_high"
    except ValueError:
        raise ValueError(
            f"{ag} has less than {num_slides // 2} mascotte slides - {(df['Source'] == 'mascotte').sum()}."
        )

    if dataset_type == "high_looser":
        df_neg = df.loc[df["Source"] == "looser_exc"].sample(
            num_slides // 2, random_state=seed
        )
        df_neg["Antigen"] = f"{ag}_looser"
    elif dataset_type == "high_95low":
        looser_max_energy = max(df.loc[df["Source"] == "looser_exc", "Energy"])
        df_neg = df.loc[
            (df["Source"] == "nonmascotte") & (df["Energy"] > looser_max_energy)
        ].sample(50000, random_state=seed)
        df_neg["Antigen"] = f"{ag}_95low"
    else:
        raise ValueError(f"Invalid dataset_type: {dataset_type}")

    df_final = pd.concat([df_pos, df_neg], axis=0)
    df_final = df_final.sample(frac=1, random_state=seed).reset_index(
        drop=True
    )  # shuffle
    assert df_final["Slide"].duplicated().sum() == 0, "Duplicated slides."

    return df_final


def validate_ag_miniabsolut(ag):
    assert ag in config.ANTIGENS, f"Antigen {ag} not in MiniAbsolut."


def mlflow_log_params_online_metrics(online_metrics: dict) -> None:
    for i, epoch_metrics in enumerate(online_metrics):
        epoch = i + 1
        try:
            mlflow.log_metrics(process_epoch_metrics(epoch_metrics), step=epoch)
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
    antigens_only = False,
) -> None:
    """Download Absolut raw data (https://ns9999k.webs.sigma2.no/10.11582_2021.00063).
    Args:
        out_dir (Path, optional): output directory. Defaults to config.DATA_ABSOLUT_DIR.
        doi_csv (Path, optional): path to csv with filepaths (as obtained from Norwegian database website). Defaults to config.DATA_ABSOLUT_DOI.
    """
    from urllib import request

    df = pd.read_csv(doi_csv, header=None)
    url_paths: List[str] = df.iloc[:, 1].to_list()

    url_paths_new = []
    if antigens_only:
        for path in url_paths:
            for ag in config.ANTIGENS:
                if ag in str(path):
                    url_paths_new.append(path)
    url_paths = url_paths_new[:]
    del url_paths_new    

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
    return re.findall(regex_pattern, liststring)


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


def get_uid() -> str:
    """Get unique id.

    Returns:
        str: unique id of length 8
    """
    return str(uuid.uuid4())[:8]


def extract_contributions_from_string(string: str):
    """Extract the contributions from a string.

    Examples:
        extract_contributions_from_string(df["contribPerAAparaBind"][0])
        extract_contributions_from_string(df["contribPerAAparaFold"][3])
    """

    if type(string) != str and np.isnan(string):
        return [0] * 11, [0.0] * 11

    # Regex pattern to extract the contributions
    # Pattern is {AA}{index as small letter}:{degree as integer}_{contribution as float}. Extract all as groups
    pattern = re.compile(r"([A-Z])([a-z]):(\d+)_(-\d+\.\d+)")

    # Extract the contributions
    contributions = pattern.findall(string)
    degrees_indexed = {}
    energies_indexed = {}
    for aa, index, degree, contribution in contributions:
        degrees_indexed[index] = int(degree)
        energies_indexed[index] = float(contribution)
    degrees = []
    energies = []
    for index in "abcdefghijk":
        if index not in degrees_indexed.keys():
            degrees.append(0)
            energies.append(0.0)
        else:
            degrees.append(degrees_indexed[index])
            energies.append(energies_indexed[index])
    return degrees, energies


def pwm(seqs: List[str], alphabet=config.AMINOACID_ALPHABET) -> np.ndarray:
    """Get the position weight matrix of a list of sequences."""

    seqs_ = [Seq(s) for s in seqs]
    m = motifs.create(seqs_, alphabet=alphabet)

    # Get the position weight matrix
    pwm: np.ndarray = pd.DataFrame(m.pwm).values
    # pwm += 1e-20  # Avoid log(0)
    return pwm


def get_pwm(slides_1, slides_2):
    # Create a list of Seq objects
    seqs_1 = [Seq(slide) for slide in slides_1]
    seqs_2 = [Seq(slide) for slide in slides_2]

    # Create a motifs instance
    m_1 = motifs.create(seqs_1, alphabet=config.AMINOACID_ALPHABET)  # type: ignore
    m_2 = motifs.create(seqs_2, alphabet=config.AMINOACID_ALPHABET)  # type: ignore

    # Get the position weight matrix
    pwm_1: np.ndarray = pd.DataFrame(m_1.pwm).values
    pwm_1 += 1e-20  # Avoid log(0)
    pwm_2: np.ndarray = pd.DataFrame(m_2.pwm).values
    pwm_2 += 1e-20  # Avoid log(0)
    return pwm_1, pwm_2


def jensen_shannon_divergence_slides(slides_1, slides_2):
    pwm_1, pwm_2 = get_pwm(slides_1, slides_2)
    return jensenshannon(pwm_1, pwm_2, axis=1, base=2).sum()  # type: ignore


def split_to_train_test_rest_dfs(N_train, N_test, df_ag, random_state=None):
    if random_state is None:
        random_state = config.SEED
    df_train = df_ag.sample(n=N_train, random_state=random_state)
    df_test = df_ag.loc[~df_ag.index.isin(df_train.index)].sample(
        n=N_test, random_state=random_state
    )
    df_rest = df_ag.loc[
        ~df_ag.index.isin(df_train.index) & ~df_ag.index.isin(df_test.index)
    ].copy()
    return df_train, df_test, df_rest


def save_train_test_rest(prefix, N_train, N_test, ag_dir, df_train, df_test, df_rest):
    df_train.to_csv(ag_dir / f"{prefix}_train_{N_train}.tsv", sep="\t")
    df_test.to_csv(ag_dir / f"{prefix}_test_{N_test}.tsv", sep="\t")
    df_rest.to_csv(ag_dir / f"{prefix}_rest.tsv", sep="\t")


def compute_frequencies_and_relative(slides):
    ohs = []
    for slide in slides:
        ohs.append(preprocessing.onehot_encode(slide))

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
        freqs.append(ohs_freq[preprocessing.onehot_encode(slide) == 1])

    ## Compute relative freq per slide
    rel_freqs = []
    for slide in slides:
        freq_rel = ohs_freq_rel[preprocessing.onehot_encode(slide) == 1]
        rel_freqs.append(freq_rel)
    
    return freqs, rel_freqs


def build_dataset_into_Absolut(N_train, N_test, MAKE_SPLITS, seed, ag, df):
    """
    Used initially in 01b to fit experimental data into MiniAbsolut data structure.
    Later used to fit into MiniAbsolut the epitope-based sequences.
    CAVE: The naming of the files is not consistent with the naming of the files in MiniAbsolut,
          but it makes our life much easier to preserve the naming. (15k, 5k sequences)
    """
    if MAKE_SPLITS:
        base_p = Path(config.DATA_MINIABSOLUT_SPLITS) / f"MiniAbsolut_Seed{seed}"
        base_p.mkdir(exist_ok=True, parents=False)
    else:
        base_p = config.DATA_MINIABSOLUT
        base_p.mkdir(exist_ok=True, parents=False)


    ag_dir = base_p / ag
    ag_dir.mkdir(exist_ok=True, parents=False)

    # Get the high binders.
    df_high = df[df["binder_type"] == f"{ag}_high"].copy()
    df_high.drop(columns=["binder_type"], inplace=True)
    df_train, df_test, df_rest = split_to_train_test_rest_dfs(
        N_train,
        N_test,
        df_high,
        random_state=seed,
    )
    # save_train_test_rest(
    #     "high", N_train, N_test, ag_dir, df_train, df_test, df_rest
    # )
    ## Hack to fit naming of files.
    ## Even though for some experimental datasets,
    ## such as Porebski, we don't have 15k and 5k sequences,
    ## it makes our life much easier to preserve the naming.
    save_train_test_rest(
        "high", 15000, 5000, ag_dir, df_train, df_test, df_rest
    )

    # Get the weak binders
    df_weak = df[df["binder_type"] == f"{ag}_looserX"].copy()
    df_weak.drop(columns=["binder_type"], inplace=True)
    df_train, df_test, df_rest = split_to_train_test_rest_dfs(
        N_train,
        N_test,
        df_weak,
        random_state=seed,
    )
    # See note from above for explaining the 2 numbers.
    save_train_test_rest(
        "looserX", 15000, 5000, ag_dir, df_train, df_test, df_rest
    )

    # Get the nonbinders
    df_nonbinder = df[df["binder_type"] == f"{ag}_95low"].copy()
    df_nonbinder.drop(columns=["binder_type"], inplace=True)
    df_train, df_test, df_rest = split_to_train_test_rest_dfs(
        N_train,
        N_test,
        df_nonbinder,
        random_state=seed,
    )
    # See note from above for explaining the 2 numbers.
    save_train_test_rest(
        "95low", 15000, 5000, ag_dir, df_train, df_test, df_rest
    )


class MlflowAPI:
    """Class to interact with mlflow API.

    Example:
    ```
    api = MlflowAPI()
    api.mlflow_request(experiment_id="11")
    df = api.build_mlflow_results_df()
    ```
    """

    def __init__(self):
        self.URL = "http://10.40.3.22:5000/api/2.0/mlflow/runs/search"
        self.response = None

    def mlflow_request(self, experiment_id: str, run_name: Optional[str] = None):
        if run_name is not None:
            filter = f'tags."mlflow.runName" = "{run_name}" and attributes.status = "FINISHED"'
        else:
            filter = 'attributes.status = "FINISHED"'
        self.response = requests.post(
            self.URL,
            json={
                "experiment_ids": [experiment_id],
                "filter": filter,
            },
        ).json()
        return self.response

    def build_mlflow_results_df(self):
        assert self.response is not None

        def build_record_from_mlflow_record(mlflow_record_data: dict) -> dict:
            record = {}
            try:
                for mlflow_record in (
                    *mlflow_record_data["params"],
                    *mlflow_record_data["metrics"],
                    *mlflow_record_data["tags"],
                ):
                    record[mlflow_record["key"]] = mlflow_record["value"]
            except KeyError:
                print(f"KeyError in `mlflow_record_data`.")
                pass
            return record

        mlflow_records = [
            self.response["runs"][idx]["data"]
            for idx in range(len(self.response["runs"]))
        ]
        df = pd.DataFrame.from_records(
            map(
                build_record_from_mlflow_record,
                mlflow_records,
            )
        )
        return df

    def list_artifacts(self):
        URL = "http://10.40.3.22:5000/api/2.0/mlflow/artifacts/list"


class MLFlowTaskAPI(MlflowAPI):
    """Helper class to fetch results from MLFlow per task."""

    def get_experiment_and_run(
        self, task: dict, most_recent: bool = True, run_name=None
    ):
        experiment_id = MLFlowTaskAPI.get_experiment_id(task)
        # Filter by ag_pos
        df = self.get_results_and_filter_ag_pos(
            experiment_id, task["ag_pos"], run_name=run_name
        )
        # Filter by ag_neg for each experiment_id
        df = self.filter_ag_neg(experiment_id, task["ag_neg"], df)
        df = df.loc[
            df["shuffle_antigen_labels"] == task["shuffle_antigen_labels"]
        ].copy()

        # Filter by most recent
        if most_recent:
            df = df.iloc[0:1].copy()
        else:
            raise NotImplementedError("Only fetching most recent is implemented.")

        # By this point should be 1
        assert df.shape[0] == 1

        run_id = MLFlowTaskAPI.get_run_id(df)
        return experiment_id, run_id

    @staticmethod
    def get_experiment_id(task: dict):
        """Given a task specification, fetch experiment_id and run_id"""
        ag_pos = task["ag_pos"]
        ag_neg = task["ag_neg"]

        # Resolve binders
        if "_" in ag_pos and "_" in ag_neg:
            return "14"
        elif "_" in ag_pos and "_" not in ag_neg:
            raise ValueError(f"Unrecognized task: {task}")
        elif "_" not in ag_pos and "_" in ag_neg:
            raise ValueError(f"Unrecognized task: {task}")
        else:
            if ag_neg == "9":
                return "13"
            elif isinstance(ag_neg, tuple) and len(ag_neg) == 2:
                return "12"
            else:
                return "11"

    def get_results_and_filter_ag_pos(self, experiment_id, ag_pos, run_name=None):
        if run_name is None:
            self.mlflow_request(experiment_id)
        else:
            self.mlflow_request(experiment_id, run_name=run_name)
        df = self.build_mlflow_results_df()
        df = df.loc[df["ag_pos"] == ag_pos]
        # sort by date
        datetimes: List[str] = []
        for idx, row in df.iterrows():
            s: str = row["mlflow.log-model.history"]
            d: List[dict] = json.loads(s)
            datetime: str = d[0]["utc_time_created"]
            datetimes.append(datetime)
        df["date"] = pd.to_datetime(datetimes)
        df = df.sort_values(by="date", ascending=False)
        return df

    def filter_ag_neg(self, experiment_id, ag_neg, df):
        if experiment_id == "11" or experiment_id == "14":
            df = df.loc[df["ag_neg"] == ag_neg]
        elif experiment_id == "12":
            idxs = []
            for i, row in df.iterrows():
                ag_neg_tupl_str = row["ag_neg"]
                ag_neg_tupl = tuple(
                    ag_neg_tupl_str.replace("(", "")
                    .replace(")", "")
                    .replace("'", "")
                    .split(", ")
                )
                if set(ag_neg_tupl) == set(ag_neg):
                    idxs.append(i)
            df = df.loc[idxs]
        elif experiment_id == "13":
            pass
        else:
            raise ValueError(f"Unrecognized experiment_id: {experiment_id}")
        return df

    @staticmethod
    def get_run_id(df):
        assert len(df) == 1, "Expected 1 result"
        model_history: str = df["mlflow.log-model.history"].values[0]
        run_id = df.run_id_from_model_history(model_history)
        return run_id

    @staticmethod
    def run_id_from_model_history(model_history: str) -> str:  # type: ignore
        model_history: dict = json.loads(model_history)
        run_id = model_history[0]["run_id"]
        return run_id

    @staticmethod
    def mlflow_results_as_dataframe(
        exp_list: List[str], run_name: str, classify_tasks=False
    ) -> pd.DataFrame:
        """
        Example:
            experiment_ids = ["11", "13", "14"]
            run_name = "dev-v0.1.2-3-with-replicates"
        """
        api = MLFlowTaskAPI()

        dfs = []
        for exp_id in exp_list:
            api.mlflow_request(exp_id, run_name=run_name)
            df = api.build_mlflow_results_df()
            df["experiment"] = exp_id
            dfs.append(df)

        df = pd.concat(dfs, axis=0)

        if "mlflow.log-model.history" in df.columns:
            df = df.loc[~df["mlflow.log-model.history"].isna()].copy()  # type: ignore
            df["run_id"] = df["mlflow.log-model.history"].apply(
                MLFlowTaskAPI.run_id_from_model_history
            )

        if classify_tasks:
            df["task_type"] = MLFlowTaskAPI.classify_tasks(df)

        df["split_seed"] = df["load_from_miniabsolut_split_seed"].copy()
        df["split_seed"].replace({"None": "42"}, inplace=True)

        return df

    @staticmethod
    def classify_tasks(df: pd.DataFrame) -> List[str]:
        tasks = []
        for i, row in df.iterrows():
            exp: str = row["experiment"]
            ag_neg: str = row["ag_neg"]
            if exp == "11":
                tasks.append("1v1")
            elif exp == "13":
                tasks.append("1v9")
            elif exp == "14":
                if ag_neg.split("_")[1] == "looser":
                    tasks.append("high_vs_looser")
                elif ag_neg.split("_")[1] == "95low":
                    tasks.append("high_vs_95low")
            else:
                raise ValueError(f"Experiment {exp} not recognized.")
        return tasks



def load_trainrest_from_miniabsolut(ag, base_path = None):

    def mask_sample_size_in_filename(fn: str) -> str:
        spl = fn.split("_")
        if all(i.isdigit() for i in spl[2]):
            fn_new_rgx = "_".join(spl[:2] + ["[0-9]*"] + spl[3:])
            return fn_new_rgx
        else:
            return fn

    if base_path is None:
        base_path = config.DATA_MINIABSOLUT / f"{ag}/energy_contributions"

    fp = list(base_path.glob(mask_sample_size_in_filename("high_train_15000_absolut_energy_contributions.tsv")))[0]
    df_high_train = pd.read_csv(fp, sep='\t', header=1)
    fp = list(base_path.glob(mask_sample_size_in_filename("high_rest_absolut_energy_contributions.tsv")))[0]
    df_high_rest = pd.read_csv(fp, sep='\t', header=1)

    fp = list(base_path.glob(mask_sample_size_in_filename("looserX_train_15000_absolut_energy_contributions.tsv")))[0]
    df_weak_train = pd.read_csv(fp, sep='\t', header=1)
    fp = list(base_path.glob(mask_sample_size_in_filename("looserX_rest_absolut_energy_contributions.tsv")))[0]
    df_weak_rest = pd.read_csv(fp, sep='\t', header=1)

    fp = list(base_path.glob(mask_sample_size_in_filename("95low_train_15000_absolut_energy_contributions.tsv")))[0]
    df_nonb_train = pd.read_csv(fp, sep='\t', header=1)
    fp = list(base_path.glob(mask_sample_size_in_filename("95low_rest_absolut_energy_contributions.tsv")))[0]
    df_nonb_rest = pd.read_csv(fp, sep='\t', header=1)

    df_high_train["binder_type"] = f"{ag}_high"
    df_high_rest["binder_type"] = f"{ag}_high"

    df_weak_train["binder_type"] = f"{ag}_looserX"
    df_weak_rest["binder_type"] = f"{ag}_looserX"

    df_nonb_train["binder_type"] = f"{ag}_95low"
    df_nonb_rest["binder_type"] = f"{ag}_95low"

    # Concatenate all
    df = pd.concat(
        [
            df_high_train,
            df_high_rest,
            df_weak_train,
            df_weak_rest,
            df_nonb_train,
            df_nonb_rest,
        ]
    )
    
    return df


def load_testrest_from_miniabsolut(ag, base_path = None):

    if base_path is None:
        base_path = config.DATA_MINIABSOLUT / f"{ag}/energy_contributions"

    df_high_test = pd.read_csv(base_path / "high_test_5000_absolut_energy_contributions.tsv", sep='\t', header=1)
    df_high_rest = pd.read_csv(base_path / "high_rest_absolut_energy_contributions.tsv", sep='\t', header=1)

    df_weak_test = pd.read_csv(base_path / "looserX_test_5000_absolut_energy_contributions.tsv", sep='\t', header=1)
    df_weak_rest = pd.read_csv(base_path / "looserX_rest_absolut_energy_contributions.tsv", sep='\t', header=1)

    df_nonb_test = pd.read_csv(base_path / "95low_test_5000_absolut_energy_contributions.tsv", sep='\t', header=1)
    df_nonb_rest = pd.read_csv(base_path / "95low_rest_absolut_energy_contributions.tsv", sep='\t', header=1)

    df_high_test["binder_type"] = f"{ag}_high"
    df_high_test["origin"] = "test"
    df_high_rest["binder_type"] = f"{ag}_high"
    df_high_rest["origin"] = "rest"

    df_weak_test["binder_type"] = f"{ag}_looserX"
    df_weak_test["origin"] = "test"
    df_weak_rest["binder_type"] = f"{ag}_looserX"
    df_weak_rest["origin"] = "rest"

    df_nonb_test["binder_type"] = f"{ag}_95low"
    df_nonb_test["origin"] = "test"
    df_nonb_rest["binder_type"] = f"{ag}_95low"
    df_nonb_rest["origin"] = "rest"

    # Concatenate all
    df = pd.concat(
        [
            df_high_test,
            df_high_rest,
            df_weak_test,
            df_weak_rest,
            df_nonb_test,
            df_nonb_rest,
        ]
    )
    
    return df
