from pathlib import Path

import yaml

# ugly
src_config_filepath = Path(__file__)  # assumes specific local install


def adjust_filepaths(p):
    return (src_config_filepath / "../../../.." / p).resolve()


def read_yaml(path: Path) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


# TODO: add filepaths as part of yaml or other config
DATA_BASE_PATH = adjust_filepaths(Path("data"))
DATA_SLACK_1 = adjust_filepaths(Path("data/slack_1"))
DATA_SLACK_1_RAW_DIR = DATA_SLACK_1 / "raw"
DATA_SLACK_1_GLOBAL = DATA_SLACK_1 / "global/slack_1_global.tsv"
DATA_SLACK_1_PROCESSED_DIR = DATA_SLACK_1 / "processed"
# IMMUNE_ML_BASE_PATH = adjust_filepaths(Path("immuneML"))
# GLOBAL_DATASETS_DIR = adjust_filepaths(Path("data/globals"))
PARAMS_PATH = adjust_filepaths(Path("params.yaml"))

DATA_ABSOLUT_DIR = DATA_BASE_PATH / "Absolut/data"
DATA_ABSOLUT_DOI = DATA_BASE_PATH / "Absolut/toc_doi10.11582_2021.00063.csv"
DATA_ABSOLUT_DATASET3_AGLIST = (
    DATA_ABSOLUT_DIR / "Datasets3/nonRedundant_11mer-based/ListAntigens142.txt"
)
DATA_ABSOLUT_DATASET3_BINDINGMTX = (
    DATA_ABSOLUT_DIR / "Datasets3/nonRedundant_11mer-based/Treated142.txt"
)
DATA_ABSOLUT_PROCESSED_MULTICLASS_DIR = DATA_BASE_PATH / "Absolut/processed/multiclass"
DATA_ABSOLUT_PROCESSED_MULTILABEL_DIR = DATA_BASE_PATH / "Absolut/processed/multilabel"

DATA_MINIABSOLUT = DATA_BASE_PATH / "MiniAbsolut"
DATA_MINIABSOLUT_SPLITS = DATA_BASE_PATH / "MiniAbsolut_Splits"

DATA_SLACK_1_PARATOPES = (
    DATA_BASE_PATH / "Absolut/processed/paratope_epitope/Task4_Merged_Slice_ParaEpi.txt"
)
DATA_SLACK_1_RAWBINDINGSMURINE = (
    DATA_BASE_PATH / "Absolut/data/RawBindingsMurine/unzipped"
)
DATA_SLACK_1_RAWBINDINGS_PERCLASS_MURINE = (
    DATA_BASE_PATH / "Absolut/data/RawBindingsPerClassMurine"
)

DATA_XSTREME = DATA_BASE_PATH / "xstreme"

DATA_ML = DATA_BASE_PATH / "Frozen_MiniAbsolut_ML"
DATA_LINEAR_ML = DATA_BASE_PATH / "Frozen_MiniAbsolut_Linear_ML"

# Experimental data paths
DATA_EXPERIMENTAL_DIR = DATA_BASE_PATH / "Experimental_Datasets"
DATA_BRIJ_DIR = DATA_EXPERIMENTAL_DIR / "Brij_Dataset"
DATA_POREBSKI_DIR = DATA_EXPERIMENTAL_DIR / "Porebski_Dataset"
DATA_POREBSKI_AFFMAT = (
    DATA_POREBSKI_DIR / "fc079_fc080_her2_affmat_paired_translated.csv"
)
DATA_POREBSKI_MLLIB = (
    DATA_POREBSKI_DIR / "fc081_fc082_her2_ml_lib_paired_translated.csv"
)
DATA_POREBSKI_IL7 = (
        DATA_POREBSKI_DIR / "fc059_hs064_il7_paired_translated.csv"
)
DATA_POREBSKI_HEL_FACS = (
        DATA_POREBSKI_DIR / "fc069_fc070_r3_facs_paired_translated.csv"
)
DATA_POREBSKI_HEL_MACS = (
        DATA_POREBSKI_DIR / "fc071_hs065_r3_macs_paired_avg_translated.csv"
)

TMP_DIR = DATA_BASE_PATH / "tmp"
TMP_DIR.mkdir(exist_ok=True)

DATA_EMB_DIR = DATA_BASE_PATH / "embeddings"

SLIDE_AMINOACIDS = [
    "D",
    "S",
    "C",
    "I",
    "W",
    "P",
    "Y",
    "M",
    "V",
    "E",
    "G",
    "N",
    "A",
    "F",
    "Q",
    "K",
    "R",
    "H",
    "L",
    "T",
]

ANTIGENS = [
    "3VRL",
    "1NSN",
    "3RAJ",
    "5E94",
    "1H0D",
    "1WEJ",
    "1ADQ",
    "1FBI",
    "2YPV",
    "1OB1",
]

ANTIGEN_EPITOPES = [
    "1H0DE1", "1OB1E1", "1WEJE1"
]

ANTIGEN_TO_ANTIGEN_EPITOPES = {
    "1H0D": "1H0DE1", 
    "1OB1": "1OB1E1", 
    "1WEJ": "1WEJE1",
}

ANTIGENS_WCHAINS = [
    '1ADQ_A', 
    '1FBI_X',
    '1H0D_C',
    '1NSN_S',
    '1OB1_C',
    '1WEJ_F',
    '2YPV_A',
    '3RAJ_A',
    '3VRL_C',
    '5E94_G',
]

ANTIGENS_CLOSEDSET = ["1FBI", "1NSN", "1OB1", "1WEJ", "3VRL", "5E94"]
ANTIGENS_OPENSET = ["1ADQ", "1H0D", "2YPV", "3RAJ"]
ANTIGENS_EXPERIMENTAL = ["HR2B", "HR2P"]
ANTIGENS_WITH_EXPERIMENTAL = ANTIGENS + ANTIGENS_EXPERIMENTAL

NUM_CLOSED_ANTIGENS_ABSOLUT_DATASET3 = 102

GLOBAL_CDR3_LEN_DISTR = {
    15: 0.17861142857142856,
    14: 0.1782842857142857,
    16: 0.17553285714285713,
    17: 0.13130714285714284,
    18: 0.08532428571428571,
    13: 0.07291857142857143,
    12: 0.05155714285714286,
    19: 0.046964285714285715,
    11: 0.04499142857142857,
    20: 0.02091,
    21: 0.008224285714285713,
    22: 0.003164285714285714,
    23: 0.0012842857142857144,
    24: 0.00043142857142857143,
    25: 0.00016,
    28: 9.857142857142856e-05,
    26: 9e-05,
    27: 5.5714285714285715e-05,
    31: 1.8571428571428572e-05,
    30: 1.4285714285714285e-05,
    36: 1.1428571428571429e-05,
    29: 1e-05,
    40: 8.571428571428571e-06,
    32: 5.7142857142857145e-06,
    35: 5.7142857142857145e-06,
    39: 4.2857142857142855e-06,
    43: 4.2857142857142855e-06,
    33: 2.8571428571428573e-06,
    37: 2.8571428571428573e-06,
    34: 1.4285714285714286e-06,
}

AMINOACID_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")
SEED = 42

MLFLOW_TRACKING_URI = "http://0.0.0.0:5000"

# PARAMS: dict = read_yaml(PARAMS_PATH)

FARMHASH_MOD_10_VAL_MASK = 8
FARMHASH_MOD_10_TEST_MASK = 9
