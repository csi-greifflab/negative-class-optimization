from pathlib import Path


# ugly
src_config_filepath = Path(__file__)  # assumes specific local install
adjust_filepaths = lambda p: (src_config_filepath / "../../../.." / p).resolve()

DATA_BASE_PATH = adjust_filepaths(Path("data"))
DATA_SLACK_1 = adjust_filepaths(Path("data/slack_1"))
DATA_SLACK_1_GLOBAL = adjust_filepaths(Path("data/globals/slack_1_global.tsv"))
IMMUNE_ML_BASE_PATH = adjust_filepaths(Path("immuneML"))
GLOBAL_DATASETS_DIR = adjust_filepaths(Path("data/globals"))

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