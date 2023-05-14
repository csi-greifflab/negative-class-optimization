from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from NegativeClassOptimization import ml
from NegativeClassOptimization import utils
from NegativeClassOptimization import preprocessing
from NegativeClassOptimization import config
from NegativeClassOptimization import search

from dataclasses import dataclass
from typing import List


BEAM_B = 100
BEAM_N = 1000
save_dir = Path("data/BeamRules")



if __name__ == "__main__":

    # Load data
    df = utils.load_global_dataframe()

    ag_pos = "3VRL"
    ag_neg = "1ADQ"
    df = df.loc[df["Antigen"].isin([ag_pos, ag_neg])].copy()
    df = df.drop_duplicates(["Slide"])

    N = 20000
    df = df.sample(n=N, random_state=42)
    df = df.sample(frac=1, random_state=42)

    df_train = df.iloc[:int(N*0.8)]
    df_test = df.iloc[int(N*0.8):]

    df_train.to_csv(save_dir / "df_train.csv", index=False)


    # Build base motifs
    base_motifs = []
    for i in range(11):
        for aa in config.AMINOACID_ALPHABET:
            motif = "."*i + aa + "."*(10-i)
            motif: List[str] = list(motif)
            base_motifs.append(search.Motif(motif))

    # Evaluate base motifs
    df_motifs = search.evaluate_motifs(base_motifs, df_train, ag_pos)
    df_motifs["beam_round"] = 0

    # Beam search
    df_motifs_i = df_motifs.copy()
    df_motifs_i.to_csv(save_dir / f"beam_0.csv", index=False)
    for i in range(BEAM_N):
        print(f"Beam round {i}")
        motifs_promoted = df_motifs_i.head(BEAM_B)["motif"].tolist()
        motifs_promoted = [search.Motif.init_from_regex(m) for m in motifs_promoted]
        
        motifs_i = search.develop_motifs(motifs_promoted, base_motifs)
        
        df_motifs_i = search.evaluate_motifs(motifs_i, df_train, ag_pos)
        df_motifs_i["beam_round"] = i+1
        
        df_motifs_i.to_csv(save_dir / "beam_results" / f"beam_{i+1}.csv", index=False)
        # df_motifs = pd.concat([df_motifs, df_motifs_i], axis=0)
        # Can later continue from saved!
