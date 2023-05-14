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


from dataclasses import dataclass
from typing import List


@dataclass
class Motif:
    sequence: List[str]

    @staticmethod
    def init_from_regex(s: str):
        sequence = []
        for e in s.split("."):
            if e == "":
                sequence.append(".")
            elif e[0] == "[":
                sequence.append(e[1:-1])
            else:
                sequence.append(e)
        return Motif(sequence)

    def as_regex(self) -> str:
        motif = ""
        for e in self.sequence:
            if len(e) == 1:
                motif += e[0]
            else:
                motif += f"[{''.join(sorted(e))}]"
        return motif

    def __str__(self):
        return self.as_regex()
    
    def __eq__(self, other):
        for i in range(len(self.sequence)):
            if set(self.sequence[i]) != set(other.sequence[i]):
                return False
        return True
    
    def __hash__(self):
        return hash(self.as_regex())


def motif_merge(motif_1: Motif, motif_2: Motif) -> Motif:

    if motif_1 == motif_2:
        return motif_1

    motif = []
    for i, aa_1 in enumerate(motif_1.sequence):
        aa_2 = motif_2.sequence[i]
        
        if aa_1 == aa_2 == ".":
            motif.append(".")
        elif aa_1 == ".":
            motif.append(aa_2)
        elif aa_2 == ".":
            motif.append(aa_1)
        else:
            motif.append("".join(sorted(set(aa_1 + aa_2))))
    return Motif(motif)


def evaluate_motifs(motifs, df, ag_pos):
    records = []
    for i, motif in enumerate(motifs):
        mask = df["Slide"].str.contains(pat=motif.as_regex())
        M = mask.sum()
        M_p = df.loc[(mask) & (df["Antigen"] == ag_pos)].shape[0]
        M_n = M-M_p
        records.append({
        "motif": motif.as_regex(),
        "M": M,
        "M_p": M_p,
        "M_n": M_n,
    })
    df_motifs = pd.DataFrame.from_records(records)
    df_motifs["acc_int"] = df_motifs["M_p"] / df_motifs["M"]
    df_motifs["acc_int"] = df_motifs[["M_p", "M_n"]].apply(max, axis=1) / df_motifs["M"]
    df_motifs["acc"] = ( df_motifs["acc_int"] * df_motifs["M"] + 0.5 * (df.shape[0] - df_motifs["M"]) ) / df.shape[0]
    df_motifs = df_motifs.sort_values(by="acc", ascending=False)
    return df_motifs


def develop_motifs(motifs, base_motifs):
    new_motifs = []
    for motif in motifs:
        for base_motif in base_motifs:
            if motif != base_motif:
                merged = motif_merge(motif, base_motif)
                if merged not in new_motifs and merged not in motifs:
                    new_motifs.append(merged)
    new_motifs = list(set(new_motifs))
    return new_motifs
