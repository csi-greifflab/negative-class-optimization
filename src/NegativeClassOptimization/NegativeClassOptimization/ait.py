"""
Algorithmic Information Theory (AIT).
"""

from typing import List
import numpy as np
import pandas as pd
from lempel_ziv_complexity import lempel_ziv_complexity


def estimate_lzc(num_shuffles, slides):
    slides = slides[:]
    lzp_1u2_samples = []
    for _ in range(num_shuffles):
        np.random.shuffle(slides)
        lzp_1u2_i = lempel_ziv_complexity("_".join(slides))
        lzp_1u2_samples.append(lzp_1u2_i)
    lzp_1u2 = np.mean(lzp_1u2_samples)
    return lzp_1u2


def lzc_numchars(slides):
    return (11+1) * len(slides)


def estimate_lzc_union(num_shuffles, slides_1, slides_2):
    slides = slides_1 + slides_2
    # slides = list(set(slides))
    lzp_1u2 = estimate_lzc(num_shuffles, slides)
    return lzp_1u2


def algo_mutual_info(
        slides_1: List[str], 
        slides_2: List[str], 
        num_shuffles = 10,
        ):

    lzp_1u2 = estimate_lzc_union(num_shuffles, slides_1, slides_2)

    lzp_1 = estimate_lzc(num_shuffles, slides_1)
    lzp_2 = estimate_lzc(num_shuffles, slides_2)

    # Min or average?
    lzp = int(lzp_1u2 - min(lzp_1, lzp_2))

    return lzp