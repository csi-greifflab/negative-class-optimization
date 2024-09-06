from pathlib import Path
import numpy as np
import pandas as pd
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

import farmhash
import multiprocessing

import NegativeClassOptimization.utils as utils
import NegativeClassOptimization.config as config
from NegativeClassOptimization import preprocessing, pipelines, visualisations


# Measured concentrations in (M), every 45' for reaching equilibrium
map_intensity_to_conc = {
    "intensity 0": 0 + 1e-20,  # to later avoid division by 0
    "intensity 1": 100e-12,
    "intensity 2": 333e-12,
    "intensity 3": 1e-9,
    "intensity 4": 3.33e-9,
    "intensity 5": 10e-9,
    "intensity 6": 33.3e-9,
    "intensity 7": 100e-9,
    "intensity 8": 0,
    "intensity 9": 0,
    "intensity 10": 0,
    "intensity 11": 0,
    "intensity 12": 0,
    "intensity 13": 0,    
}

R = 8.31446261815324  # universal gas constant, J/(mol*K)
T = 293.15  # temperature, K (20C)


def estimate_kd_and_energies(df_g: pd.DataFrame) -> dict:
    cdr3 = df_g["read 1"].iloc[0]
    # Melt df_g to have one row per intensity
    df_g_melted = df_g.melt(
        id_vars=["read 1", "barcode"],
        value_vars=[f"intensity {i}" for i in range(14)],
        var_name="intensity",
        value_name="intensity_value"
    )
    df_g_melted["intensity_int"] = df_g_melted["intensity"].apply(lambda x: int(x.split(" ")[1]))
    df_g_melted["conc"] = df_g_melted["intensity"].map(map_intensity_to_conc)  # in M
    df_g_melted.dropna(inplace=True)

    F_min = df_g_melted.loc[df_g_melted["intensity"] == "intensity 0"]["intensity_value"].min()
    F_max = df_g_melted.loc[df_g_melted["intensity"] == "intensity 7"]["intensity_value"].max()
    
    def func(conc, Kd):
        """
        Function to fit to the data. Computes intensity.
        Based on the formula from the paper
        """
        return F_max / (1 + Kd / conc) + F_min

    # Fit the curve
    popt, pcov = curve_fit(func, df_g_melted["conc"], df_g_melted["intensity_value"])
    Kd_est = popt[0]
    E_est = R * T * np.log(Kd_est) / 1000 # in kJ/mol

    record = {
        "cdr3": cdr3,
        "Kd_est": Kd_est,
        "E_est": E_est,
        "F_min": F_min,
        "F_max": F_max,
        # Add average value for all intensities
        **df_g_melted.groupby("intensity").mean(numeric_only=True)["intensity_value"].to_dict(),
    }
    
    return record


if __name__ == "__main__":
    df_mllib = pd.read_csv(config.DATA_POREBSKI_MLLIB)
    df_mllib_clean = df_mllib[~df_mllib["read 1"].map(lambda seq: "*" in seq or "X" in seq)]
    df_mllib_clean = df_mllib_clean.groupby("read 1").filter(lambda x: len(x) >= 12)


    fp = "01b_df_kd.csv"
    if Path(fp).exists():
        df_kd = pd.read_csv(fp)
    else:
        with multiprocessing.Pool(processes=20) as pool:
            res = pool.starmap(
                estimate_kd_and_energies,
                [(df_g,) for _, df_g in df_mllib_clean.groupby("read 1")],
            )

        df_kd = pd.DataFrame(res)
        df_kd["affinity_est"] = 1 / df_kd["Kd_est"]  # Kaff = 1/Kd
        df_kd.to_csv(fp)


    # Classify binder_type based on intensity 8
    # comparison with the 2 thresholds
    th_non_weak = 122
    th_weak_high = 150
    df_kd["binder_type"] = "HR2P_95low"
    df_kd.loc[df_kd["intensity 8"] > th_weak_high, "binder_type"] = "HR2P_high"
    df_kd.loc[
        (df_kd["intensity 8"] > th_non_weak) & ((df_kd["intensity 8"] <= th_weak_high)),
        "binder_type",
    ] = "HR2P_looserX"


    ### Adapt df to fit as much as possible
    ### the format we used with Absolut data.
    # ID_slide_Variant	CDR3	Best	Slide	Energy	Structure	Source	Antigen
    df_kd["ID_slide_Variant"] = None
    df_kd["CDR3"] = df_kd["cdr3"]
    df_kd["Best"] = True
    df_kd["Slide"] = df_kd["cdr3"]
    df_kd["Energy"] = None
    df_kd["Structure"] = None
    df_kd["Source"] = df_kd["binder_type"]
    df_kd["Antigen"] = "HR2P"
    df_kd = df_kd[["ID_slide_Variant", "CDR3", "Best", "Slide", "Energy", "Structure", "Source", "Antigen", "binder_type"]].copy()

    # Check no duplicates
    assert df_kd["CDR3"].duplicated().sum() == 0, "There are duplicated CDR3s"


    ag = "HR2P"  # Her2 from Porebski dataset
    df_kd = df_kd.copy(Path(config.))

    #(from Aygul to Eugen;) As df_kd matching Absolut columns is used by me in 01s_*_experimetal.ipynb let's save it in data folder
    df_kd.to_csv(config.Path(config.DATA_BASE_PATH) / "01b_df_kd_absolut_format.csv")

    N_train = 2000
    N_test = 500

    make_splits = [False] + [True]*5
    seed = [None] + list(range(5))

    for MAKE_SPLITS, seed in zip(make_splits, seed):
        utils.build_dataset_into_Absolut(N_train, N_test, MAKE_SPLITS, seed, ag, df_kd)