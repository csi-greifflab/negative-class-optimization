# We generate "new" antigens in MiniAbsolut and MiniAbsolut splits, which are actually nothing else than epitopes from the antigens. We follow the same code pattern used in 01b, in which we integrated experimental data into MiniAbsolut.
# Plan: for each Miniabsolut antigen and for each sequence type (high, weak, nonb), we combine train_15 + rest, we select 15k according to epitope/hotspot, we evaluate that it makes sense to have extra splits (if enough data), and we generate a new set train_15* and rest* accordingly. Test set remains constant. Later subsets based on the epitope of the test set can be analysed.

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

import shutil

from NegativeClassOptimization import ml, datasets, pipelines
from NegativeClassOptimization import utils, config
from NegativeClassOptimization import preprocessing

from utils import load_trainrest_from_miniabsolut, load_testrest_from_miniabsolut


N_TRAIN = 15000
N_TEST = 5000


epitope_based_ags_map = {
    "1WEJ": ("1WEJE1", "F1G2K2K1N1G3I1T2W1K2T1Y1A1T1N1"),
    "1H0D": ("1H0DE1", "P1Q1G1R1I2S1S2S1F1Q2V1G1F1V1H1L1F1"),
    "1OB1": ("1OB1E1", "S1N1S1G1L3V1N2K1I2C2C1P1F2D2"),
}

for key, value in epitope_based_ags_map.items():

    print(key)

    ag = key
    ag_new, seqAGEpitope = value

    df = load_trainrest_from_miniabsolut(ag)

    # df.query("seqAGEpitope == @seqAGEpitope").groupby("binder_type").size()
    df = df.query("seqAGEpitope == @seqAGEpitope")
    df["Antigen"] = ag_new
    # df.head()

    assert all(df.groupby("binder_type").size() > N_TRAIN / 2)

    # Rebuild the dataframes
    df_high_train = df.loc[df["binder_type"] == f"{ag}_high"].sample(N_TRAIN / 2)
    df_high_rest = df.loc[(df["binder_type"] == f"{ag}_high") & (~df.index.isin(df_high_train.index))]
    df_weak_train = df.loc[df["binder_type"] == f"{ag}_looserX"].sample(N_TRAIN / 2)
    df_weak_rest = df.loc[(df["binder_type"] == f"{ag}_looserX") & (~df.index.isin(df_weak_train.index))]
    df_nonb_train = df.loc[df["binder_type"] == f"{ag}_95low"].sample(N_TRAIN / 2)
    df_nonb_rest = df.loc[(df["binder_type"] == f"{ag}_95low") & (~df.index.isin(df_nonb_train.index))]

    # Make the new directory in MiniAbsolut
    new_ag_dir = config.DATA_MINIABSOLUT / f"{ag_new}"
    new_ag_dir.mkdir(exist_ok=True)

    # Copy the test files from the original antigen
    for file in (config.DATA_MINIABSOLUT / f"{ag}").glob("*test*.tsv"):
        # Copy file to new antigen directory
        # using shutil.copyfile(src, dst)
        new_file = new_ag_dir / file.name
        shutil.copyfile(file, new_file)

    ## Save the new files in the main folder
    ## Columns for normal tsvs in MiniAbsolut
    cols_sel = ["ID_slide_Variant", "CDR3", "Best", "Slide", "Energy", "Structure", "Antigen"]
    df_high_train[cols_sel].to_csv(new_ag_dir / f"high_train_15000.tsv", sep='\t', index=False)
    df_high_rest[cols_sel].to_csv(new_ag_dir / f"high_rest.tsv", sep='\t', index=False)
    df_weak_train[cols_sel].to_csv(new_ag_dir / f"looserX_train_15000.tsv", sep='\t', index=False)
    df_weak_rest[cols_sel].to_csv(new_ag_dir / f"looserX_rest.tsv", sep='\t', index=False)
    df_nonb_train[cols_sel].to_csv(new_ag_dir / f"95low_train_15000.tsv", sep='\t', index=False)
    df_nonb_rest[cols_sel].to_csv(new_ag_dir / f"95low_rest.tsv", sep='\t', index=False)

    ###
    # Save the new files in the "*_energy_contributions" folder,
    # where other modules expect Absolut data regarding binding
    # energy.
    new_ag_energy_dir = new_ag_dir / "energy_contributions"
    new_ag_energy_dir.mkdir(exist_ok=True)

    # Copy the test files from the original antigen
    for file in (config.DATA_MINIABSOLUT / f"{ag}/energy_contributions").glob("*test*energy_contributions.tsv"):
        # Copy file to new antigen directory
        # using shutil.copyfile(src, dst)
        new_file = new_ag_energy_dir / file.name
        shutil.copyfile(file, new_file)

    df_high_train.to_csv(new_ag_energy_dir / f"high_train_15000_absolut_energy_contributions.tsv", sep='\t', index=False)
    df_high_rest.to_csv(new_ag_energy_dir / f"high_rest_absolut_energy_contributions.tsv", sep='\t', index=False)
    df_weak_train.to_csv(new_ag_energy_dir / f"looserX_train_15000_absolut_energy_contributions.tsv", sep='\t', index=False)
    df_weak_rest.to_csv(new_ag_energy_dir / f"looserX_rest_absolut_energy_contributions.tsv", sep='\t', index=False)
    df_nonb_train.to_csv(new_ag_energy_dir / f"95low_train_15000_absolut_energy_contributions.tsv", sep='\t', index=False) 
    df_nonb_rest.to_csv(new_ag_energy_dir / f"95low_rest_absolut_energy_contributions.tsv", sep='\t', index=False)


# Develop alternative test set evaluations. 
# Evaluate how many epitope-specific sequences in the positive datasets (outside of train!)
# of the 3 antigens. It is possible to gather 3000 epitope-specific seqs for pos and
# 3000 epitope-specific for negative parrts.

N_TEST_EPI = 3000

def build_test_df(ag, df, subset):
    num_seq_in_test = ((df["binder_type"] == f"{ag}_{subset}") & (df["origin"] == "test")).sum()
    if num_seq_in_test > N_TEST_EPI:
        df_subset_test = df.loc[(df["binder_type"] == f"{ag}_{subset}") & (df["origin"] == "test")].sample(N_TEST_EPI, random_state=42)
    else:
        df_subset_test = df.loc[(df["binder_type"] == f"{ag}_{subset}") & (df["origin"] == "test")].copy()
        num_seq_to_sample = N_TEST_EPI - num_seq_in_test
        df_subset_test = pd.concat([df_subset_test, df.loc[(df["binder_type"] == f"{ag}_{subset}") & (df["origin"] == "rest")].sample(num_seq_to_sample, random_state=42)])
    return df_subset_test


for key, value in epitope_based_ags_map.items():

    print(key)

    ag = key
    ag_new, seqAGEpitope = value
    new_ag_dir = config.DATA_MINIABSOLUT / f"{ag_new}"

    # Load all sequences from `energy_contribution` for the antigen (non-epitope specific)
    df = load_testrest_from_miniabsolut(ag)
    
    df["Antigen"] = ag_new

    # Make epitope-specific
    df = df.query("seqAGEpitope == @seqAGEpitope")

    assert all(df.groupby("binder_type").size() > N_TEST_EPI)

    # Rebuild the dataframes
    df_high_test = build_test_df(ag, df, "high")
    df_weak_test = build_test_df(ag, df, "looserX")
    df_nonb_test = build_test_df(ag, df, "95low")
    
    df_high_rest = df.loc[(df["binder_type"] == f"{ag}_high") & (~df.index.isin(df_high_test.index))]
    df_weak_rest = df.loc[(df["binder_type"] == f"{ag}_looserX") & (~df.index.isin(df_weak_test.index))]
    df_nonb_rest = df.loc[(df["binder_type"] == f"{ag}_95low") & (~df.index.isin(df_nonb_test.index))]


    ## Save the new files in the main folder
    ## Columns for normal tsvs in MiniAbsolut
    cols_sel = ["ID_slide_Variant", "CDR3", "Best", "Slide", "Energy", "Structure", "Antigen"]
    df_high_test[cols_sel].to_csv(new_ag_dir / f"highepi_test_3000.tsv", sep='\t', index=False)
    df_high_rest[cols_sel].to_csv(new_ag_dir / f"highepi_rest.tsv", sep='\t', index=False)
    df_weak_test[cols_sel].to_csv(new_ag_dir / f"looserXepi_test_3000.tsv", sep='\t', index=False)
    df_weak_rest[cols_sel].to_csv(new_ag_dir / f"looserXepi_rest.tsv", sep='\t', index=False)
    df_nonb_test[cols_sel].to_csv(new_ag_dir / f"95lowepi_test_3000.tsv", sep='\t', index=False)
    df_nonb_rest[cols_sel].to_csv(new_ag_dir / f"95lowepi_rest.tsv", sep='\t', index=False)
