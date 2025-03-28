import os
import sys
from pathlib import Path
import copy

import warnings
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def partition_data_by_continious_var(target_df, continious_caharteristic_column: str):
    """ 
    Partitions data into three bins: Binder, Weak, and Non-binder, based on a continuous characteristic. 
    The bin sizes are approximately equal but may deviate to ensure no overlap in the continuous characteristic values.

    Args:
        target_df (pd.DataFrame): The input DataFrame containing the data to be partitioned.
        continuous_characteristic_column (str): The name of the column in the DataFrame representing the continuous characteristic used for partitioning.

    Returns:
        processed_df (pd.DataFrame): A DataFrame with the partitioned data and an additional column indicating bin assignment.
        binder_thr (float): The threshold value separating the Binder and Weak bins.
        weak_thr (float): The threshold value separating the Weak and Non-binder bins.
    """
    n = target_df.shape[0]//3
    processed_df = target_df.sort_values(continious_caharteristic_column, ascending=True)
    binder_thr= processed_df.iloc[n-1][continious_caharteristic_column]
    weak_thr = processed_df.iloc[2*n-1][continious_caharteristic_column]
    processed_df['Bin'] = processed_df[continious_caharteristic_column].apply(lambda x: 'Binder' if x <= binder_thr else 'Weak' if x <= weak_thr else 'Non-binder')

    return processed_df, binder_thr, weak_thr

def evaluate_partitioning(processed_df: pd.DataFrame, verbose: bool = True):
    """
    Evaluates the partitioning of the data into three bins: Binder, Weak, and Non-binder.

    Args:
        processed_df (pd.DataFrame): A DataFrame containing a 'Bin' column with values 'Binder', 'Weak', and 'Non-binder'.

    Returns:
        n (int): Total number of samples.
        n_binder (int): Number of samples in the 'Binder' bin.
        n_weak (int): Number of samples in the 'Weak' bin.
        n_non_binder (int): Number of samples in the 'Non-binder' bin.
        class_ratio (float): Ratio between the largest and smallest bin sizes.
        n_fair (int): The size of a partition if the data was divided equally.
    """
    # Total number of samples
    n = processed_df.shape[0]

    # Number of samples in each bin
    n_binder = processed_df[processed_df['Bin'] == 'Binder'].shape[0]
    n_weak = processed_df[processed_df['Bin'] == 'Weak'].shape[0]
    n_non_binder = processed_df[processed_df['Bin'] == 'Non-binder'].shape[0]

    # Calculate class ratio
    class_ratio = (max(n_binder, n_weak, n_non_binder)+1 )/ (min(n_binder, n_weak, n_non_binder)+1)

    # Fair partition size
    n_fair = n // 3

    # Print partitioning details for the user
    if verbose:
        print(f"Number of samples: {n},  Fair partition size: {n_fair}")
        print(f"Number of samples in Binder bin: {n_binder}, Number of samples in Weak bin: {n_weak}, Number of samples in Non-binder bin: {n_non_binder}")
        print(f"Largest bin sizes ratio: {class_ratio:.2f}")

    if class_ratio >=1.5:
        warnings.warn("The partitioning is skewed. The largest bin is more than 1.5 times larger than the smallest bin.", UserWarning)
        
    return n, n_binder, n_weak, n_non_binder, class_ratio, n_fair


def visualize_partition(prosessed_df, continious_caharteristic_column, binder_thr, weak_thr, plot_name=None):
    fig, ax= plt.subplots(figsize=(6,3))
    sns.kdeplot(prosessed_df[continious_caharteristic_column], ax=ax)
    ax.axvline(binder_thr, color='red', label='Binder threshold')
    ax.axvline(weak_thr, color='green', label='Weak threshold')
    ax.set_title(plot_name)
    plt.legend()
    plt.show()

def create_tasks_asin_publ(df_all_targets, target, TARGET_ID_COL, X_COL_NAME, class_size = None, random_state=42):

    """
    Created tasks as in the publication.Namely: 'vs Weak',' vs Non-binder',
    optionally 'vs all' if more than one target data used.
    Classes are created in a balanced way.

    Args:
    df_target_partitioned: dataframe with partitioned data for a target
    df_all_targets: dataframe with all targets data
    random_state: random state for reproducibility

    Also it requires global variables: 
    X_COL_NAME -  column name for object to be classified 
    TARGET_ID_COL - column name for target ID

    Returns:
    df_vs_weak: dataframe for 'vs Weak' task
    df_shuffled_weak: dataframe for 'vs Weak' task with shuffled labels

    df_vs_non_binder: dataframe for 'vs Non-binder' task
    df_shuffled_non_binder: dataframe for 'vs Non-binder' task with shuffled labels

    (optionally)
    df_vs_all: dataframe for 'vs all' task
    df_shuffled_all: dataframe for 'vs all' task with shuffled labels
   """


    df_target = df_all_targets[df_all_targets[TARGET_ID_COL] == target]

    if class_size:
        n_smallest = class_size
    else:
        n_smallest = df_target["Bin"].value_counts().min()

    
    binder = df_target.query('Bin == "Binder"').sample(n=n_smallest, random_state=random_state)
    binder['Y_binary'] = 1
    
    weak = df_target.query('Bin == "Weak"').sample(n=n_smallest, random_state=random_state)
    weak['Y_binary'] = 0
    df_vs_weak = pd.concat([binder, weak]).reset_index(drop=True)
    df_shuffled_weak = copy.deepcopy(df_vs_weak)
    df_shuffled_weak["Y_binary"] = df_shuffled_weak["Y_binary"].sample(frac=1, random_state=random_state).reset_index(drop=True)

    
    non_binder = df_target.query('Bin == "Non-binder"').sample(n=n_smallest, random_state=random_state)
    non_binder['Y_binary'] = 0
    df_vs_non_binder = pd.concat([binder, non_binder]).reset_index(drop=True)
    df_shuffled_non_binder = copy.deepcopy(df_vs_non_binder)
    df_shuffled_non_binder["Y_binary"] = df_shuffled_non_binder["Y_binary"].sample(frac=1, random_state=random_state).reset_index(drop=True)

    if df_all_targets[TARGET_ID_COL].nunique() > 1:
        vs_all = df_all_targets[df_all_targets[TARGET_ID_COL] != target]
        vs_all = vs_all[~vs_all[X_COL_NAME].isin(binder[X_COL_NAME])].sample(n=n_smallest, random_state=random_state)
        vs_all['Y_binary'] = 0
        df_vs_all = pd.concat([binder, vs_all]).reset_index(drop=True)
        df_shuffled_all = copy.deepcopy(df_vs_all)
        df_shuffled_all["Y_binary"] = df_shuffled_all["Y_binary"].sample(frac=1, random_state=random_state).reset_index(drop=True)
        return df_vs_weak, df_shuffled_weak, df_vs_non_binder,  df_shuffled_non_binder, df_vs_all,  df_shuffled_all
    else:
        return df_vs_weak, df_shuffled_weak, df_vs_non_binder, df_shuffled_non_binder
