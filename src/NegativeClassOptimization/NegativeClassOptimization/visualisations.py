"""
TODO: Move/add umap projections here.
"""
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

from NegativeClassOptimization.ml import compute_roc_curve, compute_pr_curve


def plot_abs_logit_distr(
    eval_metrics: dict, 
    metadata: dict
    ) -> tuple:
    """Plots distribution of absolute logits from the 
    metrics recorded during mlflow run from script 06.

    Args:
        eval_metrics (dict): contains logits in a currently hard-coded way.
        metadata (dict): metadata regarding the run (data, model, etc.).

    Returns:
        (fig, axs)
    """    
    df_hist = pd.DataFrame(data={
        "abs_logits": eval_metrics["open"]["y_open_abs_logits"],
        "test_type": np.where(eval_metrics["open"]["y_open_true"] == 1, "closed", "open")
    })
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.histplot(data=df_hist, x="abs_logits", hue="test_type", stat="probability", ax=ax, common_norm=False, kde=True)
    ax.set_title(
        "Absolute logit distribution per open and closed set evaluation.\n"
        f'NDB1({metadata.get("ag_pos")} vs {metadata.get("ag_neg")})\n'
        r'$N_{train} = $'f'{metadata.get("N_train")}\n'
        r'$N_{closed}$ = 'f'{metadata.get("N_closed")}\n'
        r'$N_{open}$ = 'f'{metadata.get("N_open")}'
    )
    ax.grid()
    return (fig, ax)


def plot_roc_open_and_closed_testsets(eval_metrics, metadata: dict):
    """Plot ROC plots for the open and closed test sets evaluations.

    Args:
        eval_metrics (dict): contains the necessary metrics.

    Returns:
        (fig, axs)
    """

    fpr_open, tpr_open, thresholds_open, th_open_opt = compute_roc_curve(
        eval_metrics["open"]["y_open_true"],
        eval_metrics["open"]["y_open_abs_logits"]
    )

    fpr_closed, tpr_closed, thresholds_closed, _ = compute_roc_curve(
        eval_metrics["closed"]["y_test_true"],
        eval_metrics["closed"]["y_test_logits"],
    )

    fig, axs = plt.subplots(ncols=2, figsize=(14, 7))
    lw = 2
    axs[0].plot(
        fpr_open,
        tpr_open,
        color="darkorange",
        lw=lw,
        label=f'Open, ROCAUC={eval_metrics["open"]["roc_auc_open"]:.2f}',
    )
    axs[0].plot(
        fpr_closed,
        tpr_closed,
        color="darkred",
        lw=lw,
        label=f'Closed, ROCAUC={eval_metrics["closed"]["roc_auc_closed"]:.2f}',
    )
    axs[0].plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    axs[0].set_xlim([0.0, 1.0])
    axs[0].set_ylim([0.0, 1.05])
    axs[0].set_xlabel("False Positive Rate")
    axs[0].set_ylabel("True Positive Rate")
    axs[0].set_title(
        "Open set ROC\n"
        f'NDB1({metadata.get("ag_pos")} vs {metadata.get("ag_neg")})\n'
        r'$N_{train} = $'f'{metadata.get("N_train")}\n'
        r'$N_{closed}$ = 'f'{metadata.get("N_closed")}\n'
        r'$N_{open}$ = 'f'{metadata.get("N_open")}'
    )
    axs[0].legend(loc="lower right")
    axs[0].grid()

    axs[1].scatter(
        thresholds_closed, 
        np.abs(fpr_closed + tpr_closed - 1), 
        label="Closed", 
        color="darkred"
    )
    axs[1].scatter(
        thresholds_open, 
        np.abs(fpr_open + tpr_open - 1), 
        label=r"Open, $th_{opt}$="f"{th_open_opt:.2f}", 
        color="darkorange"
    )
    axs[1].legend(loc="lower right")
    axs[1].set_xlabel("Threshold, logit | abs(logit)")
    axs[1].set_ylabel("$|FPR+TPR-1|$")
    axs[1].set_title("Optimal threshold for closed and open set test sets")
    axs[1].grid()
    return (fig, axs)


def plot_pr_open_and_closed_testsets(eval_metrics, metadata: dict):
    """Plot ROC plots for the open and closed test sets evaluations.

    Args:
        eval_metrics (dict): contains the necessary metrics.

    Returns:
        (fig, axs)
    """

    precision_open, recall_open, _, _ = compute_pr_curve(
        eval_metrics["open"]["y_open_true"],
        eval_metrics["open"]["y_open_abs_logits"]
    )

    precision_closed, recall_closed, _, _ = compute_pr_curve(
        eval_metrics["closed"]["y_test_true"],
        eval_metrics["closed"]["y_test_logits"],
    )

    fig, ax = plt.subplots(figsize=(14, 7))
    lw = 2
    ax.plot(
        precision_open,
        recall_open,
        color="darkorange",
        lw=lw,
        label=f'Open, Average precision={eval_metrics["open"]["avg_precision_open"]:.2f}',
    )
    ax.plot(
        precision_closed,
        recall_closed,
        color="darkred",
        lw=lw,
        label=f'Closed, Average precision={eval_metrics["closed"]["avg_precision_closed"]:.2f}',
    )
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Precision")
    ax.set_ylabel("Recall")
    ax.set_title(
        "Open set PR curve\n"
        f'NDB1({metadata.get("ag_pos")} vs {metadata.get("ag_neg")})\n'
        r'$N_{train} = $'f'{metadata.get("N_train")}\n'
        r'$N_{closed}$ = 'f'{metadata.get("N_closed")}\n'
        r'$N_{open}$ = 'f'{metadata.get("N_open")}'
    )
    ax.legend(loc="lower right")
    ax.grid()

    return (fig, ax)