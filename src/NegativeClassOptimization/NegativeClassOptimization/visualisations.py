"""
TODO: Move/add umap projections here.
"""
from typing import Dict, List, Optional, Tuple

import logomaker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics

from NegativeClassOptimization.ml import compute_pr_curve, compute_roc_curve

map_task_type_to_clean = {
    "1v1": "vs 1",
    "1v1_adapted": "vs 1 (adapted)",
    "1v1_epitope": "vs 1 (Epitopes)",
    "1v9": "vs 9",
    "1v9_adapted": "vs 9 (adapted)",
    "high_vs_randseq": "vs Randomized",
    "high_vs_randpos": "vs Shuffled Pos",
    "high_vs_looser": "vs Weak",
    "high_vs_95low": "vs Non-binder",
}


class FinalPlotParams:
    
    cmap_tasks = {
        'vs 9' : '#FF5733',
        'vs Non-binder':'#00A6ED',
        'vs Weak':'#FFC300',
        'vs 1': '#8B5F4D',
        'Randomized': '#00BFA0',
    }

    antigens_palette =['#008080','#FFA07A','#000080','#FFD700','#228B22','#FF69B4','#800080','#FF6347','#00FF00','#FF1493']
    ag_order = [
        "1FBI",
        "3VRL",
        "2YPV",
        "5E94",
        "1WEJ",
        "1OB1",
        "1NSN",
        "1H0D",
        "3RAJ",
        "1ADQ",
    ]

class PlotParams:
    """
    Class for storing plotting parameters for the NCO project.

    For source of Dutch Field color palette, see:
    https://www.heavy.ai/blog/12-color-palettes-for-telling-better-stories-with-your-data
    """

    # Cmaps
    cmap_tasks = [
        # Selection from Dutch Field
        "#00bfa0",  # Dark Green
        "#e6d800",  # Yellow
        "#0bb4ff",  # Blue
        "#e60049",  # Red
        "#ffa300",  # Orange
    ]
    cmap_tasks_no1v1 = [
        # Selection from Dutch Field
        "#00bfa0",  # Dark Green
        "#0bb4ff",  # Blue
        "#e60049",  # Red
    ]
    cmap_antigens = [
        # Extended Dutch Field (+ #EB4C00)
        "#e60049",
        "#0bb4ff",  # Blue
        "#50e991",
        "#e6d800",
        "#9b19f5",
        "#ffa300",
        "#dc0ab4",
        "#b3d4ff",
        "#00bfa0",  # Dark Green
        "#eb4c00",
    ]
    cmap_binders_classification = [
        # Selection from Dutch Field
        "#0bb4ff",  # Blue
        "#e60049",
        "#00bfa0",  # Dark Green
    ]
    # cmap_divergent = sns.diverging_palette(220, 20, as_cmap=True)
    # cmap_divergent = sns.color_palette("Spectral", as_cmap=True)
    cmap_divergent = [
        "#1984c5",
        "#22a7f0",
        "#63bff0",
        "#a7d5ed",
        "#e2e2e2",
        "#e1a692",
        "#de6e56",
        "#e14b31",
        "#c23728",
    ]
    ## Not nice
    # cmap_categorical_tableau = [
    #     # https://help.tableau.com/current/pro/desktop/en-us/formatting_create_custom_colors.htm
    #     "#eb912b",
    #     "#7099a5",
    #     "#c71f34",
    #     "#1d437d",
    #     "#e8762b",
    #     "#5b6591",
    #     "#59879b",
    # ]
    cmap_cat_cblind_tableau = [
        # https://jrnold.github.io/ggthemes/reference/tableau_color_pal.html
        "#1170aa",
        "#fc7d0b",
        "#a3acb9",
        "#57606c",
        "#5fa2ce",
        "#c85200",
        "#7b848f",
        "#a3cce9",
        "#ffbc79",
        "#c8d0d9",
    ]

    map_task_type_to_clean = map_task_type_to_clean

    # Orders
    order_tasks = ["high_vs_95low", "1v1", "1v9", "high_vs_looser"]
    order_tasks_clean = list(map(lambda x: map_task_type_to_clean[x], order_tasks))
    order_antigens = [
        "1FBI",
        "3VRL",
        "2YPV",
        "5E94",
        "1WEJ",
        "1OB1",
        "1NSN",
        "1H0D",
        "3RAJ",
        "1ADQ",
    ]


def plot_abs_logit_distr(
    open_metrics: dict,
    metadata: Optional[dict] = None,
) -> tuple:
    """Plots distribution of absolute logits from the
    metrics recorded during mlflow run from script 06.

    Args:
        open_metrics (dict): contains logits in a currently hard-coded way.
        metadata (dict): metadata regarding the run (data, model, etc.).

    Returns:
        (fig, axs)
    """
    df_hist = pd.DataFrame(
        data={
            "abs_logits": open_metrics["y_open_abs_logits"],
            "test_type": np.where(open_metrics["y_open_true"] == 1, "closed", "open"),
        }
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.histplot(
        data=df_hist,
        x="abs_logits",
        hue="test_type",
        stat="probability",
        ax=ax,
        common_norm=False,
        kde=True,
    )
    if metadata is not None:
        ax.set_title(
            "Absolute logit distribution per open and closed set evaluation.\n"
            f'NDB1({metadata.get("ag_pos")} vs {metadata.get("ag_neg")})\n'
            r"$N_{train} = $"
            f'{metadata.get("N_train")}\n'
            r"$N_{closed}$ = "
            f'{metadata.get("N_closed")}\n'
            r"$N_{open}$ = "
            f'{metadata.get("N_open")}'
        )
    else:
        ax.set_title(
            "Absolute logit distribution per open and closed set evaluation.\n"
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
        eval_metrics["open"]["y_open_true"], eval_metrics["open"]["y_open_abs_logits"]
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
    axs[0].set_xlim(0.0, 1.0)
    axs[0].set_ylim(0.0, 1.05)
    axs[0].set_xlabel("False Positive Rate")
    axs[0].set_ylabel("True Positive Rate")
    axs[0].set_title(
        "Open set ROC\n"
        f'NDB1({metadata.get("ag_pos")} vs {metadata.get("ag_neg")})\n'
        r"$N_{train} = $"
        f'{metadata.get("N_train")}\n'
        r"$N_{closed}$ = "
        f'{metadata.get("N_closed")}\n'
        r"$N_{open}$ = "
        f'{metadata.get("N_open")}'
    )
    axs[0].legend(loc="lower right")
    axs[0].grid()

    axs[1].scatter(
        thresholds_closed,
        np.abs(fpr_closed + tpr_closed - 1),
        label="Closed",
        color="darkred",
    )
    axs[1].scatter(
        thresholds_open,
        np.abs(fpr_open + tpr_open - 1),
        label=r"Open, $th_{opt}$=" f"{th_open_opt:.2f}",
        color="darkorange",
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
        eval_metrics["open"]["y_open_true"], eval_metrics["open"]["y_open_abs_logits"]
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
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Precision")
    ax.set_ylabel("Recall")
    ax.set_title(
        "Open set PR curve\n"
        f'NDB1({metadata.get("ag_pos")} vs {metadata.get("ag_neg")})\n'
        r"$N_{train} = $"
        f'{metadata.get("N_train")}\n'
        r"$N_{closed}$ = "
        f'{metadata.get("N_closed")}\n'
        r"$N_{open}$ = "
        f'{metadata.get("N_open")}'
    )
    ax.legend(loc="lower right")
    ax.grid()

    return (fig, ax)


def plot_confusion(
    cm: np.ndarray,
    cm_normed: np.ndarray,
    class_names: Optional[List[str]] = None,
):
    fig, axs = plt.subplots(ncols=2, figsize=(12, 5))

    metrics.ConfusionMatrixDisplay(cm_normed, display_labels=class_names).plot(
        ax=axs[0]
    )
    axs[0].set_title("Confusion matrix: normalized")

    metrics.ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=axs[1])
    axs[1].set_title("Confusion matrix: counts")
    return fig, axs


def plot_logo(df_attr: pd.DataFrame, allow_other_shape: bool = False, ax=None):
    if not allow_other_shape:
        assert df_attr.shape == (11, 20)
    if ax:
        return logomaker.Logo(
            df_attr,
            flip_below=False,
            fade_below=0.5,
            shade_below=0.5,
            figsize=(10, 6),
            ax=ax,
        )
    else:
        return logomaker.Logo(
            df_attr,
            flip_below=False,
            fade_below=0.5,
            shade_below=0.5,
            figsize=(10, 6),
        )


def plot_binding_distribution(figsize=(2 * 3.14, 3.14), dpi=600):
    """
    Plot a normal distribution function, with 3 colors under the area
    corresponding to top 1%, top 1-5%, and top 5-100% of the distribution.
    """
    x = np.linspace(-100, -5, 1000)
    y = 1 / np.sqrt(2 * np.pi) * np.exp(-((x + 77) ** 2) / 120)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    cmap = PlotParams.cmap_binders_classification

    ax.plot(x, y, color="black")
    ax.fill_between(x, y, where=x < -95, color=cmap[0], alpha=1)  # type: ignore
    ax.fill_between(x, y, where=(x >= -95) & (x <= -90), color=cmap[1], alpha=1)  # type: ignore
    ax.fill_between(x, y, where=x > -90, color=cmap[2], alpha=1)  # type: ignore
    ax.set_xlim(-100, -55)
    ax.set_ylim(0, 0.5)
    ax.set_xlabel("Binding Energy (kcal/mol)")

    # No y-axis, no spines
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # x-axis, keep numbers, no visible ticks
    ax.set_xticks([-100, -95, -90, -85, -80, -75, -70, -65, -60, -55])
    ax.set_xticklabels(
        ["-100", "-95", "-90", "-85", "-80", "-75", "-70", "-65", "-60", "-55"]
    )
    ax.tick_params(
        axis="x",
        which="both",
        bottom=False,
        top=False,
        labelbottom=True,
        labeltop=False,
    )

    # Add text, same y position, corresponding to each color
    fontsize = 8
    texts = ["0-1%\nBinders", "1-5%\nWeak", "5-100%\nNon-binders"]
    locations = [-97.5, -92.5, -77]

    for i in range(3):
        ax.text(
            locations[i],
            0.45,
            texts[i],
            color=cmap[i],  # type: ignore
            fontsize=fontsize,
            fontweight="bold",
            ha="center",
        )

    return fig, ax


def add_median_labels(
    ax,
    fmt=".2f",
    fontsize=6,
    y_level=None,
    y_offset=0.002,
    x_offset=0.0,
):
    """Add labels to boxplot medians.
    https://stackoverflow.com/questions/38649501/labeling-boxplot-in-seaborn-with-median-value
    """
    import matplotlib.patheffects as path_effects

    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == "PathPatch"]
    lines_per_box = int(len(lines) / len(boxes))
    for median in lines[4 : len(lines) : lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        text = ax.text(
            x + x_offset,
            y + y_offset if y_level is None else y_level,
            f"{value:{fmt}}",
            ha="center",
            va="center",
            # fontweight="bold",
            color="white",
            fontsize=fontsize,
            rotation=90,  # vertical orientation
        )
        # create median-colored border around white text for contrast
        text.set_path_effects(
            [
                path_effects.Stroke(linewidth=3, foreground=median.get_color()),
                path_effects.Normal(),
            ]
        )