"""
Title: Plotting utilities 
Author: @nehabinish
Created: 25/11/2023

Description:
Plotting utilities for demo analysis.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Apply custom matplotlib style 
# ---------------------------------------------------------------------
def apply_plot_style(style_path=None):
    """
    Apply a custom matplotlib style from a .mplstyle file.
    Args:
        style_path (str or Path): Path to the .mplstyle file. If None,
                                  defaults to 'custom_plotstyle.mplstyle' 
                                  in the same directory as this script.
    Returns:
        None
    """

    if style_path is None:
        try:
            base = Path(__file__).parent
        except NameError:
            # Running in a notebook
            base = Path.cwd()
        style_path = base / "custom_plotstyle.mplstyle"

    style_path = Path(style_path)

    if not style_path.exists():
        print(f"[WARN] Plot style not found: {style_path}. Using default.")
        return

    plt.style.use(style_path)
    print(f"[INFO] Loaded plot style: {style_path}")

# ---------------------------------------------------------------------
# Plot neural data
# ---------------------------------------------------------------------    
def plot_neural_data(data, ax=None, fig=None, regions=("R1", "R2"), time_key="time"):
    """
    Plot the mean and SEM of example neural data for specified brain regions.

    Args:
        data (dict): Dictionary containing trial-wise neural data.
                     Expected keys: region names (e.g., 'R1', 'R2') and 'time'.
                     Region data shape: [n_trials, n_channels, n_timepoints]
        ax (matplotlib.axes.Axes): Axis to plot on. If None, a new figure and axis are created.
        fig (matplotlib.figure.Figure): Figure to plot on. If None, a new figure is created.
        regions (tuple): Names of regions to plot. Defaults to ('R1', 'R2').
        time_key (str): Key for the time vector in `data`. Defaults to 'time'.

    Returns:
        None
    """
    if ax is None:
        apply_plot_style()
        fig, ax = plt.subplots()

    for region in regions:
        if region not in data:
            raise KeyError(f"Region '{region}' not found in data.")

        # Compute mean across trials and channels
        mean_signal = np.mean(data[region], axis=(0, 1))

        # Compute SEM (standard error of mean across channels)
        sem_signal = np.std(np.mean(data[region], axis=0), axis=0) / np.sqrt(data[region].shape[1])

        # Plot mean trace
        ax.plot(data[time_key], mean_signal, label=region, linewidth=2)

        # Shade SEM
        ax.fill_between(data[time_key], mean_signal - sem_signal, mean_signal + sem_signal,
                        alpha=0.3)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("HFA [z]")
    ax.legend(frameon=False)
    ax.grid(False)  

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------
# KDE + boxplot visualization
# ---------------------------------------------------------------------
def plot_kde(
    plot_data,
    ax=None, 
    fig=None,
    *,
    bxplt=True,
    plt_colors=None,
    title=None,
    xlabel=None,
    ylabel=None,
):
    """
    Plot KDE distributions with optional boxplots and statistics.

    Args:
        plot_data (dict): {condition: array-like}
        ax (matplotlib.axes.Axes): Axis to plot on. If None, a new figure and axis are created.
        fig (matplotlib.figure.Figure): Figure to plot on. If None, a new
        bxplt (bool): Add boxplot above KDE
        plt_colors (dict): {condition: color}
        title, xlabel, ylabel (str): Axis labels
    Returns:
        None
    """

    if ax is None:
        apply_plot_style()
        fig, ax = plt.subplots()

    if not isinstance(plot_data, dict) or len(plot_data) == 0:
        raise ValueError("plot_data must be a non-empty dictionary")

    conditions = list(plot_data.keys())

    if plt_colors is None:
        palette = sns.color_palette("Set2", len(conditions))
        plt_colors = dict(zip(conditions, palette))

    # --- KDE plots ---
    for cond in conditions:
        values = np.asarray(plot_data[cond])
        sns.kdeplot(
            values,
            ax=ax,
            color=plt_colors[cond],
            fill=True,
            alpha=0.4,
            linewidth=2,
            label=cond,
        )
        ax.axvline(values.mean(), color=plt_colors[cond], linestyle="--")

    if title:
        ax.set_title(title, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    # --- Boxplot inset ---
    if bxplt:
        df = pd.DataFrame({
            "condition": np.concatenate([[c] * len(plot_data[c]) for c in conditions]),
            "values": np.concatenate([plot_data[c] for c in conditions]),
        })

        ax_box = ax.inset_axes([0, 1.05, 1, 0.2])

        sns.boxplot(
            data=df,
            x="values",
            y="condition",
            hue="condition",
            orient="h",
            palette=plt_colors,
            showmeans=True,
            meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "white",
            "markersize": 6,
            },
            medianprops={
            "visible": False
            },
            showfliers=False,   # hides outliers
            legend=False,
            ax=ax_box,
        )

        ax_box.set_xlim(ax.get_xlim())
        ax_box.set_xlabel("")
        ax_box.set_ylabel("")
        ax_box.tick_params(labelbottom=False, bottom=False)
        sns.despine(ax=ax_box)

    # --- Legend ---
    ax.legend(frameon=False, prop={"style": "italic"})

    plt.tight_layout()
    return fig