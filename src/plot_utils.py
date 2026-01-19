"""
Title: Plotting utilities 
Author: @nehabinish
Created: 25/11/2023

Description:
Plotting utilities for demo analysis.
"""

from ast import In
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
def plot_neural_data(data, ax=None, fig=None, regions=("R1", "R2"), time_key="time", region_colours=None):
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
        region_colours (dict or None): Optional dictionary mapping region names to colors.
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
        # Determine color
        colors = region_colours if region_colours else None
        # Plot mean trace
        ax.plot(data[time_key], mean_signal, label=region, linewidth=2, color=colors[region] if colors else None)
        ax.axvline(0, color='gray', linestyle='--', linewidth=1)
        # Add text annotations for task events
        ax.text(0.5, 0.88, 'Target', transform=ax.transAxes, fontsize=10, fontweight='medium', color='grey')
        # Shade SEM
        ax.fill_between(data[time_key], mean_signal - sem_signal, mean_signal + sem_signal,
                        alpha=0.3, color=colors[region] if colors else None)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("HFA [z]")
    ax.legend(frameon=False)
    ax.grid(False)  

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------
# Cumulative variance plotting
# ---------------------------------------------------------------------
def plot_cumulative_explained_variance(Results, dimsR1, dimsR2, fig=None, ax=None, roi_colours=None, n_dims=10):
    """
    Plot cumulative explained variance with SEM and mean latent dimensionality.

    Args:
        fig (matplotlib.figure.Figure): Figure to plot on. If None, a new figure is created.
        ax (matplotlib.axes.Axes): Axis to plot on. If None, a new axis is created.
        Results (dict): Should contain 'R1' and 'R2' with 'expvar' arrays.
        dimsR1 (array-like): Estimated latent dimensions for R1.
        dimsR2 (array-like): Estimated latent dimensions for R2.
        roi_colours (dict): Must have 'R1' and 'R2' colors.
        n_dims (int): Number of latent dimensions to plot (default 10).
    """

    # Check if fig and ax are provided
    if fig is None or ax is None:
        apply_plot_style()
        fig, ax = plt.subplots()

    # Default ROI colours
    if roi_colours is None:
        roi_colours = {'R1': '#1f77b4', 'R2': '#ff7f0e'}  # Default matplotlib colors

    # Stack and average explained variance
    explained_var_R1 = np.mean(np.vstack(Results['R1']['expvar']), axis=0)
    explained_var_R2 = np.mean(np.vstack(Results['R2']['expvar']), axis=0)

    # Cumulative explained variance in %
    cumsum_explained_var_R1 = np.cumsum(explained_var_R1) * 100
    cumsum_explained_var_R2 = np.cumsum(explained_var_R2) * 100

    # Compute SEM
    sem_explained_var_R1 = np.std(np.vstack(Results['R1']['expvar']), axis=0) / np.sqrt(len(Results['R1']['expvar']))
    sem_explained_var_R2 = np.std(np.vstack(Results['R2']['expvar']), axis=0) / np.sqrt(len(Results['R2']['expvar']))

    # Plot
    ax.plot(cumsum_explained_var_R1[:n_dims], marker='o', markersize=5, color=roi_colours['R1'], label='R1')
    ax.plot(cumsum_explained_var_R2[:n_dims], marker='o', markersize=5, color=roi_colours['R2'], label='R2')

    # SEM shading
    ax.fill_between(
        np.arange(n_dims),
        cumsum_explained_var_R1[:n_dims] - sem_explained_var_R1[:n_dims]*100,
        cumsum_explained_var_R1[:n_dims] + sem_explained_var_R1[:n_dims]*100,
        color=roi_colours['R1'], alpha=0.2
    )
    ax.fill_between(
        np.arange(n_dims),
        cumsum_explained_var_R2[:n_dims] - sem_explained_var_R2[:n_dims]*100,
        cumsum_explained_var_R2[:n_dims] + sem_explained_var_R2[:n_dims]*100,
        color=roi_colours['R2'], alpha=0.2
    )

    # Mean estimated dimensionality
    ax.axvline(np.mean(dimsR1), color=roi_colours['R1'], linestyle='--', linewidth=1, label='Mean dim R1')
    ax.axvline(np.mean(dimsR2), color=roi_colours['R2'], linestyle='--', linewidth=1, label='Mean dim R2')

    ax.set_ylabel('Cumulative Explained Variance [%]')
    ax.set_xlabel('# Latent Dimensions') 
    ax.legend(prop={'size': 9})  # smaller legend
    plt.tight_layout()
    plt.show()
    
    return fig

# ---------------------------------------------------------------------
# Communication subspace plotting
# ---------------------------------------------------------------------
def plot_xy_communication_subspace(XY_metrics, XY_numdims, xy_opt_dim, title=None):
    """
    Plots predictive performance with error bars and optimal dimensions (XY only).

    Args:
        XY_metrics (dict): Output from compute_performance() for XY.
            Expected keys: 'mean_cv', 'sem_cv', 'sd_cv','performance_full', 'error_full'
        XY_numdims (np.ndarray or list): Tested number of predictive dimensions.
        xy_opt_dim (int): Optimal number of dimensions to indicate with vertical line.
        title (str, optional): Plot title.
    """
    # Apply a consistent plotting style
    apply_plot_style()
    plt.figure()

    # XY performance curve
    plt.errorbar(
        XY_numdims,
        1 - XY_metrics['mean_cv'],
        yerr=XY_metrics['sd_cv'],
        fmt='o-', markersize=4, capsize=2, color='#752D72',
        label='XY subspace'
    )

    # Mark optimal number of dimensions
    plt.axvline(x=xy_opt_dim, color='#752D72', linestyle='--', label='Estimated optimal dim')

    # Full model performance marker
    plt.errorbar(
        -0.5,  # Offset on x-axis
        XY_metrics['performance_full'],
        yerr=XY_metrics['error_full'],
        fmt='^', markersize=6, capsize=2, color='#752D72',
        label='Full model'
    )

    plt.xlabel('Number of predictive dimensions')
    plt.ylabel('Predictive performance')
    if title:
        plt.title(title)
    plt.legend(prop={'size': 9})
    plt.tight_layout()
    plt.show()
