import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from wntr.network import WaterNetworkModel

def plot_network(wn : WaterNetworkModel, node_size : int=18) -> None:
    """Plot a Graph of WNTR WaterNetworkModel """
    
    # Collect positions
    pos = {}
    for name in wn.node_name_list:
        node = wn.get_node(name)
        try:
            x, y = node.coordinates
        except AttributeError:
            x, y = node.x, node.y
        pos[name] = (x, y)

    # Collect edges
    xlines, ylines = [], []
    for lname in wn.link_name_list:
        link = wn.get_link(lname)
        x0, y0 = pos[link.start_node_name]
        x1, y1 = pos[link.end_node_name]
        xlines += [x0, x1, None]
        ylines += [y0, y1, None]

    # Draw edges (grey lines)
    _, ax = plt.subplots(figsize=(8, 6))
    ax.plot(xlines, ylines, linewidth=1.2, alpha=0.8, zorder=1, color="gray")

    # Split junctions into regular vs leak
    reservoirs = [n for n in wn.reservoir_name_list]
    regular_nodes = [n for n in wn.junction_name_list if not n.startswith("LEAK")]
    leak_nodes    = [n for n in wn.junction_name_list if n.startswith("LEAK")]

     # Plot reservoir nodes ('RESERVOIR')
    xs = [pos[n][0] for n in reservoirs]
    ys = [pos[n][1] for n in reservoirs]
    for xi, yi in zip(xs, ys):
        ax.text(xi, yi+50, 'RESERVOIR', ha='center', va='center', weight="bold")
        
    # Plot junctions (black dots)
    xs = [pos[n][0] for n in regular_nodes]
    ys = [pos[n][1] for n in regular_nodes]
    ax.scatter(xs, ys, s=node_size**2/10, alpha=0.95, zorder=2,
               color="black", marker="o", label="Regular nodes")

    # Plot leak nodes (grey dots)
    xs_leak = [pos[n][0] for n in leak_nodes]
    ys_leak = [pos[n][1] for n in leak_nodes]
    ax.scatter(xs_leak, ys_leak, s=node_size**2/10, alpha=0.95, zorder=2,
               color="gray", marker="o", label="Leak nodes")

    # Clean background
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    plt.show()


def plot_percentiles(
    Q_pred: np.ndarray,
    Q_true: np.ndarray,
    percentiles =(10, 50, 90, 100), # four informative percentiles
    font_scale: float = 1.4, # scale all text sizes
    figsize=(9.5, 7.2)
):
    """
    Compute NRMSD per scenario (across last axis), select four percentile samples,
    and plot them in a compact 2x2 grid with shared axis labels.
    """
    # scaled font sizes
    title_fs  = int(12 * font_scale)
    label_fs  = int(15 * font_scale)
    tick_fs   = int(10 * font_scale)
    legend_fs = int(12 * font_scale)

    n_channels = Q_true.shape[-1]

    # NRMSD across last axis per scenario
    rmse = np.sqrt(((Q_pred - Q_true) ** 2).mean(axis=-1)) 
    rng = Q_true.max(axis=-1) - Q_true.min(axis=-1)
    nrmsd = rmse / rng
    flat = nrmsd.reshape(-1)

    # Get indices for requested percentiles
    percentile_list = []
    for p in percentiles:
        perc_val = np.percentile(flat, p)
        idx = (np.abs(flat - perc_val)).argmin()
        val = float(flat[idx])
        percentile_list.append((p, val, idx))

    # Plot in a compact 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
    axes = axes.ravel()
    xs = np.arange(n_channels)
    tick_labels = ["" for _ in xs] 

    for k, (ax, (p_val, val, idx)) in enumerate(zip(axes, percentile_list)):
        
        q_t = Q_true.reshape(-1, Q_true.shape[-1])[idx, :]
        q_p = Q_pred.reshape(-1, Q_pred.shape[-1])[idx, :]

        ax.plot(xs, q_p, linestyle='--', label='PINN')
        ax.plot(xs, q_t, marker='o', label='Simulator')

        # Title
        ax.set_title(f"{p_val}th Percentile • NRMSD={100*val:.1f}%", fontsize=title_fs)

        # Ticks: bottom row shows x labels, all show y labels
        row, _ = divmod(k, 2)
        ax.set_xticks(xs)
        if row == 1:
            ax.set_xticklabels(tick_labels, fontsize=tick_fs)
        else:
            ax.set_xticklabels([])

        ax.tick_params(axis='y', labelsize=tick_fs, labelleft=True)

        # Lighten spines
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        # Legend only on the top-left panel
        if k == 0:
            ax.legend(frameon=False, fontsize=legend_fs, loc="best")

    # Shared axis labels
    try:
        fig.supxlabel("Pipe", fontsize=label_fs, x=0.55)
        fig.supylabel("Q", fontsize=label_fs)
    except Exception:
        fig.text(0.5, 0.02, "Channel index (pipes + leak)", ha="center", fontsize=label_fs)
        fig.text(0.02, 0.5, "Flow", va="center", rotation="vertical", fontsize=label_fs)

    fig.tight_layout()
    plt.subplots_adjust(wspace=0.18, hspace=0.28, bottom=0.10, left=0.10)

    plt.show()
    

    return

