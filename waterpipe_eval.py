import numpy as np
import matplotlib.pyplot as plt

def nrmsd_range_summary(
    Q_pred: np.ndarray,
    Q_true: np.ndarray,
    eps: float = 1e-8,
):
    """
    NRMSD per leak scenario (across the last axis) + summary stats.

    Shapes
    ------
    Q_pred, Q_true : [n_demands, n_pipes, n_pipes+1]
        Last axis = channels (all pipes + the leak channel).
        A "scenario" is each (demand_idx, leak_idx) pair; we aggregate over channels.

    Definition
    ----------
    RMSE(d,l) = sqrt( mean_over_channels( (Q_pred - Q_true)^2 ) )
    RANGE(d,l) = max(Q_true) - min(Q_true)  (over channels), or robust P_hi - P_lo
    NRMSD(d,l) = RMSE(d,l) / max(RANGE(d,l), eps)

    Parameters
    ----------
    use_robust_range : bool
        If True, use robust range (P_hi - P_lo) with 'robust_percentiles' instead of (max - min).
        This can reduce sensitivity to outliers on the channel axis.
    robust_percentiles : (lo, hi)
        Percentiles (e.g., (5, 95)) used when 'use_robust_range=True'.
    eps : float
        Small floor to avoid division by zero when the per-scenario range is ~0.

    Returns
    -------
    result : dict
        {
          "nrmsd": np.ndarray of shape [n_demands, n_pipes],  # per-scenario NRMSD
          "summary": {
             "mean": float, "median": float, "p90": float, "p95": float, "p99": float,
             "<= 5%": float, "<= 10%": float, "<= 15%": float   # fractions in [0,1]
          }
        }
    """
    assert Q_pred.shape == Q_true.shape and Q_true.ndim == 3, \
        "Q_pred and Q_true must both be [n_demands, n_pipes, n_pipes+1]"

    # RMSE across channels (last axis)
    rmse = np.sqrt(((Q_pred - Q_true) ** 2).mean(axis=-1))  # [n_demands, n_pipes]

    # Range (over channels) per scenario
    rng = Q_true.max(axis=-1) - Q_true.min(axis=-1)

    # Normalize with a small floor for stability
    den = np.maximum(rng, eps)
    nrmsd = rmse / den  # [n_demands, n_pipes]

    # Summaries across all scenarios
    flat = nrmsd.reshape(-1)
    summary = {
        "mean": float(np.mean(flat)),
        "median": float(np.median(flat)),
        "p90": float(np.percentile(flat, 90)),
        "p95": float(np.percentile(flat, 95)),
        "p99": float(np.percentile(flat, 99)),
        "<= 5%": float(np.mean(flat <= 0.05)),
        "<= 10%": float(np.mean(flat <= 0.10)),
        "<= 15%": float(np.mean(flat <= 0.15)),
    }

    return {"nrmsd": nrmsd, "summary": summary}


def plot_percentiles(
    Q_pred: np.ndarray,
    Q_true: np.ndarray,
    percentiles=(10, 50, 90, 100),   # four informative percentiles
    font_scale: float = 1.4,         # scale all text sizes (1.4 is nice and clear)
    figsize=(9.5, 7.2)               # slightly larger canvas for bigger text
):
    """
    Compute NRMSD per scenario (across last axis), select four percentile samples,
    and plot them in a compact 2x2 grid with shared axis labels.
    Shows y-ticks on all panels; only bottom row shows x-tick labels.
    Last x-tick label is set to 'virtual'. No vertical marker line.
    """
    assert Q_pred.shape == Q_true.shape and Q_true.ndim == 3, \
        "Q_pred and Q_true must be [n_demands, n_pipes, n_pipes+1]"
    assert len(percentiles) == 4, "This grid plot expects exactly 4 percentiles."

    # ----- font sizes (scaled) -----
    title_fs  = int(12 * font_scale)
    label_fs  = int(15 * font_scale)
    tick_fs   = int(10 * font_scale)
    legend_fs = int(12 * font_scale)

    n_demands, n_pipes, n_channels = Q_true.shape

    # --- NRMSD across last axis (channels), per scenario ---
    rmse = np.sqrt(((Q_pred - Q_true) ** 2).mean(axis=-1))   # [n_demands, n_pipes]
    rng = Q_true.max(axis=-1) - Q_true.min(axis=-1)

    #tau_val = np.percentile(rng, 10.0) if tau is None else float(tau)
    nrmsd = rmse / rng
    flat = nrmsd.reshape(-1)

    # Helper to avoid duplicates when percentiles coincide
    chosen = set()
    def nearest_unique_index(target_value: float) -> int:
        order = np.argsort(np.abs(flat - target_value))
        for idx in order:
            if int(idx) not in chosen:
                chosen.add(int(idx))
                return int(idx)
        return int(order[0])

    # Resolve indices for requested percentiles
    resolved = []
    for p in percentiles:
        p_val = 100.0 if (isinstance(p, str) and p.lower() in {"max","worst","100"}) else float(p)
        if p_val >= 100.0:
            idx = int(np.argmax(flat))
        else:
            target = float(np.percentile(flat, p_val))
            idx = nearest_unique_index(target)
        d_idx = idx // n_pipes
        l_idx = idx % n_pipes
        val = float(flat[idx])
        #print(f"{int(p_val):>3}th percentile — NRMSD={100*val:.6f}%  (demand={d_idx}, leak={l_idx})")
        resolved.append((int(p_val), val, d_idx, l_idx, idx))

    # --- Plot in a compact 2x2 grid with shared axis labels ---
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
    axes = axes.ravel()
    xs = np.arange(n_channels)
    tick_labels = ["" for i in xs] #[str(i) for i in xs]
    #tick_labels[-1] = ""

    for k, (ax, (p_val, val, d_idx, l_idx, _)) in enumerate(zip(axes, resolved)):
        q_t = Q_true[d_idx, l_idx, :]
        q_p = Q_pred[d_idx, l_idx, :]

        # Two lines only (no vertical marker)
        ax.plot(xs, q_p, linestyle='--', label='PINN')
        ax.plot(xs, q_t, marker='o', label='Simulator')

        # Title
        ax.set_title(f"{p_val}th Percentile • NRMSD={100*val:.1f}%", fontsize=title_fs)

        # Ticks: bottom row shows x labels; all show y labels
        row, _col = divmod(k, 2)
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
