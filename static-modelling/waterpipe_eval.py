
import os
import json
import numpy as np
import matplotlib.pyplot as plt

__all__ = [
    "compute_per_sample_metrics",
    "global_summary",
    "tolerance_coverage",
    "pipe_mape_heatmap",
    "run_analysis",
]

def _flatten_samples(x):
    # x shape: [n_demands, n_pipes]
    return x.reshape(-1)

def compute_per_sample_metrics(
    Q_pred: np.ndarray,
    Q_true: np.ndarray,
    leak_channel: int = -1,
    include_leak_in_system_metrics: bool = True,
    eps: float = 1e-8,
    small_flow: float = 0.0,
):
    """
    Compute per-(demand, leak_idx) metrics.

    Shapes:
      Q_pred, Q_true: [n_demands, n_pipes, n_pipes+1]
      Returns a dict of arrays with shape [n_demands, n_pipes] each.

    Args:
      leak_channel: which index along the last axis corresponds to the artificial leak.
      include_leak_in_system_metrics: whether the global system metrics should include the leak channel.
      eps: numerical stability.
      small_flow: if >0, ignore terms where |true| <= small_flow in relative metrics (prevents division noise).

    Returns:
      {
        "relL1": ...,
        "relL2": ...,
        "sMAPE": ...,
        "leak_abs_err": ...,
        "leak_rel_err": ...,
      }
    """
    assert Q_pred.shape == Q_true.shape, "Q_pred and Q_true must have the same shape"
    assert Q_pred.ndim == 3, "Expected [n_demands, n_pipes, n_pipes+1] tensors"

    n_demands, n_pipes, n_channels = Q_true.shape
    assert n_channels == n_pipes + 1, "Last dim should be n_pipes + 1"

    # Select channels for system-wide metrics
    if include_leak_in_system_metrics:
        sys_slice = slice(None)
    else:
        # exclude leak channel
        sys_slice = [i for i in range(n_channels) if i != (n_channels + leak_channel) % n_channels]

    # Errors
    E = Q_pred - Q_true

    # Masks to ignore tiny true flows in relative metrics if requested
    if small_flow > 0.0:
        mask = np.abs(Q_true) > small_flow
    else:
        mask = np.ones_like(Q_true, dtype=bool)

    # ---- Per-sample relative L1 (sum |e| / sum |true|) ----
    num_relL1 = np.sum(np.abs(E[:, :, sys_slice]) * mask[:, :, sys_slice], axis=2)
    den_relL1 = np.sum(np.abs(Q_true[:, :, sys_slice]) * mask[:, :, sys_slice], axis=2) + eps
    relL1 = num_relL1 / den_relL1

    # ---- Per-sample relative L2 (||e||2 / ||true||2) ----
    num_relL2 = np.sqrt(np.sum((E[:, :, sys_slice] ** 2) * mask[:, :, sys_slice], axis=2))
    den_relL2 = np.sqrt(np.sum((Q_true[:, :, sys_slice] ** 2) * mask[:, :, sys_slice], axis=2)) + eps
    relL2 = num_relL2 / den_relL2

    # ---- Per-sample sMAPE over channels ----
    # mean over channels of 2|e| / (|pred| + |true|)
    denom_smape = (np.abs(Q_pred) + np.abs(Q_true)) + eps
    smape_per_elem = 2.0 * np.abs(E) / denom_smape
    if isinstance(sys_slice, list):
        smape = np.mean(smape_per_elem[:, :, sys_slice], axis=2)
    else:
        smape = np.mean(smape_per_elem[:, :, sys_slice], axis=2)

    # ---- Leak-channel specific errors ----
    leak_idx = (n_channels + leak_channel) % n_channels
    leak_true = Q_true[:, :, leak_idx]
    leak_pred = Q_pred[:, :, leak_idx]
    leak_abs_err = np.abs(leak_pred - leak_true)
    leak_rel_err = leak_abs_err / (np.abs(leak_true) + eps)

    return {
        "relL1": relL1,
        "relL2": relL2,
        "sMAPE": smape,
        "leak_abs_err": leak_abs_err,
        "leak_rel_err": leak_rel_err,
    }


def _percentiles(x_flat: np.ndarray, ps=(50, 90, 95, 99)):
    return {f"p{p}": float(np.percentile(x_flat, p)) for p in ps}


def global_summary(per_sample_metric: np.ndarray):
    """
    Summarize a per-(demand, leak_idx) metric array of shape [n_demands, n_pipes].
    Returns mean/median and high-percentiles.
    """
    x = _flatten_samples(per_sample_metric)
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        **_percentiles(x, ps=(90, 95, 99)),
    }


def tolerance_coverage(per_sample_metric: np.ndarray, thresholds=(0.05, 0.10, 0.15)):
    """
    Fraction of samples whose metric <= each threshold.
    """
    x = _flatten_samples(per_sample_metric)
    return {f"<= {int(t*100)}%": float(np.mean(x <= t)) for t in thresholds}


def pipe_mape_heatmap(Q_pred: np.ndarray, Q_true: np.ndarray, leak_channel: int = -1, eps: float = 1e-8, small_flow: float = 0.0):
    """
    Mean Absolute Percentage Error per [leak_idx, pipe_idx], excluding the leak channel itself.
    Returns an array of shape [n_pipes, n_pipes].
    """
    assert Q_pred.shape == Q_true.shape and Q_true.ndim == 3
    n_demands, n_pipes, n_channels = Q_true.shape
    leak_idx = (n_channels + leak_channel) % n_channels
    pipe_channels = [i for i in range(n_channels) if i != leak_idx]

    # Absolute percentage error per element
    denom = np.abs(Q_true[:, :, pipe_channels]) + eps
    if small_flow > 0.0:
        m = np.abs(Q_true[:, :, pipe_channels]) > small_flow
    else:
        m = np.ones_like(denom, dtype=bool)

    ape = np.abs(Q_pred[:, :, pipe_channels] - Q_true[:, :, pipe_channels]) / denom
    ape = np.where(m, ape, 0.0)

    # Mean over demands -> shape [n_pipes (leak axis), n_pipes (pipe axis)]
    mape = np.mean(ape, axis=0)
    return mape


def _save_fig(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches="tight", dpi=160)
    plt.close()


def _plot_hist(x, title, xlabel, path):
    plt.figure()
    plt.hist(x.reshape(-1), bins=60)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    _save_fig(path)


def _plot_box_by_leak(x2d, title, ylabel, path):
    # x2d: [n_demands, n_pipes]
    plt.figure()
    plt.boxplot([x2d[:, i] for i in range(x2d.shape[1])], showfliers=False)
    plt.title(title)
    plt.xlabel("Leak index")
    plt.ylabel(ylabel)
    _save_fig(path)


def _plot_heatmap(mat, title, path):
    plt.figure()
    plt.imshow(mat, aspect="auto", interpolation="nearest")
    plt.title(title)
    plt.xlabel("Pipe index")
    plt.ylabel("Leak index")
    plt.colorbar(label="MAPE")
    _save_fig(path)


def run_analysis(
    Q_pred: np.ndarray,
    Q_true: np.ndarray,
    outdir: str = "pipe_eval",
    leak_channel: int = -1,
    include_leak_in_system_metrics: bool = True,
    eps: float = 1e-8,
    small_flow: float = 0.0,
    sample_demands: int = None,
    random_state: int = 0,
):
    """
    Turn-key analysis. Saves figures and a JSON/CSV summary in `outdir`.

    Args:
      sample_demands: if provided and smaller than n_demands, randomly subsample that many demand scenarios for speed.
      small_flow: ignore true flows with |true| <= small_flow in relative metrics.
    """
    os.makedirs(outdir, exist_ok=True)

    # Optional subsampling along the demand axis for speed
    if sample_demands is not None and sample_demands < Q_true.shape[0]:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(Q_true.shape[0], size=sample_demands, replace=False)
        Qp = Q_pred[idx]
        Qt = Q_true[idx]
    else:
        Qp, Qt = Q_pred, Q_true

    metrics = compute_per_sample_metrics(
        Qp, Qt,
        leak_channel=leak_channel,
        include_leak_in_system_metrics=include_leak_in_system_metrics,
        eps=eps,
        small_flow=small_flow,
    )

    # Summaries
    summaries = {
        name: global_summary(val) for name, val in metrics.items()
    }
    coverages = {
        "relL1": tolerance_coverage(metrics["relL1"]),
        "relL2": tolerance_coverage(metrics["relL2"]),
        "sMAPE": tolerance_coverage(metrics["sMAPE"]),
        "leak_rel_err": tolerance_coverage(metrics["leak_rel_err"]),
    }
    result = {"summaries": summaries, "coverage": coverages}

    # Save JSON
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(result, f, indent=2)

    # Heatmap
    mape_mat = pipe_mape_heatmap(Qp, Qt, leak_channel=leak_channel, eps=eps, small_flow=small_flow)

    # Plots (each metric gets its own chart)
    _plot_hist(metrics["relL1"], "Per‑scenario relative L1", "relL1", os.path.join(outdir, "relL1_hist.png"))
    _plot_hist(metrics["relL2"], "Per‑scenario relative L2", "relL2", os.path.join(outdir, "relL2_hist.png"))
    _plot_hist(metrics["sMAPE"], "Per‑scenario sMAPE", "sMAPE", os.path.join(outdir, "sMAPE_hist.png"))
    _plot_hist(metrics["leak_rel_err"], "Leak magnitude relative error", "rel. error", os.path.join(outdir, "leak_rel_err_hist.png"))

    _plot_box_by_leak(metrics["relL1"], "Relative L1 by leak index", "relL1", os.path.join(outdir, "relL1_box_by_leak.png"))
    _plot_box_by_leak(metrics["leak_rel_err"], "Leak relative error by leak index", "rel. error", os.path.join(outdir, "leak_rel_err_box_by_leak.png"))

    _plot_heatmap(mape_mat, "MAPE per [leak index, pipe index] (excluding leak channel)", os.path.join(outdir, "pipe_mape_heatmap.png"))

    # Save a CSV of mean MAPE per pipe (averaged over leak indices)
    mean_mape_per_pipe = mape_mat.mean(axis=0)
    np.savetxt(os.path.join(outdir, "mean_mape_per_pipe.csv"), mean_mape_per_pipe, delimiter=",", header="mean_mape_per_pipe", comments="")

    return {
        "outdir": outdir,
        "n_demands_used": Qp.shape[0],
        "n_pipes": Qp.shape[1],
        "summaries": summaries,
        "coverage": coverages,
        "artifacts": [
            "summary.json",
            "relL1_hist.png",
            "relL2_hist.png",
            "sMAPE_hist.png",
            "leak_rel_err_hist.png",
            "relL1_box_by_leak.png",
            "leak_rel_err_box_by_leak.png",
            "pipe_mape_heatmap.png",
            "mean_mape_per_pipe.csv",
        ],
    }
    


def select_and_plot_percentiles(
    Q_pred: np.ndarray,
    Q_true: np.ndarray,
    percentiles=(25, 50, 75, 100),   # use 100 or "max" for the worst
    leak_channel: int = -1,          # index of the artificial leak channel along the last axis
    eps: float = 1e-8,               # numerical stability for the denominator
    outdir: str | None = None,       # e.g., "pipe_eval_samples" to save PNGs; None = don't save
    prefix: str = "sample",          # filename prefix when saving
    show: bool = True                # call plt.show() for each plot
):
    """
    Select samples near specified percentiles of per-sample relative error and plot them.

    Shapes:
      Q_pred, Q_true: [n_demands, n_pipes, n_pipes+1] (last axis = pipes + leak)

    Relative error (your definition):
      rel_err[d, l] = mean(|Q_pred - Q_true| over channels) / (mean(|Q_true| over channels) + eps)

    Returns:
      A list of dicts for each selected sample:
        {
          "percentile": int,
          "rel_error": float,
          "demand_idx": int,
          "leak_idx": int,
          "flat_index": int,
          "figure_path": str | None
        }
    """
    assert Q_pred.shape == Q_true.shape and Q_true.ndim == 3, \
        "Q_pred and Q_true must be [n_demands, n_pipes, n_pipes+1]"

    n_demands, n_pipes, n_channels = Q_true.shape
    leak_idx = (n_channels + leak_channel) % n_channels

    # Per-sample relative error (your formula)
    rel_err = (np.abs(Q_pred - Q_true)).mean(axis=-1) / (np.abs(Q_true).mean(axis=-1) + eps)  # [n_demands, n_pipes]
    rel_flat = rel_err.reshape(-1)  # flatten over (demand, leak)

    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)

    # To avoid returning the exact same sample twice when percentiles coincide
    chosen = set()
    def nearest_unique_index(target_value: float) -> int:
        order = np.argsort(np.abs(rel_flat - target_value))
        for idx in order:
            if int(idx) not in chosen:
                chosen.add(int(idx))
                return int(idx)
        return int(order[0])

    results = []

    for p in percentiles:
        # Normalize percentile input
        if isinstance(p, str) and p.lower() in {"max", "worst", "100"}:
            p_val = 100.0
        else:
            p_val = float(p)

        # Pick sample for this percentile
        if p_val >= 100.0:
            idx = int(np.argmax(rel_flat))  # true worst
        else:
            target = float(np.percentile(rel_flat, p_val))
            idx = nearest_unique_index(target)

        d_idx = idx // n_pipes
        l_idx = idx % n_pipes
        err_val = float(rel_flat[idx])

        # Print a concise line
        print(f"{int(p_val):>3}th percentile — rel_err={err_val:.6f}  (demand={d_idx}, leak={l_idx})")

        # Extract series and plot
        q_true = Q_true[d_idx, l_idx, :]
        q_pred = Q_pred[d_idx, l_idx, :]
        xs = np.arange(n_channels)

        plt.figure(figsize=(10, 3))
        plt.plot(xs, q_pred, linestyle='--', label='prediction')
        plt.plot(xs, q_true, marker='o', label='ground truth')
        plt.axvline(leak_idx, linestyle=':', linewidth=1)  # mark leak channel
        plt.title(f"{int(p_val)}th percentile • rel_err={err_val:.4f} • demand={d_idx} • leak={l_idx}")
        plt.xlabel("Channel index (pipes + leak)")
        plt.ylabel("Flow")
        plt.legend()

        fig_path = None
        if outdir is not None:
            fig_name = f"{prefix}_{int(p_val)}pct_d{d_idx}_l{l_idx}.png"
            fig_path = os.path.join(outdir, fig_name)
            plt.savefig(fig_path, bbox_inches="tight", dpi=160)

        if show:
            plt.show()
        else:
            plt.close()

        results.append({
            "percentile": int(p_val),
            "rel_error": err_val,
            "demand_idx": d_idx,
            "leak_idx": l_idx,
            "flat_index": int(idx),
            "figure_path": fig_path,
        })

    return results

