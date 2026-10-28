"""Save raw forecast windows and plot them without loading the model again."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def save_window(path: Path, context: np.ndarray, truth: np.ndarray,
                prediction: np.ndarray, truth_standardized: np.ndarray,
                prediction_standardized: np.ndarray, metadata: dict) -> None:
    """All arrays use [time, channels]; standardized arrays match evaluation."""
    arrays = [context, truth, prediction, truth_standardized, prediction_standardized]
    if any(a.ndim != 2 or not np.isfinite(a).all() for a in arrays):
        raise ValueError("Forecast arrays must be finite two-dimensional arrays")
    if any(a.shape != truth.shape for a in arrays[2:]) or context.shape[1] != truth.shape[1]:
        raise ValueError("Forecast and context channel dimensions do not match")
    if truth.shape[1] != 3 * metadata["n_nodes"]:
        raise ValueError("Expected channel order [all x, all y, all z] with 3N channels")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Never pickle metadata: archives can be reopened with allow_pickle=False.
    np.savez_compressed(path, context=context, truth=truth, prediction=prediction,
                        truth_standardized=truth_standardized,
                        prediction_standardized=prediction_standardized,
                        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)))


def plot_window(path: Path, node_ids: list[int] | None = None,
                context_steps: int = 128) -> list[Path]:
    """Plot raw coordinates and error at each lead time, including persistence."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(path, allow_pickle=False) as archive:
        context, truth, pred = (archive[k] for k in ("context", "truth", "prediction"))
        true_std, pred_std = (archive[k] for k in ("truth_standardized", "prediction_standardized"))
        meta = json.loads(str(archive["metadata"].item()))
    n = meta["n_nodes"]
    if node_ids is None:
        node_ids = sorted({0, n // 2, n - 1})
    if not node_ids or any(i < 0 or i >= n for i in node_ids):
        raise ValueError(f"Plot node IDs must be between 0 and {n - 1}")
    dt = meta.get("dt")
    scale = dt if dt is not None else 1.0
    xlabel = "Time relative to forecast start (Lorenz units)" if dt is not None else "Steps relative to forecast start"
    past_length = min(context_steps, len(context))
    past = context[-past_length:]
    past_time = np.arange(-past_length, 0) * scale
    future_time = np.arange(len(truth)) * scale
    entire_time = np.concatenate([past_time, future_time])
    result = []
    for node in node_ids:
        fig, axes = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True)
        for component, ax in enumerate(axes):
            channel = component * n + node
            ax.plot(entire_time, np.r_[past[:, channel], truth[:, channel]],
                    color="#263238", linewidth=1.1, label="Ground truth")
            ax.plot(future_time, pred[:, channel], color="#d55e00", linewidth=1.2,
                    linestyle="--", label=meta["model"])
            ax.axvline(0, color="#607d8b", linewidth=0.8, linestyle=":")
            ax.set_ylabel("xyz"[component] + f"_{node}")
            ax.grid(alpha=0.18)
        axes[0].legend(loc="upper right", frameon=False)
        axes[0].set_title(f"{meta['model']} | N={n}, D={3*n} | {meta['system']} | window {meta['window_index']}")
        axes[-1].set_xlabel(xlabel)
        fig.tight_layout()
        out = path.with_name(path.stem + f"_node{node}.png")
        fig.savefig(out, dpi=160)
        plt.close(fig)
        result.append(out)

    # Both curves use the SAME context-derived per-channel scale. This is a
    # diagnostic plot, not an additional definition of a paper metric.
    std = np.nanstd(context, axis=0)
    std = np.where(std < 1e-10, 1e-10, std)
    model_rmse = np.sqrt(np.mean((pred_std - true_std) ** 2, axis=1))
    persistence_rmse = np.sqrt(np.mean(((context[-1] - truth) / std) ** 2, axis=1))
    raw_mae = np.mean(np.abs(pred - truth), axis=1)
    raw_rmse = np.sqrt(np.mean((pred - truth) ** 2, axis=1))
    curve_path = path.with_name(path.stem + "_errors.csv")
    with curve_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["lead_step", "relative_time", "raw_mae", "raw_rmse",
                         "standardized_rmse", "persistence_standardized_rmse"])
        writer.writerows(zip(range(1, len(truth) + 1), future_time, raw_mae, raw_rmse,
                             model_rmse, persistence_rmse))
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(np.arange(1, len(truth) + 1), model_rmse, label=meta["model"], color="#d55e00")
    ax.plot(np.arange(1, len(truth) + 1), persistence_rmse, label="Persistence", color="#607d8b", alpha=0.8)
    ax.set(xlabel="Forecast lead step", ylabel="RMSE (context-standardized)",
           title=f"All {3*n} channels | N={n} | window {meta['window_index']}")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = path.with_name(path.stem + "_errors.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    result.extend([curve_path, out])
    return result


def write_index(run_dir: Path) -> Path:
    """Create one entry point for browsing all saved figures and forecast data."""
    plan = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    lines = ["# High-dimensional zero-shot forecast results", "",
             "Original model weights; no fine-tuning. A graph is used for simulation, not passed to the model.", "",
             "The paper metrics use the original evaluation normalization. Time-series plots show raw Lorenz coordinates.", "",
             "Error plots use context-standardized per-step RMSE and a persistence reference; these are diagnostics.", "",
             "A completed run is not, by itself, a reproduction of the paper's numerical results.", "",
             f"Context: {plan['context_length']} steps; forecast: {plan['prediction_length']} steps.", "",
             "- [Run configuration](run_config.json)",
             "- [Aggregated metrics](summary/by_n_metrics.csv)", ""]
    for job in plan["jobs"]:
        folder = run_dir / job["model"] / f"N{job['N']}"
        lines.extend([f"## {job['architecture']} / N={job['N']} / D={3*job['N']}", ""])
        for filename in ["evaluation.log", "checkpoint_loading.json", "environment.json"]:
            file = folder / filename
            if file.exists():
                lines.append(f"- [{filename}]({file.relative_to(run_dir).as_posix()})")
        for file in sorted(folder.glob("forecasts/*/*.npz")):
            relative = file.relative_to(run_dir).as_posix()
            lines.extend(["", f"### {file.parent.name} / {file.stem}", "", f"[Raw forecast arrays]({relative})", ""])
            for png in sorted(file.parent.glob(file.stem + "_*.png")):
                lines.extend([f"![{png.stem}]({png.relative_to(run_dir).as_posix()})", ""])
    target = run_dir / "RESULTS.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target
