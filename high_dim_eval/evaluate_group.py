#!/usr/bin/env python3
"""Evaluate one model and one subsystem count, using the public repository."""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
from pathlib import Path

from common import write_json


def remap_legacy_wavelet(state: dict) -> tuple[dict, list[dict]]:
    """Rename a historical module prefix without changing any tensor values."""
    prefix, replacement = "wavelet_analyzer.", "freq_analyzer."
    changes = []
    mapped = {}
    for key, value in state.items():
        new_key = replacement + key[len(prefix):] if key.startswith(prefix) else key
        if new_key != key:
            if new_key in state:
                raise ValueError(f"Ambiguous checkpoint contains both {key} and {new_key}")
            changes.append({"from": key, "to": new_key})
        mapped[new_key] = value
    return mapped, changes


def load_model_module(root: Path, architecture: str):
    """Select the public MODEL_NAME branch in memory; do not edit the checkout."""
    if architecture not in {"Nexus", "Panda"}:
        raise ValueError(f"Unsupported architecture: {architecture}")
    name = "scaleformer.scaleformer.scaleformer"
    if name in sys.modules:
        raise RuntimeError("Model module was imported before architecture selection")
    source = root / "scaleformer/scaleformer/scaleformer.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "MODEL_NAME" for t in node.targets)]
    if len(assignments) != 1:
        raise RuntimeError("Public MODEL_NAME selector changed; review compatibility before running")
    assignments[0].value = ast.copy_location(ast.Constant(architecture), assignments[0].value)
    ast.fix_missing_locations(tree)
    # Import the namespace parents before registering their child module.
    import importlib
    importlib.import_module("scaleformer.scaleformer")
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(tree, str(source), "exec"), module.__dict__)
    return module


def compute_metrics(evaluation, labels, predictions, plan: dict, dt: float | None):
    """Use released metric definitions and the coupled system's sampling clock."""
    names = plan["metrics"]
    output = {}
    for horizon in plan["horizons"]:
        true, pred = labels[:, :horizon], predictions[:, :horizon]
        values = {name: evaluation._format_ci(raw) for name, raw in
                  evaluation._compute_standard_metrics_per_sample(true, pred, names).items()}
        if "ME-LRw" in names:
            raw = evaluation._calculate_psd_metrics_raw(true, pred, "ME-LRw", nperseg=min(128, horizon))
            values["ME-LRw"] = evaluation._format_ci(raw)
        if "max_lyap_gt" in names or "max_lyap_pred" in names:
            if dt is None:
                raise ValueError("Lyapunov metrics require a calibrated dt")
            raw_true, raw_pred = evaluation._calculate_le_metrics_raw(
                true, pred, tau=dt, trajectory_len=32 if horizon <= 128 else 128)
            if "max_lyap_gt" in names:
                values["max_lyap_gt"] = evaluation._format_ci(raw_true)
            if "max_lyap_pred" in names:
                values["max_lyap_pred"] = evaluation._format_ci(raw_pred)
        if "gd_rmse" in names:
            values["gd_rmse"] = evaluation._format_rmse_ci(evaluation._calculate_gpdim_raw(true, pred))
        # The original coupled run computes KLD only at the longest horizon.
        if "kld" in names and horizon == plan["prediction_length"]:
            from kld import estimate_kld
            raw = estimate_kld(true, pred, n_mc=plan["kld_samples"], device=plan["device"])
            values["kld"] = evaluation._format_ci(raw)
        output[horizon] = values
    return output


def evaluate(plan: dict, job: dict) -> None:
    root = Path(plan["repo_root"])
    sys.path.insert(0, str(root))
    model_module = load_model_module(root, job["architecture"])
    import numpy as np
    import torch
    import transformers
    from gluonts.dataset.common import FileDataset
    from gluonts.itertools import batcher
    from scaleformer.scaleformer.dataset import TimeSeriesDataset
    from scaleformer.scaleformer.pipeline import PatchTSTPipeline
    from scaleformer.scaleformer import evaluation
    from scaleformer.utils import safe_standardize, save_evaluation_results

    import importlib.metadata
    versions = {}
    for package in ["torch", "transformers", "dysts", "gluonts", "kymatio", "numpy", "scipy", "pyarrow"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    write_json(Path(job["output_dir"]) / "environment.json", {"python": sys.version, "packages": versions})

    transformers.set_seed(job["seed"])
    if plan["device"].startswith("cuda:") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; choose a CUDA environment or --device cpu for a small check")
    load_kwargs = {}
    renamed = []
    # The distributed historical checkpoint predates the public module rename.
    # Inspect its header first; only load a replacement state dict if needed.
    weights = Path(job["checkpoint"]) / "model.safetensors"
    if job["architecture"] == "Nexus" and weights.is_file():
        from safetensors import safe_open
        from safetensors.torch import load_file
        with safe_open(str(weights), framework="pt", device="cpu") as f:
            needs_remap = any(key.startswith("wavelet_analyzer.") for key in f.keys())
        if needs_remap:
            state, renamed = remap_legacy_wavelet(load_file(str(weights), device="cpu"))
            load_kwargs["state_dict"] = state
    model, loading = model_module.PatchTSTForPrediction.from_pretrained(
        job["checkpoint"], torch_dtype=torch.float32,
        output_loading_info=True, local_files_only=True, **load_kwargs)
    loading["renamed_keys"] = renamed
    write_json(Path(job["output_dir"]) / "checkpoint_loading.json", loading)
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError("Checkpoint does not match the selected public model; see checkpoint_loading.json")
    model.to(torch.device(plan["device"]))
    model.eval()
    pipeline = PatchTSTPipeline(mode="predict", model=model)
    if model.config.context_length != plan["context_length"]:
        raise ValueError("Loaded model context length does not match the evaluation plan")
    print(f"Architecture={job['architecture']} N={job['N']} device={plan['device']} "
          f"seed={job['seed']} window_seed={plan['window_seed']}", flush=True)
    all_metrics = {h: {} for h in plan["horizons"]}
    dimensions, counts, windows = {}, {}, {}
    for system in job["systems"]:
        name = system["name"]
        datasets = [FileDataset(path=Path(path), freq="h", one_dim_target=False) for path in system["files"]]
        # Check actual Arrow contents, including dimensions and usable time length.
        for path, dataset in zip(system["files"], datasets):
            rows = list(dataset)
            if len(rows) != 1:
                raise ValueError(f"Expected one trajectory per Arrow file: {path}")
            target = np.asarray(rows[0]["target"])
            if target.ndim != 2 or target.shape[0] != 3 * job["N"]:
                raise ValueError(f"Expected [3N, T] for {path}, got {target.shape}")
            if target.shape[1] < plan["context_length"] + plan["prediction_length"]:
                raise ValueError(f"Trajectory too short for requested context + forecast: {path}")
            if not np.isfinite(target).all():
                raise ValueError(f"Nonfinite trajectory values: {path}")
        del target, rows
        stream = TimeSeriesDataset(
            datasets=datasets, probabilities=[1 / len(datasets)] * len(datasets),
            context_length=plan["context_length"], prediction_length=plan["prediction_length"],
            num_test_instances=plan["num_test_instances"], window_style=plan["window_style"],
            window_stride=plan["window_stride"], model_type="predict", mode="test",
            random_seed=plan["window_seed"])
        labels, predictions = [], []
        window_offset = 0
        with torch.inference_mode():
            for batch in batcher(stream, batch_size=job["batch_size"]):
                past = torch.stack([row["past_values"] for row in batch]).to(pipeline.device)
                future = torch.stack([row["future_values"] for row in batch]).numpy()
                pred = pipeline.predict(past, prediction_length=plan["prediction_length"],
                                        sliding_context=True, limit_prediction_length=False,
                                        verbose=False).transpose(0, 1).cpu().numpy()
                # The released pipeline can return a full native forecast block.
                pred = pred[..., :plan["prediction_length"], :]
                # Public pipeline returns [batch, parallel_samples, time, channels].
                if pred.ndim != 4 or pred.shape[1] != len(batch) or pred.shape[2:] != future.shape[1:]:
                    raise ValueError(f"Unexpected prediction shape {pred.shape}; target shape {future.shape}")
                context = past.cpu().numpy()
                raw_prediction = np.mean(pred, axis=0)
                pred = safe_standardize(pred, context=context[None], axis=2)
                future_standardized = safe_standardize(future, context=context, axis=1)
                if not np.isfinite(pred).all() or not np.isfinite(future_standardized).all():
                    raise RuntimeError(f"Nonfinite model forecasts for {name}")
                mean_prediction = np.mean(pred, axis=0)
                for local_index in range(len(batch)):
                    window_index = window_offset + local_index
                    if plan.get("save_trajectories") or window_index < plan.get("plot_windows", 0):
                        from forecast_outputs import save_window, plot_window
                        path = Path(job["output_dir"]) / "forecasts" / name / f"window_{window_index:04d}.npz"
                        save_window(path, context[local_index], future[local_index], raw_prediction[local_index],
                                    future_standardized[local_index], mean_prediction[local_index],
                                    {"model": job["architecture"], "system": name, "n_nodes": job["N"],
                                     "window_index": window_index, "dt": system["dt"],
                                     "context_length": plan["context_length"], "prediction_length": plan["prediction_length"],
                                     "channel_order": "x_0..x_N-1,y_0..y_N-1,z_0..z_N-1",
                                     "window_seed": plan["window_seed"]})
                        if window_index < plan.get("plot_windows", 0):
                            plot_window(path, plan.get("plot_nodes"), plan.get("plot_context_steps", 128))
                window_offset += len(batch)
                predictions.append(mean_prediction)
                labels.append(future_standardized)
        if not labels:
            raise RuntimeError(f"No evaluation windows produced for {name}")
        true, pred = np.concatenate(labels), np.concatenate(predictions)
        print(f"{name}: trajectories={len(datasets)}, windows={len(true)}, shape={true.shape}, dt={system['dt']}", flush=True)
        per_horizon = compute_metrics(evaluation, true, pred, plan, system["dt"])
        for horizon, metrics in per_horizon.items():
            all_metrics[horizon][name] = metrics
        dimensions[name], counts[name], windows[name] = 3 * job["N"], len(datasets), len(true)
        # Incremental writes preserve finished graphs if a later graph fails.
        save_evaluation_results(all_metrics, metrics_metadata={"system_dims": dimensions,
                                "n_system_samples": counts, "n_eval_windows": windows},
                                metrics_save_dir=job["output_dir"], metrics_fname="metrics", overwrite=True)
    write_json(Path(job["output_dir"]) / "COMPLETE.json", {"systems": list(dimensions), "n_eval_windows": windows})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--job-index", type=int, required=True)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    evaluate(plan, plan["jobs"][args.job_index])


if __name__ == "__main__":
    main()
