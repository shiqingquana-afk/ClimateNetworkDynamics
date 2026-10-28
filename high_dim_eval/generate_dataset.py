#!/usr/bin/env python3
"""Generate the original coupled Lorenz-63 benchmark without model dependencies."""
from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from datetime import datetime
from pathlib import Path

from common import csv_list, resolve_path, write_json


def stable_seed(base_seed: int, *parts: int) -> int:
    # Identical to the original experiment's _stable_seed.
    value = int(base_seed)
    for part in parts:
        value = (value * 1_000_003 + int(part) + 0x9E3779B9) % (2**31 - 1)
    return value


def can_generate_at(path: Path) -> bool:
    """Only a nonexistent directory or an empty checkout placeholder is writable."""
    if path.is_symlink():
        return False
    return not path.exists() or (path.is_dir() and all(p.name == ".gitkeep" and p.is_file() for p in path.iterdir()))


def generate(output_dir: Path, n_values: list[int], graph_seeds_per_n: int = 5,
             num_ics_per_graph: int = 20, n_steps: int = 4096,
             base_seed: int = 20260326) -> dict:
    if not n_values or min(n_values) < 5 or len(set(n_values)) != len(n_values):
        raise ValueError("Distinct N >= 5 required for edge probability p=4/(N-1)")
    if graph_seeds_per_n < 1 or num_ics_per_graph < 1 or n_steps < 2 or base_seed < 0:
        raise ValueError("Invalid generation counts, trajectory length, or seed")
    if not can_generate_at(output_dir):
        raise FileExistsError(f"Refusing to overwrite an existing dataset: {output_dir}")
    import numpy as np
    import pyarrow as pa
    from gluonts.dataset.arrow import ArrowWriter
    from coupled_lorenz63 import (CoupledLorenz63Config, generate_connected_er_graph,
                                  rk4_integrate, sample_initial_state)
    from lyapunov_scale import derive_dt_for_steps_per_lyapunov, estimate_lambda_max

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.generating-", dir=output_dir.parent))
    for subdir in ["graphs", "lyapunov"]:
        (staging / subdir).mkdir()
    started = time.monotonic()
    cfg = CoupledLorenz63Config(kappa=0.5)
    records, retries, checked = [], [], []
    manifest = {
        "generator": "high_dim_eval", "generation_complete": False,
        "split_dir": str(output_dir), "n_values": n_values,
        "graph_seeds_per_n": graph_seeds_per_n, "num_ics_per_graph": num_ics_per_graph,
        "n_steps": n_steps, "kappa": 0.5, "steps_per_lyapunov": 128,
        "base_seed": base_seed,
        "lyapunov_estimation": {"base_dt": 0.001, "warmup_steps": 256,
                                "estimate_steps": 1024, "lambda_retries": 3},
        "graph_sampling": "ER rejection up to 10000 attempts, then original connected fallback",
        "ic_retries": 8,
    }
    print(f"[GENERATE] staging={staging}", flush=True)
    write_json(staging / "manifest.json", manifest)
    try:
        for n in n_values:
            p = 4.0 / (n - 1)
            for graph_seed in range(graph_seeds_per_n):
                name = f"CoupledLorenz63_N{n}_k0p5_g{graph_seed}"
                print(f"[GENERATE] {name}: graph and Lyapunov calibration", flush=True)
                graph_rng_seed = stable_seed(base_seed, n, graph_seed, 11)
                graph = generate_connected_er_graph(n, p, graph_rng_seed)
                ref_state = sample_initial_state(n, np.random.default_rng(stable_seed(base_seed, n, graph_seed, 23)))
                lam = None
                for attempt in range(3):
                    try:
                        lam = estimate_lambda_max(graph, ref_state, cfg, warmup_steps=256,
                                                  estimate_steps=1024, base_dt=0.001, perturbation=1e-8,
                                                  random_seed=stable_seed(base_seed, n, graph_seed, attempt, 29))
                        if math.isfinite(lam) and lam > 1e-6:
                            break
                    except Exception as exc:
                        retries.append({"system": name, "stage": "lyapunov", "try": attempt, "error": str(exc)})
                if lam is None or not math.isfinite(lam) or lam <= 1e-6:
                    raise RuntimeError(f"Could not calibrate a positive Lyapunov exponent for {name}: {lam}")
                dt = derive_dt_for_steps_per_lyapunov(lam, 128)
                write_json(staging / "graphs" / f"N{n}_g{graph_seed}.json", {
                    "seed": graph_rng_seed, "num_nodes": n, "num_edges": graph.num_edges,
                    "edge_probability": p, "min_degree": float(graph.degrees.min()),
                    "max_degree": float(graph.degrees.max()), "mean_degree": float(graph.degrees.mean()),
                    "degrees": graph.degrees.astype(int).tolist(),
                    "edge_u": graph.edge_u.tolist(), "edge_v": graph.edge_v.tolist()})
                write_json(staging / "lyapunov" / f"N{n}_g{graph_seed}.json", {
                    "system": name, "n_nodes": n, "graph_seed": graph_seed,
                    "lambda_max_est": float(lam), "dt": dt, "steps_per_lyapunov": 128,
                    "identity_check": 128 * dt * lam, "warmup_steps": 256, "estimate_steps": 1024})
                (staging / name).mkdir()
                ic_retries_used = 0
                for ic in range(num_ics_per_graph):
                    for attempt in range(8):
                        init = sample_initial_state(n, np.random.default_rng(stable_seed(base_seed, n, graph_seed, ic, attempt, 41)))
                        trajectory = rk4_integrate(init, n_steps, dt, graph, cfg)
                        if trajectory.shape == (3 * n, n_steps) and np.isfinite(trajectory).all():
                            break
                        ic_retries_used += 1
                    else:
                        raise RuntimeError(f"Invalid trajectory after 8 attempts: {name}, IC={ic}")
                    relative = Path(name) / f"{ic}_T-{n_steps}.arrow"
                    # Same ArrowWriter and target representation as public process_trajs.
                    ArrowWriter(compression="lz4").write_to_file(
                        [{"start": datetime.now().strftime("%Y-%m-%d %H:%M"), "target": trajectory}],
                        staging / relative)
                    with pa.memory_map(str(staging / relative), "r") as file:
                        table = pa.ipc.open_file(file).read_all()
                    shape = table["target._np_shape"][0].as_py()
                    values = table["target"][0].values.to_numpy()
                    if table.num_rows != 1 or shape != [3 * n, n_steps] or not np.array_equal(values.reshape(shape), trajectory):
                        raise RuntimeError(f"Arrow round-trip failed: {relative}")
                    checked.append({"file": str(relative), "shape": shape, "finite": True, "exact_round_trip": True})
                    print(f"[GENERATE] {name}: saved {ic + 1}/{num_ics_per_graph}", flush=True)
                records.append({"system": name, "n_nodes": n, "graph_seed": graph_seed,
                                "n_trajectories_saved": num_ics_per_graph, "dt": dt,
                                "lambda_max_est": float(lam), "retries_used": ic_retries_used})
        expected_systems = len(n_values) * graph_seeds_per_n
        expected_files = expected_systems * num_ics_per_graph
        actual_files = len(list(staging.glob("CoupledLorenz63_*/*.arrow")))
        if len(records) != expected_systems or actual_files != expected_files:
            raise RuntimeError("Generated dataset counts do not match the requested counts")
        manifest.update({"generation_complete": True, "systems": records, "failures": [],
                         "retry_events": retries, "runtime_seconds": time.monotonic() - started,
                         "validation": {"expected_system_count": expected_systems, "actual_system_count": len(records),
                                        "expected_arrow_count": expected_files, "actual_arrow_count": actual_files,
                                        "all_file_checks": checked}})
        write_json(staging / "manifest.json", manifest)
        write_json(staging / "generation_report.json", manifest)
        write_json(staging / "COMPLETE.json", {"systems": len(records), "trajectories": actual_files})
        if not can_generate_at(output_dir):
            raise FileExistsError(f"Dataset destination became nonempty during generation: {output_dir}")
        if output_dir.exists():
            # A checkout may contain only .gitkeep; no dataset is removed.
            (output_dir / ".gitkeep").unlink(missing_ok=True)
            output_dir.rmdir()
        staging.rename(output_dir)
    except BaseException as exc:
        write_json(staging / "FAILED.json", {"error": str(exc), "type": type(exc).__name__})
        print(f"[FAILED] Incomplete generation retained at {staging}; target dataset was not published", flush=True)
        raise
    print(f"[GENERATED] {output_dir}: {len(records)} graphs, {actual_files} trajectories", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--data-dir", default="data/standard/test_coupled_lorenz63")
    parser.add_argument("--n-values", default="256,512")
    parser.add_argument("--graph-seeds-per-n", type=int, default=5)
    parser.add_argument("--num-ics-per-graph", type=int, default=20)
    parser.add_argument("--n-steps", type=int, default=4096)
    parser.add_argument("--base-seed", type=int, default=20260326)
    args = parser.parse_args()
    generate(resolve_path(args.data_dir, args.repo_root.expanduser().resolve()),
             [int(n) for n in csv_list(args.n_values)], args.graph_seeds_per_n,
             args.num_ics_per_graph, args.n_steps, args.base_seed)


if __name__ == "__main__":
    main()
