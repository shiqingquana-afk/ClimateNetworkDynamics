#!/usr/bin/env python3
"""Portable evaluation launcher for a public ChaosNexus checkout."""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from common import (BATCH_SIZES, CHECKPOINT_NAMES, DEFAULT_METRICS, MODEL_NAMES, SYSTEM_RE,
                    SUPPORTED_METRICS, csv_list, discover_systems, resolve_path,
                    sha256, system_dt, trajectory_files, write_json, check_manifest_completeness)
from generate_dataset import can_generate_at


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    p.add_argument("--data-dir", default="data/standard/test_coupled_lorenz63")
    p.add_argument("--no-generate", action="store_true", help="Fail if the dataset is missing instead of generating it")
    p.add_argument("--generation-graphs-per-n", type=positive, default=5,
                   help="Graphs per N when generating missing data; independent of evaluation caps")
    p.add_argument("--generation-ics-per-graph", type=positive, default=20,
                   help="Initial conditions per graph when generating missing data")
    p.add_argument("--lyapunov-dir", help="Default: <data-dir>/lyapunov")
    p.add_argument("--output-dir", help="New run directory; relative paths are relative to repo root")
    p.add_argument("--models", default="full", help="full, panda, or full,panda")
    p.add_argument("--full-checkpoint", help="Default: checkpoints/Nexus1.0-base/checkpoint-final")
    p.add_argument("--panda-checkpoint", help="Default: checkpoints/PANDA-STANDARD/checkpoint-final")
    p.add_argument("--n-values", default="256,512", help="Subsystem counts, NOT state dimensions")
    p.add_argument("--device", default="cuda:0", help="CUDA recommended; cpu for small checks")
    p.add_argument("--batch-size", type=positive, help="Override batch size for every N; default N256=4, N512=2")
    p.add_argument("--context-length", type=positive, default=512)
    p.add_argument("--prediction-length", type=positive, default=512)
    p.add_argument("--num-test-instances", type=positive, default=6)
    p.add_argument("--systems-per-n", type=positive, help="Optional smoke-test cap for EACH N")
    p.add_argument("--samples-per-system", type=positive, help="Optional cap on Arrow files per graph")
    p.add_argument("--window-style", choices=["sampled", "rolling"], default="sampled")
    p.add_argument("--window-stride", type=positive, default=1)
    p.add_argument("--metrics", default=DEFAULT_METRICS)
    p.add_argument("--kld-samples", type=positive, default=256)
    p.add_argument("--seed", type=int, help="Default: checkpoint training_info seed, otherwise 99")
    p.add_argument("--window-seed", type=int, default=8097, help="Shared window sampling seed across models")
    p.add_argument("--plot-windows", type=int, default=0, help="Save and plot this many windows per graph; 0 disables plots")
    p.add_argument("--plot-nodes", help="Zero-based node IDs; default: first, middle, last")
    p.add_argument("--plot-context-steps", type=positive, default=128)
    p.add_argument("--save-trajectories", action="store_true", help="Save every forecast window, including windows not plotted")
    p.add_argument("--resume", action="store_true", help="Reuse an existing output and skip completed model/N groups")
    p.add_argument("--dry-run", action="store_true", help="Validate paths and print plan without imports, inference, or writes")
    return p


def make_plan(args: argparse.Namespace, allow_missing_data: bool = False) -> dict:
    root = args.repo_root.expanduser().resolve()
    sources = ["scaleformer/scaleformer/scaleformer.py", "scaleformer/scaleformer/pipeline.py",
               "scaleformer/scaleformer/evaluation.py", "scaleformer/scaleformer/dataset.py"]
    for name in sources:
        if not (root / name).is_file():
            raise FileNotFoundError(f"Expected public ChaosNexus source: {root / name}")
    models = csv_list(args.models)
    if not models or set(models) - MODEL_NAMES.keys():
        raise ValueError("--models must select full and/or panda")
    n_values = [int(n) for n in csv_list(args.n_values)]
    if not n_values or min(n_values) < 2:
        raise ValueError("--n-values must contain subsystem counts >= 2")
    if args.plot_windows < 0:
        raise ValueError("--plot-windows must be nonnegative")
    plot_nodes = [int(v) for v in csv_list(args.plot_nodes)] if args.plot_nodes else None
    if plot_nodes is not None and any(v < 0 or v >= min(n_values) for v in plot_nodes):
        raise ValueError("--plot-nodes must be valid zero-based node IDs for every requested N")
    if args.resume and not args.output_dir:
        raise ValueError("--resume requires the original --output-dir")
    if args.window_seed < 0 or (args.seed is not None and args.seed < 0):
        raise ValueError("Seeds must be nonnegative")
    if not re.fullmatch(r"cpu|cuda:\d+", args.device):
        raise ValueError("--device must be cpu or cuda:<index>")
    metrics = csv_list(args.metrics)
    if not metrics or set(metrics) - SUPPORTED_METRICS:
        raise ValueError(f"Supported metrics: {sorted(SUPPORTED_METRICS)}")
    data_dir = resolve_path(args.data_dir, root)
    lyap_dir = resolve_path(args.lyapunov_dir, root) if args.lyapunov_dir else data_dir / "lyapunov"
    generation = None
    if can_generate_at(data_dir) and allow_missing_data and not args.no_generate:
        if min(n_values) < 5:
            raise ValueError("Automatic generation requires N >= 5 for p=4/(N-1)")
        if args.context_length + args.prediction_length > 4096:
            raise ValueError("Automatic generation produces 4096 steps; requested context + forecast is longer")
        if args.lyapunov_dir and lyap_dir != data_dir / "lyapunov":
            raise ValueError("Automatic generation writes dt to <data-dir>/lyapunov; omit --lyapunov-dir")
        systems = {n: [] for n in n_values}
        generation = {"n_values": n_values, "graph_seeds_per_n": args.generation_graphs_per_n,
                      "num_ics_per_graph": args.generation_ics_per_graph, "n_steps": 4096,
                      "base_seed": 20260326}
    else:
        check_manifest_completeness(data_dir, n_values)
        groups = discover_systems(data_dir, n_values, args.systems_per_n)
        need_dt = bool(set(metrics) & {"max_lyap_gt", "max_lyap_pred"})
        def read_dt(name):
            if not (need_dt or args.plot_windows or args.save_trajectories):
                return None
            match = SYSTEM_RE.fullmatch(name)
            if need_dt or (lyap_dir / f"N{match['n']}_g{match['seed']}.json").is_file():
                return system_dt(name, data_dir, lyap_dir)
            manifest_path = data_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
            if any(row.get("system") == name and row.get("dt") is not None for row in manifest.get("systems", [])):
                return system_dt(name, data_dir, lyap_dir)
            return None
        systems = {
            n: [{"name": d.name, "files": [str(f) for f in trajectory_files(d, args.samples_per_system)],
                 "dt": read_dt(d.name)} for d in dirs]
            for n, dirs in groups.items()
        }
    output = resolve_path(args.output_dir, root) if args.output_dir else root / "eval_results" / "high_dim_coupled" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if output.exists() and not args.resume:
        raise FileExistsError(f"Output already exists; choose a NEW --output-dir: {output}")
    if args.resume and not (output / "run_config.json").is_file():
        raise FileNotFoundError(f"Resume needs an existing run_config.json in {output}")
    horizons = list(range(64, args.prediction_length + 1, 64))
    if args.prediction_length not in horizons:
        horizons.append(args.prediction_length)
    jobs = []
    for model in models:
        checkpoint = resolve_path(getattr(args, f"{model}_checkpoint") or f"checkpoints/{CHECKPOINT_NAMES[model]}/checkpoint-final", root)
        config_file = checkpoint / "config.json"
        if not config_file.is_file() or not any(checkpoint.glob("*.safetensors")) and not any(checkpoint.glob("pytorch_model*.bin")):
            raise FileNotFoundError(f"Checkpoint requires config.json and model weights: {checkpoint}")
        config = json.loads(config_file.read_text(encoding="utf-8"))
        if config.get("context_length") != args.context_length:
            raise ValueError(f"{model} checkpoint context_length={config.get('context_length')} differs from --context-length={args.context_length}")
        train_info = checkpoint / "training_info.json"
        info = json.loads(train_info.read_text(encoding="utf-8")) if train_info.is_file() else {}
        train = info.get("train_config") or info.get("training_config") or {}
        seed = args.seed if args.seed is not None else train.get("seed", 99)
        for n in n_values:
            jobs.append({"model": model, "architecture": MODEL_NAMES[model], "N": n,
                         "checkpoint": str(checkpoint), "checkpoint_config_sha256": sha256(config_file),
                         "seed": seed, "systems": systems[n], "batch_size": args.batch_size or BATCH_SIZES.get(n, 1),
                         "output_dir": str(output / model / f"N{n}")})
    return {"created_at": datetime.now(timezone.utc).isoformat(), "repo_root": str(root),
            "source_sha256": {name: sha256(root / name) for name in sources},
            "data_dir": str(data_dir), "output_dir": str(output), "metrics": metrics,
            "data_generation": generation,
            "device": args.device, "context_length": args.context_length,
            "prediction_length": args.prediction_length, "horizons": horizons,
            "num_test_instances": args.num_test_instances, "window_style": args.window_style,
            "window_stride": args.window_stride, "window_seed": args.window_seed,
            "plot_windows": args.plot_windows, "plot_nodes": plot_nodes,
            "plot_context_steps": args.plot_context_steps, "save_trajectories": args.save_trajectories,
            "kld_samples": args.kld_samples,
            "kld_protocol": {"estimator": "isotropic_gmm_logsumexp", "variance": 1.0,
                             "seeds": [0, 1, 2, 3, 4], "chunk_size": 64, "longest_horizon_only": True},
            "jobs": jobs}


def validate_resume(plan: dict, previous: dict) -> None:
    # Generation is already complete on resume; its original recipe remains in
    # the saved plan. Every evaluation setting and input path must still match.
    ignored = {"created_at", "data_generation"}
    changed = [key for key in set(plan) | set(previous)
               if key not in ignored and plan.get(key) != previous.get(key)]
    if changed:
        raise ValueError(f"Resume settings or sources changed: {', '.join(sorted(changed))}. Use the original settings or a new output directory.")


def main() -> None:
    args = parser().parse_args()
    # Validate repository, checkpoint, and output before expensive data generation.
    plan = make_plan(args, allow_missing_data=True)
    if args.resume:
        previous = json.loads((Path(plan["output_dir"]) / "run_config.json").read_text(encoding="utf-8"))
        validate_resume(plan, previous)
        plan = previous
    if plan["data_generation"] and not args.resume:
        generation = plan["data_generation"]
        print(f"Dataset missing: {plan['data_dir']}\n"
              f"Will generate N={generation['n_values']}, {generation['graph_seeds_per_n']} graphs per N, "
              f"{generation['num_ics_per_graph']} trajectories per graph, 4096 steps.", flush=True)
        if args.dry_run:
            print("DRY RUN OK: generation planned; no data generated and no files written.")
            return
        command = [sys.executable, str(Path(__file__).resolve().parent / "generate_dataset.py"),
                   "--repo-root", plan["repo_root"], "--data-dir", plan["data_dir"],
                   "--n-values", ",".join(map(str, generation["n_values"])),
                   "--graph-seeds-per-n", str(generation["graph_seeds_per_n"]),
                   "--num-ics-per-graph", str(generation["num_ics_per_graph"])]
        subprocess.run(command, check=True)
        plan = make_plan(args)
        plan["data_generation"] = generation
    print(f"Repository: {plan['repo_root']}\nDataset: {plan['data_dir']}\nOutput: {plan['output_dir']}", flush=True)
    for job in plan["jobs"]:
        print(f"{job['model']}: N={job['N']}, D={3*job['N']}, graphs={len(job['systems'])}, "
              f"trajectories={sum(len(s['files']) for s in job['systems'])}, batch={job['batch_size']}, seed={job['seed']}", flush=True)
    if args.dry_run:
        print("DRY RUN OK: paths/configuration checked; no files written or model inference performed.")
        return
    output = Path(plan["output_dir"])
    output.mkdir(parents=True, exist_ok=args.resume)
    plan_path = output / "run_config.json"
    if not args.resume:
        write_json(plan_path, plan)
    here = Path(__file__).resolve().parent
    env = os.environ.copy()
    env["PYTHONPATH"] = plan["repo_root"] + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("MPLCONFIGDIR", str(output / "_mplconfig"))
    for index, job in enumerate(plan["jobs"]):
        job_dir = Path(job["output_dir"])
        if args.resume and (job_dir / "COMPLETE.json").is_file():
            print(f"[SKIP] completed {job['model']} N={job['N']}", flush=True)
            continue
        job_dir.mkdir(parents=True, exist_ok=args.resume)
        command = [sys.executable, str(here / "evaluate_group.py"), "--plan", str(plan_path), "--job-index", str(index)]
        print(f"[RUN] {shlex.join(command)}\n[LOG] {job_dir / 'evaluation.log'}", flush=True)
        with (job_dir / "evaluation.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=job_dir, env=env, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            write_json(output / "FAILED.json", {"job_index": index, "model": job["model"], "N": job["N"], "exit_code": completed.returncode})
            print("\n".join((job_dir / "evaluation.log").read_text(encoding="utf-8", errors="replace").splitlines()[-25:]), file=sys.stderr)
            raise SystemExit(f"Evaluation failed; see {job_dir / 'evaluation.log'}")
    subprocess.run([sys.executable, str(here / "summarize.py"), "--run_dir", str(output)], check=True, env=env)
    if plan["plot_windows"] or plan["save_trajectories"]:
        from forecast_outputs import write_index
        write_index(output)
    (output / "FAILED.json").unlink(missing_ok=True)
    write_json(output / "COMPLETE.json", {"jobs_completed": len(plan["jobs"]), "summary": "summary/by_n_metrics.csv"})
    print(f"DONE: {output}")


if __name__ == "__main__":
    main()
