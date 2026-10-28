#!/usr/bin/env python3
"""Run a small check, a high-dimensional preview, or the full paper protocol."""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


PROFILES = {
    "smoke": {"nodes": "32", "graphs": "1", "ics": "1", "windows": "2", "horizon": "128", "metrics": "smape,mse,mae"},
    "highdim": {"nodes": "256,512", "graphs": "1", "ics": "1", "windows": "2", "horizon": "512", "metrics": "smape,mse,mae,ME-LRw"},
    "paper": {"nodes": "256,512", "graphs": "5", "ics": "20", "windows": "6", "horizon": "512", "metrics": "smape,ME-LRw,max_lyap_gt,max_lyap_pred,gd_rmse,kld"},
}


def checkpoint_dir(path: Path) -> Path:
    """Accept a checkpoint itself or a model directory containing one final checkpoint."""
    candidates = [path, path / "checkpoint-final"]
    valid = [p.resolve() for p in candidates if (p / "config.json").is_file()
             and (any(p.glob("*.safetensors")) or any(p.glob("pytorch_model*.bin")))]
    if len(valid) == 1:
        return valid[0]
    if not valid:
        raise ValueError(f"No complete checkpoint in {path}. Need config.json and model weights, possibly inside checkpoint-final/.")
    raise ValueError(f"More than one checkpoint in {path}; specify the exact checkpoint directory.")


def find_checkpoint(root: Path, supplied: str | None, panda: bool = False) -> Path:
    if supplied:
        path = Path(supplied).expanduser()
        return checkpoint_dir((root / path) if not path.is_absolute() else path)
    names = ["PANDA-STANDARD", "Panda", "panda"] if panda else ["Nexus1.0-base", "ChaosNexus", "Nexus"]
    found = set()
    for parent in [root / "checkpoints", root / "models", root]:
        for name in names:
            path = parent / name
            if path.is_dir():
                try:
                    found.add(checkpoint_dir(path))
                except ValueError:
                    pass
    if len(found) == 1:
        return found.pop()
    option = "--panda-checkpoint" if panda else "--checkpoint"
    if not found:
        raise ValueError(f"Model weights are missing. Pass {option} /path/to/model/checkpoint-final; obtain the complete original weights from the official README. This script does not use randomly initialized weights.")
    raise ValueError(f"Multiple model directories found. Select one with {option}.")


def command_for(args: argparse.Namespace, extra: list[str]) -> list[str]:
    root = args.repo_root.resolve()
    profile = PROFILES[args.profile]
    checkpoint = find_checkpoint(root, args.checkpoint)
    model_label = "/".join([checkpoint.name, checkpoint.parent.name]).lower()
    if any(word in model_label for word in ["weather", "-sft"]):
        raise ValueError("This zero-shot benchmark needs the original ChaosNexus checkpoint; the selected path looks like weather fine-tuning weights.")
    command = [sys.executable, str(root / "high_dim_eval/run_evaluation.py"), "--repo-root", str(root),
               "--full-checkpoint", str(checkpoint), "--models", "full,panda" if args.compare_panda else "full",
               "--n-values", profile["nodes"], "--generation-graphs-per-n", profile["graphs"],
               "--generation-ics-per-graph", profile["ics"], "--num-test-instances", profile["windows"],
               "--prediction-length", profile["horizon"], "--metrics", profile["metrics"],
               "--batch-size", "1", "--plot-windows", "2", "--device", args.device,
               "--data-dir", args.data_dir or f"data/high_dim_{args.profile}"]
    if args.compare_panda:
        command.extend(["--panda-checkpoint", str(find_checkpoint(root, args.panda_checkpoint, panda=True))])
    if args.output_dir:
        command.extend(["--output-dir", args.output_dir])
    if args.dry_run:
        command.append("--dry-run")
    if args.resume:
        command.append("--resume")
    # Extra run_evaluation flags override a profile, without editing this script.
    return command + extra


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="highdim")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--checkpoint", "--full-checkpoint", dest="checkpoint")
    parser.add_argument("--compare-panda", action="store_true")
    parser.add_argument("--panda-checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args, extra = parser.parse_known_args()
    try:
        command = command_for(args, extra)
    except ValueError as exc:
        parser.error(str(exc))
    print(shlex.join(command), flush=True)
    raise SystemExit(subprocess.run(command).returncode)


if __name__ == "__main__":
    main()
