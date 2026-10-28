#!/usr/bin/env python3
"""Replot saved prediction windows; does not need PyTorch or a GPU."""
from __future__ import annotations

import argparse
from pathlib import Path

from forecast_outputs import plot_window, write_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--nodes", help="Zero-based node IDs, e.g. 0,127,255")
    parser.add_argument("--max-windows", type=int, default=2, help="Per model and graph; -1 for all")
    parser.add_argument("--context-steps", type=int, default=128)
    args = parser.parse_args()
    if args.max_windows == 0 or args.max_windows < -1 or args.context_steps < 1:
        parser.error("max-windows must be positive or -1; context-steps must be positive")
    nodes = [int(v) for v in args.nodes.split(",")] if args.nodes else None
    files = sorted(args.run_dir.glob("*/N*/forecasts/*/*.npz"))
    if not files:
        parser.error("No saved forecasts found; run evaluation with --plot-windows 2 or --save-trajectories")
    counts = {}
    for file in files:
        counts[file.parent] = counts.get(file.parent, 0) + 1
        if args.max_windows == -1 or counts[file.parent] <= args.max_windows:
            print(file, flush=True)
            plot_window(file, nodes, args.context_steps)
    print(write_index(args.run_dir.resolve()))


if __name__ == "__main__":
    main()
