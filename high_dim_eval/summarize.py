#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


PRED_FILE_RE = re.compile(r".*_pred(\d+)\.csv$")
SYSTEM_N_RE = re.compile(r"_N(\d+)_")
PLUSMINUS_RE = re.compile(
    r"^\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*±\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
)

ID_COLS = {"run_id", "model", "pred_horizon", "system", "N"}
WIDE_ID_COLS = {"run_id", "pred_horizon", "N"}


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def get_pred_horizon(metrics_file: Path) -> str:
    match = PRED_FILE_RE.match(metrics_file.name)
    if match:
        return match.group(1)
    return "unknown"


def parse_n_from_system(system_name: str) -> int | None:
    match = SYSTEM_N_RE.search(system_name)
    if not match:
        return None
    return int(match.group(1))


def maybe_float(raw: str) -> float | None:
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return val


def parse_metric_value(raw: str) -> tuple[str, float | None, float | None]:
    if raw is None:
        return "empty", None, None
    text = str(raw).strip()
    if text == "":
        return "empty", None, None
    if text.upper() in {"N/A", "NA", "NAN"}:
        return "literal-NA", None, None
    match = PLUSMINUS_RE.match(text)
    if match:
        mean_val = maybe_float(match.group(1))
        pm_val = maybe_float(match.group(2))
        if mean_val is None or pm_val is None:
            return "parse-fail", None, None
        return "plusminus-like", mean_val, pm_val
    numeric = maybe_float(text)
    if numeric is not None:
        return "numeric-like", numeric, None
    return "parse-fail", None, None


def ordered_fieldnames(rows: Iterable[Dict[str, str]], preferred: List[str]) -> List[str]:
    union = set()
    for row in rows:
        union.update(row.keys())
    tail = sorted([k for k in union if k not in preferred])
    return [k for k in preferred if k in union] + tail


def discover_model_dirs(run_dir: Path) -> List[Path]:
    return sorted(d for d in run_dir.iterdir() if d.is_dir() and d.name in {"full", "panda"})


def collect_long_rows(run_dir: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    rows: List[Dict[str, str]] = []
    bad_systems: List[str] = []
    for model_dir in discover_model_dirs(run_dir):
        model = model_dir.name
        metric_files = sorted(model_dir.glob("N*/metrics_pred*.csv"))
        for metric_file in metric_files:
            pred_horizon = get_pred_horizon(metric_file)
            for row in read_csv_rows(metric_file):
                system = row.get("system", "")
                n_value = parse_n_from_system(system)
                if n_value is None:
                    bad_systems.append(system)
                    continue
                wrapped = {
                    "run_id": run_dir.name,
                    "model": model,
                    "pred_horizon": pred_horizon,
                    "N": str(n_value),
                    "system": system,
                }
                wrapped.update({k: str(v) for k, v in row.items()})
                rows.append(wrapped)
    return rows, sorted(set(bad_systems))


def collect_metric_columns(long_rows: List[Dict[str, str]]) -> List[str]:
    metric_cols = set()
    for row in long_rows:
        for col, val in row.items():
            if col in ID_COLS:
                continue
            # Keep even all-missing metric columns so validity counts are visible.
            metric_cols.add(col)
    return sorted(metric_cols)


def aggregate_by_n(long_rows: List[Dict[str, str]], metric_cols: List[str]) -> List[Dict[str, str]]:
    grouped: Dict[tuple[str, str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in long_rows:
        key = (row["run_id"], row["model"], row["pred_horizon"], row["N"])
        grouped[key].append(row)

    aggregated = []
    for key in sorted(grouped):
        rows = grouped[key]
        systems = sorted({r["system"] for r in rows})
        base = {
            "run_id": key[0],
            "model": key[1],
            "pred_horizon": key[2],
            "N": key[3],
            "count_rows_total": str(len(rows)),
            "count_systems_total": str(len(systems)),
            # Backward compatible aliases used by older downstream files.
            "count_rows": str(len(rows)),
            "count_systems": str(len(systems)),
        }
        for col in metric_cols:
            means: List[float] = []
            pms: List[float] = []
            count_valid = 0
            count_na = 0
            count_empty = 0
            count_parsefail = 0

            for row in rows:
                value_kind, mean_val, pm_val = parse_metric_value(row.get(col, ""))
                if value_kind in {"numeric-like", "plusminus-like"} and mean_val is not None:
                    means.append(mean_val)
                    count_valid += 1
                    if pm_val is not None:
                        pms.append(pm_val)
                elif value_kind == "literal-NA":
                    count_na += 1
                elif value_kind == "empty":
                    count_empty += 1
                elif value_kind == "parse-fail":
                    count_parsefail += 1

            base[f"count_rows_valid_{col}"] = str(count_valid)
            base[f"count_rows_na_{col}"] = str(count_na)
            base[f"count_rows_empty_{col}"] = str(count_empty)
            base[f"count_rows_parsefail_{col}"] = str(count_parsefail)

            if not means:
                continue
            mean_v = sum(means) / len(means)
            if len(means) <= 1:
                std_v = 0.0
            else:
                variance = sum((v - mean_v) ** 2 for v in means) / (len(means) - 1)
                std_v = math.sqrt(variance)
            base[f"{col}__mean_of_means"] = f"{mean_v:.12g}"
            base[f"{col}__std_across_systems"] = f"{std_v:.12g}"
            if pms:
                base[f"{col}__mean_pm"] = f"{(sum(pms) / len(pms)):.12g}"
        aggregated.append(base)
    return aggregated


def build_wide_rows(by_n_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    grouped: Dict[tuple[str, str, str], Dict[str, str]] = defaultdict(dict)
    for row in by_n_rows:
        key = (row["run_id"], row["pred_horizon"], row["N"])
        model = row["model"]
        base = grouped[key]
        base["run_id"] = key[0]
        base["pred_horizon"] = key[1]
        base["N"] = key[2]
        for col, val in row.items():
            if col in WIDE_ID_COLS or col == "model":
                continue
            base[f"{col}__{model}"] = val
    return [grouped[k] for k in sorted(grouped)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect coupled Lorenz63 evaluation csv into summary tables."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="One run folder generated by run_evaluation.py.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    summary_dir = run_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    long_rows, bad_systems = collect_long_rows(run_dir)
    if not long_rows:
        raise RuntimeError(f"No metrics files found under run_dir: {run_dir}")

    if bad_systems:
        raise ValueError(f"Unrecognized system names: {bad_systems}")
    keys = [(r["model"], r["pred_horizon"], r["system"]) for r in long_rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate model/horizon/system rows; do not pool repeated runs")

    long_path = summary_dir / "long_metrics.csv"
    long_fields = ordered_fieldnames(
        long_rows,
        preferred=["run_id", "model", "pred_horizon", "N", "system"],
    )
    write_csv(long_path, long_rows, long_fields)

    metric_cols = collect_metric_columns(long_rows)
    by_n_rows = aggregate_by_n(long_rows, metric_cols)
    by_n_path = summary_dir / "by_n_metrics.csv"
    by_n_fields = ordered_fieldnames(
        by_n_rows,
        preferred=["run_id", "model", "pred_horizon", "N", "count_rows", "count_systems"],
    )
    write_csv(by_n_path, by_n_rows, by_n_fields)

    wide_rows = build_wide_rows(by_n_rows)
    wide_path = summary_dir / "full_vs_panda_by_n.csv"
    wide_fields = ordered_fieldnames(
        wide_rows,
        preferred=["run_id", "pred_horizon", "N"],
    )
    write_csv(wide_path, wide_rows, wide_fields)

    notes = {
        "run_id": run_dir.name,
        "total_long_rows": len(long_rows),
        "total_by_n_rows": len(by_n_rows),
        "total_wide_rows": len(wide_rows),
        "metric_columns_aggregated": metric_cols,
        "dropped_rows_due_to_unparsed_system_N": bad_systems,
        "long_metrics_csv": str(long_path),
        "by_n_metrics_csv": str(by_n_path),
        "full_vs_panda_by_n_csv": str(wide_path),
    }
    (summary_dir / "run_notes.md").write_text(
        "# Coupled Lorenz63 Summary\n\n"
        + "```json\n"
        + json.dumps(notes, indent=2, ensure_ascii=False)
        + "\n```\n"
    )

    print(f"[DONE] Summary files written to: {summary_dir}")


if __name__ == "__main__":
    main()
