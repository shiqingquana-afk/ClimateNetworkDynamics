"""Path, dataset, and metadata helpers; no model dependencies required."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

SYSTEM_RE = re.compile(r"^CoupledLorenz63_N(?P<n>\d+)_k[^_]+_g(?P<seed>\d+)$")
BATCH_SIZES = {32: 64, 64: 32, 128: 16, 256: 4, 512: 2}
DEFAULT_METRICS = "smape,ME-LRw,max_lyap_gt,max_lyap_pred,gd_rmse,kld"
SUPPORTED_METRICS = set(DEFAULT_METRICS.split(",")) | {"mse", "mae", "spearman"}
MODEL_NAMES = {"full": "Nexus", "panda": "Panda"}
CHECKPOINT_NAMES = {"full": "Nexus1.0-base", "panda": "PANDA-STANDARD"}


def csv_list(value: str) -> list[str]:
    return list(dict.fromkeys(x.strip() for x in value.split(",") if x.strip()))


def resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value).expanduser()
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def discover_systems(data_dir: Path, n_values: list[int], limit: int | None) -> dict[int, list[Path]]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    groups = {}
    for n in n_values:
        systems = sorted(
            (p for p in data_dir.iterdir()
             if p.is_dir() and SYSTEM_RE.fullmatch(p.name)
             and int(SYSTEM_RE.fullmatch(p.name)["n"]) == n
             and any(p.glob("*.arrow"))),
            key=lambda p: (int(SYSTEM_RE.fullmatch(p.name)["seed"]), p.name),
        )
        if not systems:
            raise ValueError(f"No nonempty CoupledLorenz63 system directories for N={n} in {data_dir}")
        groups[n] = systems[:limit]
    return groups


def check_manifest_completeness(data_dir: Path, n_values: list[int]) -> None:
    """Detect interrupted/partial datasets before applying evaluation sample caps."""
    path = data_dir / "manifest.json"
    if not path.is_file():
        return  # User-prepared data are validated by discovery and Arrow reads.
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("generation_complete") is False:
        raise ValueError(f"Dataset generation is incomplete: {data_dir}")
    declared = manifest.get("systems", [])
    for n in n_values:
        systems = [r for r in declared if SYSTEM_RE.fullmatch(r.get("system", ""))
                   and int(SYSTEM_RE.fullmatch(r["system"])["n"]) == n]
        expected_graphs = manifest.get("graph_seeds_per_n")
        if expected_graphs is not None and len(systems) != expected_graphs:
            raise ValueError(f"Incomplete dataset manifest for N={n}: expected {expected_graphs} graphs, found {len(systems)}")
        for record in systems:
            directory = data_dir / record["system"]
            expected = record.get("n_trajectories_saved", manifest.get("num_ics_per_graph"))
            configured = manifest.get("num_ics_per_graph")
            count = len(list(directory.glob("*.arrow")))
            if not directory.is_dir() or (expected is not None and count != expected) or (configured is not None and count != configured):
                raise ValueError(f"Incomplete dataset at {directory}: found {count} trajectories; manifest expects {configured or expected}")


def trajectory_files(system_dir: Path, limit: int | None) -> list[Path]:
    def key(path: Path):
        prefix = path.stem.split("_")[0]
        return (0, int(prefix), path.name) if prefix.isdigit() else (1, 0, path.name)
    return sorted(system_dir.glob("*.arrow"), key=key)[:limit]


def system_dt(system: str, data_dir: Path, lyapunov_dir: Path) -> float:
    """Read the generated sampling interval, never silently substitute tau=1."""
    match = SYSTEM_RE.fullmatch(system)
    if not match:
        raise ValueError(f"Invalid coupled-system name: {system}")
    path = lyapunov_dir / f"N{match['n']}_g{match['seed']}.json"
    value = None
    if path.is_file():
        value = json.loads(path.read_text(encoding="utf-8")).get("dt")
    else:
        manifest = data_dir / "manifest.json"
        if manifest.is_file():
            rows = json.loads(manifest.read_text(encoding="utf-8")).get("systems", [])
            value = next((row.get("dt") for row in rows if row.get("system") == system), None)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"Missing/invalid dt for {system}. Supply {path} or a manifest.json systems entry with dt.")
    return float(value)
