# ruff: noqa: E402, T201, S607
"""Run grids with algorithms and collect the results in one table.

Edit the settings below, then from the repository folder run:

    uv run python plug_and_play/run.py

Each run is ToOp's DC optimisation with the chosen algorithm, followed by ToOp's AC check on the best
plans. Results go to plug_and_play/results/<date and time>/:

    runs.csv        one row per grid, algorithm and seed
    candidates.csv  one row per plan sent to the AC check, with the rejection reason
    settings.json   the settings and code version used
"""

# ---------------------------------------------------------------- settings
GRIDS = [
    "path/to/your/grid",  # a grid folder prepared by ToOp's preprocessing (holds static_information.hdf5)
]
ALGORITHMS = ["map_elites", "random_search"]  # file names from plug_and_play/algorithms
SEEDS = 3  # runs per grid and algorithm
SECONDS = 60  # DC search time per run
AC_PLANS = 10  # how many of the best DC plans go to the AC check
# -------------------------------------------------------------------------

import json
import shutil
import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
import structlog
from omegaconf import DictConfig
from toop_engine_topology_optimizer.ac.scoring_functions import evaluate_acceptance
from toop_engine_topology_optimizer.benchmark.benchmark_utils import (
    perform_ac_analysis,
    run_dc_optimization_stage,
    suppress_jax_logs,
)
from toop_engine_topology_optimizer.dc.algorithm_loader import available
from toop_engine_topology_optimizer.interfaces.messages.results import Metrics

OVERLOAD = "overload_energy_artificial_limit_n_1"
RESULTS = Path(__file__).resolve().parent / "results"


def grid_data_folder(grid: str) -> Path:
    """Accept either the folder holding static_information.hdf5 or its parent with a toop/ subfolder."""
    folder = Path(grid).expanduser().resolve()
    if (folder / "toop" / "static_information.hdf5").exists():
        folder = folder / "toop"
    if not (folder / "static_information.hdf5").exists():
        raise FileNotFoundError(f"No static_information.hdf5 in {folder}. Is this a grid prepared by ToOp?")
    return folder


def load_metrics(path: Path) -> Metrics:
    """Read an AC metrics file written by ToOp's AC stage."""
    data = json.loads(path.read_text())
    return Metrics(fitness=data["fitness"], extra_scores=data["extra_scores"])


def run_one(data_folder: Path, algorithm: str, seed: int, run_dir: Path) -> tuple[dict, list[dict]]:
    """One DC optimisation and AC check. Returns the row for runs.csv and the rows for candidates.csv."""
    dc_config = DictConfig(
        {
            "task_name": f"{algorithm}_{seed}",
            "fixed_files": [str(data_folder / "static_information.hdf5")],
            "double_precision": None,
            "tensorboard_dir": str(run_dir / "tensorboard"),
            "stats_dir": str(run_dir / "stats"),
            "summary_frequency": None,
            "checkpoint_frequency": None,
            "num_cuda_devices": 1,
            "omp_num_threads": 1,
            "xla_force_host_platform_device_count": None,
            "lf_config": {"distributed": False},
            "ga_config": {"runtime_seconds": SECONDS, "random_seed": seed, "algorithm": algorithm},
        }
    )
    ac_config = DictConfig(
        {"n_processes": 1, "k_best_topos": AC_PLANS, "critical_voltage_jump_percent": 5.0, "critical_va_diff_degree": 20.0}
    )

    start = time.time()
    run_dc_optimization_stage(dc_config, run_dir)
    dc_seconds = time.time() - start
    perform_ac_analysis(data_folder, run_dir, ac_validation_cfg=ac_config)
    ac_seconds = time.time() - start - dc_seconds
    shutil.rmtree(run_dir / "stats", ignore_errors=True)  # per-epoch snapshots, not needed once res.json exists

    # ToOp's AC stage only logs whether a plan passes, so it is worked out again here with ToOp's own rule
    res = json.loads((run_dir / "res.json").read_text())
    best_topos = sorted(res["best_topos"], key=lambda t: t["metrics"]["fitness"], reverse=True)
    unsplit = load_metrics(run_dir / "unsplit_ac_metrics.json")
    candidates = []
    for i in range(min(AC_PLANS, len(best_topos))):
        ac = load_metrics(run_dir / f"topology_{i}" / "ac_metrics.json")
        rejection = evaluate_acceptance(metrics_split=ac, metrics_unsplit=unsplit)
        candidates.append(
            {
                "rank": i,
                "splits": len(best_topos[i].get("actions") or []),
                "disconnections": len(best_topos[i].get("disconnections") or []),
                "dc_overload": -best_topos[i]["metrics"]["fitness"],
                "ac_overload": ac.extra_scores.get(OVERLOAD),
                "passed_ac": rejection is None,
                "rejection_reason": "" if rejection is None else rejection.criterion,
            }
        )

    passed = [c for c in candidates if c["passed_ac"]]
    run = {
        "sent_to_ac": len(candidates),
        "passed_ac": len(passed),
        "best_ac_overload_passed": min((c["ac_overload"] for c in passed), default=None),
        "ac_overload_unsplit": unsplit.extra_scores.get(OVERLOAD),
        "dc_overload_best": min((c["dc_overload"] for c in candidates), default=None),
        "dc_overload_unsplit": -res["initial_fitness"],
        "dc_seconds": round(dc_seconds, 1),
        "ac_seconds": round(ac_seconds, 1),
    }
    return run, candidates


def code_version() -> str:
    """Return the git commit, so a result can be traced back to the code that made it."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        changed = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()
        return commit + (" + uncommitted changes" if changed else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run(grids: list[str], algorithms: list[str], seeds: int, out: Path = RESULTS) -> Path:
    """Run every grid with every algorithm and seed. Returns the results folder."""
    unknown = [a for a in algorithms if a not in available()]
    if unknown:
        raise ValueError(f"Unknown algorithms {unknown}. Available: {available()}")
    folders = {grid: grid_data_folder(grid) for grid in grids}

    out = out / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out.mkdir(parents=True)
    settings = {"grids": grids, "algorithms": algorithms, "seeds": seeds, "seconds": SECONDS, "ac_plans": AC_PLANS}
    settings |= {"code_version": code_version(), "started": datetime.now().isoformat()}
    (out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")

    runs, candidates = [], []
    for data_folder in folders.values():
        name = data_folder.parent.name if data_folder.name == "toop" else data_folder.name
        for algorithm in algorithms:
            for seed in range(seeds):
                key = {"grid": name, "algorithm": algorithm, "seed": seed}
                print(f"{name}  {algorithm}  seed {seed} ...", flush=True)
                try:
                    row, cands = run_one(data_folder, algorithm, seed, out / "runs" / name / algorithm / f"seed_{seed}")
                    runs.append({**key, **row, "error": ""})
                    candidates += [{**key, **c} for c in cands]
                    print(f"    {row['passed_ac']} of {row['sent_to_ac']} plans passed AC", flush=True)
                except Exception as exc:  # keep going, and record what went wrong
                    runs.append({**key, "error": repr(exc)})
                    print(f"    FAILED: {exc!r}", flush=True)
                    traceback.print_exc()
                # write after every run, so a crash keeps what was done
                pd.DataFrame(runs).to_csv(out / "runs.csv", index=False)
                pd.DataFrame(candidates).to_csv(out / "candidates.csv", index=False)

    print(f"\nResults in {out}")
    table = pd.DataFrame(runs)
    if "passed_ac" in table:
        summary = table.groupby(["grid", "algorithm"])[["sent_to_ac", "passed_ac", "best_ac_overload_passed"]].mean()
        print(summary.round(2))
    return out


if __name__ == "__main__":
    suppress_jax_logs()
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(30))  # warnings and errors only
    run(GRIDS, ALGORITHMS, SEEDS)
