from __future__ import annotations

import argparse
import importlib
import math
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse

PARTIAL_WARNING_MESSAGE = (
    "functools\\.partial will be a method descriptor in future "
    "Python versions; wrap it in enum\\.member\\(\\) if you want "
    "to preserve the old behavior"
)
PARTIAL_WARNING_ENV_FILTER = (
    "ignore:functools.partial will be a method descriptor in future "
    "Python versions; wrap it in enum.member() if you want to preserve "
    "the old behavior:FutureWarning"
)

existing_warning_filters = os.environ.get("PYTHONWARNINGS")
if existing_warning_filters:
    if PARTIAL_WARNING_ENV_FILTER not in existing_warning_filters:
        os.environ["PYTHONWARNINGS"] = (
            f"{existing_warning_filters},{PARTIAL_WARNING_ENV_FILTER}"
        )
else:
    os.environ["PYTHONWARNINGS"] = PARTIAL_WARNING_ENV_FILTER

warnings.filterwarnings(
    "ignore",
    message=PARTIAL_WARNING_MESSAGE,
    category=FutureWarning,
)

ROOT = Path(__file__).resolve().parent
MPLCONFIGDIR = ROOT / ".tmp_mpl"
NUMBA_CACHE_DIR = ROOT / ".tmp_numba"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
NUMBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("NUMBA_CACHE_DIR", str(NUMBA_CACHE_DIR))

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

sq = importlib.import_module("squidpy")


DEFAULT_N_JOBS = (1, 2, 8)
DEFAULT_SCENARIOS = (
    "sparse_5k_256g",
    "dense_5k_256g",
    "sparse_10k_256g",
    "dense_10k_256g",
)
SPATIAL_KEY = "spatial"


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    target_n_cells: int
    n_genes: int
    matrix_kind: str
    max_neighs: int = 6


SCENARIOS = (
    ScenarioConfig(
        name="sparse_5k_256g",
        target_n_cells=5_000,
        n_genes=256,
        matrix_kind="sparse",
    ),
    ScenarioConfig(
        name="sparse_10k_256g",
        target_n_cells=10_000,
        n_genes=256,
        matrix_kind="sparse",
    ),
    ScenarioConfig(
        name="sparse_40k_512g",
        target_n_cells=40_000,
        n_genes=512,
        matrix_kind="sparse",
    ),
    ScenarioConfig(
        name="sparse_120k_1024g",
        target_n_cells=120_000,
        n_genes=1024,
        matrix_kind="sparse",
    ),
    ScenarioConfig(
        name="dense_5k_256g",
        target_n_cells=5_000,
        n_genes=256,
        matrix_kind="dense",
    ),
    ScenarioConfig(
        name="dense_10k_256g",
        target_n_cells=10_000,
        n_genes=256,
        matrix_kind="dense",
    ),
    ScenarioConfig(
        name="dense_40k_512g",
        target_n_cells=40_000,
        n_genes=512,
        matrix_kind="dense",
    ),
)


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_base_adata() -> ad.AnnData:
    adata = ad.read_h5ad(ROOT / "tests" / "_data" / "test_data.h5ad")
    if SPATIAL_KEY not in adata.obsm:
        raise ValueError(f"Expected `{SPATIAL_KEY}` in `adata.obsm`.")
    return adata


def tile_spatial_adata(adata: ad.AnnData, repeats: int) -> ad.AnnData:
    if repeats == 1:
        tiled = adata.copy()
    else:
        coords = np.asarray(adata.obsm[SPATIAL_KEY], dtype=np.float64)
        width = float(coords[:, 0].max() - coords[:, 0].min())
        height = float(coords[:, 1].max() - coords[:, 1].min())
        x_gap = width + 500.0
        y_gap = height + 500.0
        n_cols = int(math.ceil(math.sqrt(repeats)))

        copies: list[ad.AnnData] = []
        for idx in range(repeats):
            copy = adata.copy()
            row, col = divmod(idx, n_cols)
            shifted = np.asarray(copy.obsm[SPATIAL_KEY], dtype=np.float64).copy()
            shifted[:, 0] += col * x_gap
            shifted[:, 1] += row * y_gap
            copy.obsm[SPATIAL_KEY] = shifted
            copies.append(copy)

        tiled = ad.concat(
            copies,
            label="_tile_id",
            keys=[str(i) for i in range(repeats)],
            index_unique="-",
        )

    for key in ("spatial_connectivities", "spatial_distances", "connectivities", "distances"):
        if key in tiled.obsp:
            del tiled.obsp[key]
    sq.gr.spatial_neighbors(tiled, coord_type="grid", spatial_key=SPATIAL_KEY)
    return tiled


def prepare_scenario(base: ad.AnnData, config: ScenarioConfig) -> tuple[ad.AnnData, list[str]]:
    if config.n_genes > base.n_vars:
        raise ValueError(
            f"Scenario `{config.name}` requires {config.n_genes} genes, "
            f"but only {base.n_vars} are available."
        )

    repeats = max(1, (config.target_n_cells + base.n_obs - 1) // base.n_obs)
    adata = tile_spatial_adata(base[:, : config.n_genes].copy(), repeats)
    genes = list(map(str, adata.var_names[: config.n_genes]))

    if config.matrix_kind == "dense":
        if issparse(adata.X):
            adata.X = np.asarray(adata.X.toarray(), dtype=np.float32)
        else:
            adata.X = np.asarray(adata.X, dtype=np.float32)
    else:
        if not issparse(adata.X):
            adata.X = csr_matrix(np.asarray(adata.X, dtype=np.float32))
        else:
            adata.X = csr_matrix(adata.X)

    return adata, genes


def benchmark_case(
    adata: ad.AnnData,
    genes: list[str],
    *,
    max_neighs: int,
    n_iter: int,
    dt: float,
    thresh: float,
    n_jobs: int,
    repeats: int,
    timeout_seconds: float | None,
) -> dict[str, float | int | str]:
    timings: list[float] = []
    status = "completed"
    reason = ""

    for _ in range(repeats):
        start = perf_counter()
        sq.gr.sepal(
            adata,
            max_neighs=max_neighs,
            genes=genes,
            n_iter=n_iter,
            dt=dt,
            thresh=thresh,
            copy=True,
            n_jobs=n_jobs,
            show_progress_bar=True,
        )
        elapsed = perf_counter() - start
        timings.append(elapsed)

        if timeout_seconds is not None and elapsed > timeout_seconds:
            status = "timed_out"
            reason = (
                f"Observed single run took {elapsed:.3f}s, "
                f"which exceeds timeout {timeout_seconds:.3f}s."
            )
            break

    return {
        "status": status,
        "reason": reason,
        "seconds_min": float(min(timings)),
        "seconds_median": float(np.median(timings)),
        "seconds_mean": float(np.mean(timings)),
        "seconds_per_gene": float(np.median(timings) / len(genes)),
    }


def run_benchmarks(
    *,
    scenario_names: set[str],
    n_jobs_values: list[int],
    repeats: int,
    n_iter: int,
    dt: float,
    thresh: float,
    label: str,
    timeout_seconds: float | None,
) -> pd.DataFrame:
    base = load_base_adata()
    rows: list[dict[str, object]] = []
    selected = [scenario for scenario in SCENARIOS if scenario.name in scenario_names]

    if not selected:
        raise ValueError("No scenarios selected.")

    warmup_adata, warmup_genes = prepare_scenario(base, selected[0])
    sq.gr.sepal(
        warmup_adata,
        max_neighs=selected[0].max_neighs,
        genes=warmup_genes[: min(16, len(warmup_genes))],
        n_iter=min(n_iter, 250),
        dt=dt,
        thresh=thresh,
        copy=True,
        n_jobs=1,
        show_progress_bar=False,
    )

    for scenario in selected:
        adata, genes = prepare_scenario(base, scenario)
        n_cells = int(adata.n_obs)
        n_genes = int(len(genes))

        print(
            f"\nRunning scenario `{scenario.name}` "
            f"with {n_cells} cells, {n_genes} genes and "
            f"{scenario.matrix_kind} input."
        )

        for n_jobs in n_jobs_values:
            if n_jobs > n_genes:
                continue

            print(f"  Benchmarking n_jobs={n_jobs}...")
            stats = benchmark_case(
                adata,
                genes,
                max_neighs=scenario.max_neighs,
                n_iter=n_iter,
                dt=dt,
                thresh=thresh,
                n_jobs=n_jobs,
                repeats=repeats,
                timeout_seconds=timeout_seconds,
            )
            rows.append(
                {
                    "label": label,
                    "scenario": scenario.name,
                    "n_cells": n_cells,
                    "n_genes": n_genes,
                    "matrix_kind": scenario.matrix_kind,
                    "max_neighs": scenario.max_neighs,
                    "n_iter": n_iter,
                    "dt": dt,
                    "thresh": thresh,
                    "n_jobs": n_jobs,
                    "repeats": repeats,
                    **stats,
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(["scenario", "n_jobs"])
        .reset_index(drop=True)
    )


def print_table(df: pd.DataFrame, columns: list[str]) -> None:
    printable = df[columns].copy()
    float_columns = printable.select_dtypes(include="float").columns
    for column in float_columns:
        printable[column] = printable[column].map(lambda value: f"{value:.3f}")
    print(printable.to_string(index=False))


def compare_results(current: pd.DataFrame, baseline_path: Path) -> pd.DataFrame:
    baseline = pd.read_csv(baseline_path)
    current = current[current["status"] == "completed"].copy()
    baseline = baseline[baseline["status"] == "completed"].copy()
    keys = [
        "scenario",
        "n_cells",
        "n_genes",
        "matrix_kind",
        "max_neighs",
        "n_iter",
        "dt",
        "thresh",
        "n_jobs",
    ]
    merged = current.merge(
        baseline[
            keys
            + [
                "seconds_min",
                "seconds_mean",
                "seconds_median",
                "seconds_per_gene",
                "label",
            ]
        ],
        on=keys,
        how="inner",
        suffixes=("", "_baseline"),
    )
    merged["baseline_label"] = merged["label_baseline"]
    merged["pct_change_vs_baseline"] = (
        (merged["seconds_mean_baseline"] - merged["seconds_mean"])
        / merged["seconds_mean_baseline"]
        * 100.0
    )
    return merged.sort_values(["scenario", "n_jobs"]).reset_index(drop=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark sq.gr.sepal across n_jobs and input sizes."
    )
    parser.add_argument(
        "--label",
        default="current",
        help="Label stored in the output file.",
    )
    parser.add_argument(
        "--scenarios",
        default=",".join(DEFAULT_SCENARIOS),
        help="Comma-separated scenario names.",
    )
    parser.add_argument(
        "--n-jobs",
        default=",".join(str(value) for value in DEFAULT_N_JOBS),
        help="Comma-separated core counts.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of timing repeats per benchmark case.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=30_000,
        help="Maximum diffusion iterations passed to sepal.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.001,
        help="Diffusion time step passed to sepal.",
    )
    parser.add_argument(
        "--thresh",
        type=float,
        default=1e-8,
        help="Convergence threshold passed to sepal.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV path for saving results.",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        help="Optional CSV from another branch to compare against.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="Soft per-case timeout.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    scenario_names = {
        name.strip() for name in args.scenarios.split(",") if name.strip()
    }
    df = run_benchmarks(
        scenario_names=scenario_names,
        n_jobs_values=parse_int_list(args.n_jobs),
        repeats=args.repeats,
        n_iter=args.n_iter,
        dt=args.dt,
        thresh=args.thresh,
        label=args.label,
        timeout_seconds=args.timeout_seconds,
    )

    print("\nBenchmark summary\n")
    completed = df[df["status"] == "completed"]
    if len(completed):
        print_table(
            completed,
            [
                "scenario",
                "n_jobs",
                "seconds_min",
                "seconds_median",
                "seconds_mean",
                "seconds_per_gene",
            ],
        )

    skipped = df[df["status"] != "completed"]
    if len(skipped):
        print("\nSkipped or truncated cases\n")
        print_table(
            skipped,
            [
                "scenario",
                "n_jobs",
                "status",
                "reason",
            ],
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"\nSaved results to {args.output}")

    if args.compare is not None:
        compared = compare_results(df, args.compare)
        print(f"\nComparison vs {args.compare}\n")
        print_table(
            compared,
            [
                "scenario",
                "n_jobs",
                "baseline_label",
                "seconds_mean_baseline",
                "seconds_mean",
                "pct_change_vs_baseline",
            ],
        )


if __name__ == "__main__":
    main()
