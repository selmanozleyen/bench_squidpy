from __future__ import annotations

import argparse
import importlib
import sys
import warnings
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from time import perf_counter

import anndata as ad
import numpy as np
import pandas as pd

warnings.filterwarnings(
    "ignore",
    message=(
        "functools\\.partial will be a method descriptor in future "
        "Python versions; wrap it in enum\\.member\\(\\) if you want "
        "to preserve the old behavior"
    ),
    category=FutureWarning,
)

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

sq = importlib.import_module("squidpy")


DEFAULT_N_PERMS = (100, 250, 500)
DEFAULT_N_JOBS = (1, 2, 8)
CLUSTER_KEY = "leiden"
DEFAULT_SCENARIOS = ("large", "more_clusters")


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    cell_repeats: int
    n_genes: int
    interaction_genes: int
    split_clusters: bool = False


SCENARIOS = (
    ScenarioConfig(
        name="baseline",
        cell_repeats=8,
        n_genes=24,
        interaction_genes=24,
    ),
    ScenarioConfig(
        name="more_interactions",
        cell_repeats=8,
        n_genes=48,
        interaction_genes=48,
    ),
    ScenarioConfig(
        name="more_cells",
        cell_repeats=32,
        n_genes=24,
        interaction_genes=24,
    ),
    ScenarioConfig(
        name="more_clusters",
        cell_repeats=8,
        n_genes=24,
        interaction_genes=24,
        split_clusters=True,
    ),
    ScenarioConfig(
        name="large",
        cell_repeats=12,
        n_genes=32,
        interaction_genes=32,
        split_clusters=True,
    ),
)


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_base_adata() -> ad.AnnData:
    adata = ad.read_h5ad(ROOT / "tests" / "_data" / "test_data.h5ad")
    adata.obs[CLUSTER_KEY] = adata.obs[CLUSTER_KEY].astype("category")
    return adata


def repeat_adata(adata: ad.AnnData, repeats: int) -> ad.AnnData:
    if repeats == 1:
        return adata.copy()

    copies = [adata.copy() for _ in range(repeats)]
    keys = [str(i) for i in range(repeats)]
    return ad.concat(copies, label="_repeat_id", keys=keys, index_unique="-")


def prepare_scenario(
    base: ad.AnnData,
    config: ScenarioConfig,
) -> tuple[ad.AnnData, list[tuple[str, str]]]:
    if config.interaction_genes > config.n_genes:
        raise ValueError(
            "Scenario "
            f"`{config.name}` has more interaction genes than available genes."
        )

    adata = repeat_adata(
        base[:, : config.n_genes].copy(),
        repeats=config.cell_repeats,
    )
    if config.split_clusters:
        repeated = adata.obs["_repeat_id"].astype("string")
        clusters = adata.obs[CLUSTER_KEY].astype("string")
        adata.obs[CLUSTER_KEY] = (clusters + "_" + repeated).astype("category")
    else:
        adata.obs[CLUSTER_KEY] = adata.obs[CLUSTER_KEY].astype("category")

    genes = list(map(str, adata.var_names[: config.interaction_genes]))
    interactions = list(product(genes, repeat=2))
    return adata, interactions


def benchmark_case(
    adata: ad.AnnData,
    interactions: list[tuple[str, str]],
    *,
    n_perms: int,
    n_jobs: int,
    repeats: int,
    seed: int,
    timeout_seconds: float | None,
) -> dict[str, float | int]:
    timings: list[float] = []
    status = "completed"
    reason = ""

    for repeat_idx in range(repeats):
        start = perf_counter()
        sq.gr.ligrec(
            adata,
            CLUSTER_KEY,
            interactions=interactions,
            use_raw=False,
            copy=True,
            show_progress_bar=True,
            n_perms=n_perms,
            n_jobs=n_jobs,
            seed=seed + repeat_idx,
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
        "seconds_per_perm": float(np.median(timings) / n_perms),
    }


def run_benchmarks(
    *,
    scenario_names: set[str],
    n_perms_values: list[int],
    n_jobs_values: list[int],
    repeats: int,
    seed: int,
    label: str,
    timeout_seconds: float | None,
) -> pd.DataFrame:
    base = load_base_adata()
    rows: list[dict[str, object]] = []
    history: dict[tuple[str, int], tuple[int, float]] = {}
    selected = [
        scenario for scenario in SCENARIOS if scenario.name in scenario_names
    ]

    if not selected:
        raise ValueError("No scenarios selected.")

    warmup_adata, warmup_interactions = prepare_scenario(base, selected[0])
    sq.gr.ligrec(
        warmup_adata,
        CLUSTER_KEY,
        interactions=warmup_interactions[: min(16, len(warmup_interactions))],
        use_raw=False,
        copy=True,
        show_progress_bar=False,
        n_perms=2,
        n_jobs=1,
        seed=seed,
    )

    for scenario in selected:
        adata, interactions = prepare_scenario(base, scenario)
        n_cells = int(adata.n_obs)
        n_genes = int(adata.n_vars)
        n_clusters = int(len(adata.obs[CLUSTER_KEY].cat.categories))
        n_interactions = int(len(interactions))

        print(
            f"\nRunning scenario `{scenario.name}` "
            f"with {n_cells} cells, {n_genes} genes, "
            f"{n_clusters} clusters and {n_interactions} interactions."
        )

        for n_perms in n_perms_values:
            for n_jobs in n_jobs_values:
                if n_jobs > n_cells:
                    continue

                print(
                    f"  Benchmarking n_perms={n_perms}, "
                    f"n_jobs={n_jobs}..."
                )

                history_key = (scenario.name, n_jobs)
                if timeout_seconds is not None and history_key in history:
                    previous_n_perms, previous_seconds = history[history_key]
                    estimated_seconds = previous_seconds * (
                        n_perms / previous_n_perms
                    )
                    if estimated_seconds > timeout_seconds:
                        rows.append(
                            {
                                "label": label,
                                "scenario": scenario.name,
                                "n_cells": n_cells,
                                "n_genes": n_genes,
                                "n_clusters": n_clusters,
                                "n_interactions": n_interactions,
                                "n_jobs": n_jobs,
                                "n_perms": n_perms,
                                "repeats": repeats,
                                "status": "skipped_timeout",
                                "reason": (
                                    f"Estimated {estimated_seconds:.3f}s from "
                                    f"{previous_n_perms} perms, which exceeds "
                                    f"timeout {timeout_seconds:.3f}s."
                                ),
                                "seconds_min": np.nan,
                                "seconds_median": np.nan,
                                "seconds_mean": np.nan,
                                "seconds_per_perm": np.nan,
                            }
                        )
                        print(
                            "  Skipping due to timeout estimate: "
                            f"{estimated_seconds:.3f}s > "
                            f"{timeout_seconds:.3f}s."
                        )
                        continue

                stats = benchmark_case(
                    adata,
                    interactions,
                    n_perms=n_perms,
                    n_jobs=n_jobs,
                    repeats=repeats,
                    seed=seed,
                    timeout_seconds=timeout_seconds,
                )
                history[history_key] = (
                    n_perms,
                    float(stats["seconds_median"]),
                )
                rows.append(
                    {
                        "label": label,
                        "scenario": scenario.name,
                        "n_cells": n_cells,
                        "n_genes": n_genes,
                        "n_clusters": n_clusters,
                        "n_interactions": n_interactions,
                        "n_jobs": n_jobs,
                        "n_perms": n_perms,
                        "repeats": repeats,
                        **stats,
                    }
                )

    df = (
        pd.DataFrame(rows)
        .sort_values(["scenario", "n_perms", "n_jobs"])
        .reset_index(drop=True)
    )
    return df


def print_table(df: pd.DataFrame, columns: list[str]) -> None:
    printable = df[columns].copy()
    float_columns = printable.select_dtypes(include="float").columns
    for column in float_columns:
        printable[column] = printable[column].map(lambda value: f"{value:.3f}")
    print(printable.to_string(index=False))


def compare_results(
    current: pd.DataFrame, baseline_path: Path
) -> pd.DataFrame:
    baseline = pd.read_csv(baseline_path)
    current = current[current["status"] == "completed"].copy()
    baseline = baseline[baseline["status"] == "completed"].copy()
    keys = [
        "scenario",
        "n_cells",
        "n_genes",
        "n_clusters",
        "n_interactions",
        "n_jobs",
        "n_perms",
    ]
    merged = current.merge(
        baseline[
            keys
            + [
                "seconds_min",
                "seconds_mean",
                "seconds_median",
                "seconds_per_perm",
                "label",
            ]
        ],
        on=keys,
        how="inner",
        suffixes=("", "_baseline"),
    )
    merged["baseline_label"] = merged["label_baseline"]
    merged["pct_change_vs_baseline"] = (
        (merged["seconds_median_baseline"] - merged["seconds_median"])
        / merged["seconds_median_baseline"]
        * 100.0
    )
    return (
        merged.sort_values(["scenario", "n_perms", "n_jobs"])
        .reset_index(drop=True)
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark sq.gr.ligrec across n_perms, "
            "n_jobs and input sizes."
        )
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
        "--n-perms",
        default=",".join(str(value) for value in DEFAULT_N_PERMS),
        help="Comma-separated permutation counts.",
    )
    parser.add_argument(
        "--n-jobs",
        default=",".join(str(value) for value in DEFAULT_N_JOBS),
        help="Comma-separated core counts.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="Number of timing repeats per benchmark case.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Base random seed."
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
        default=30,
        help=(
            "Soft per-case timeout. Larger cases can be skipped when a "
            "smaller run suggests they will exceed this limit."
        ),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    scenario_names = {
        name.strip() for name in args.scenarios.split(",") if name.strip()
    }
    df = run_benchmarks(
        scenario_names=scenario_names,
        n_perms_values=parse_int_list(args.n_perms),
        n_jobs_values=parse_int_list(args.n_jobs),
        repeats=args.repeats,
        seed=args.seed,
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
                "n_perms",
                "n_jobs",
                "seconds_min",
                "seconds_median",
                "seconds_mean",
                "seconds_per_perm",
            ],
        )

    skipped = df[df["status"] != "completed"]
    if len(skipped):
        print("\nSkipped or truncated cases\n")
        print_table(
            skipped,
            [
                "scenario",
                "n_perms",
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
                "n_perms",
                "n_jobs",
                "baseline_label",
                "seconds_min_baseline",
                "seconds_mean_baseline",
                "seconds_median_baseline",
                "seconds_min",
                "seconds_mean",
                "seconds_median",
                "pct_change_vs_baseline",
            ],
        )


if __name__ == "__main__":
    main()
