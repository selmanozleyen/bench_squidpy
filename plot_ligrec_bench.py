from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = ROOT / "results"
DEFAULT_OUTPUT_DIR = ROOT / "plots"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create n_jobs timing plots from all benchmark CSVs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing benchmark CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where plot PNG files will be written.",
    )
    parser.add_argument(
        "--metric",
        choices=("seconds_median", "seconds_mean", "seconds_min"),
        default="seconds_mean",
        help="Benchmark timing column to plot on the y-axis.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Overwrite existing output files without prompting.",
    )
    return parser.parse_args()


def sanitize_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "plot"


def load_results(input_dir: Path) -> tuple[pd.DataFrame, list[Path]]:
    csv_paths = sorted(input_dir.rglob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under {input_dir}.")

    frames: list[pd.DataFrame] = []
    for path in csv_paths:
        df = pd.read_csv(path)
        df["source_csv"] = path.name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    completed = combined[combined["status"] == "completed"].copy()
    if completed.empty:
        raise ValueError("No completed benchmark rows found in the CSV files.")

    return completed, csv_paths


def build_output_paths(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    paths = [output_dir / "overview_time_vs_n_jobs.png"]
    for scenario in sorted(df["scenario"].unique()):
        plot_name = f"{sanitize_filename(scenario)}_time_vs_n_jobs.png"
        paths.append(output_dir / plot_name)
    return paths


def confirm_overwrite(paths: list[Path], assume_yes: bool) -> None:
    existing_paths = [path for path in paths if path.exists()]
    if not existing_paths or assume_yes:
        return

    print("The following plot files already exist:")
    for path in existing_paths:
        print(f"  - {path}")

    response = input("Overwrite them? [y/N]: ").strip().lower()
    if response not in {"y", "yes"}:
        print("Keeping existing plots. Nothing was written.")
        raise SystemExit(0)


def label_lines_in_place(ax: plt.Axes, *, fontsize: int = 9) -> None:
    lines = [line for line in ax.get_lines() if len(line.get_xdata())]
    if not lines:
        return

    line_items = []
    for line in lines:
        xdata = np.asarray(line.get_xdata(), dtype=float)
        ydata = np.asarray(line.get_ydata(), dtype=float)
        mean_y = float(np.mean(ydata))
        line_items.append((mean_y, xdata, ydata, line))

    line_items.sort(key=lambda item: (item[1][-1], item[2][-1], item[0]))
    min_gap = 0.04
    placed_y: list[float] = []

    for _, xdata, ydata, line in line_items:
        x_last = float(xdata[-1])
        y_last = float(ydata[-1])
        y_text = y_last

        if placed_y:
            y_span = max(ax.get_ylim()[1] - ax.get_ylim()[0], 1e-12)
            min_delta = y_span * min_gap
            for previous_y in placed_y:
                if abs(y_text - previous_y) < min_delta:
                    y_text = previous_y + min_delta

        placed_y.append(y_text)
        ax.annotate(
            line.get_label(),
            xy=(x_last, y_last),
            xytext=(8, y_text - y_last),
            textcoords="offset points",
            color=line.get_color(),
            fontsize=fontsize,
            va="center",
            ha="left",
            clip_on=False,
        )


def plot_scenario(
    scenario_df: pd.DataFrame,
    metric: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    line_columns = ["label", "n_perms"]
    if (
        "source_csv" in scenario_df.columns
        and scenario_df["source_csv"].nunique() > 1
    ):
        line_columns.append("source_csv")

    for key, group in scenario_df.groupby(line_columns, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        label_parts = [
            f"label={value}"
            for value in key[:1]
        ]
        label_parts.append(f"n_perms={key[1]}")
        if len(key) > 2:
            label_parts.append(f"csv={key[2]}")
        line_label = ", ".join(label_parts)
        group = group.sort_values("n_jobs")
        ax.plot(
            group["n_jobs"],
            group[metric],
            marker="o",
            linewidth=2,
            label=line_label,
        )

    scenario = scenario_df["scenario"].iloc[0]
    n_cells = int(scenario_df["n_cells"].iloc[0])
    ax.set_title(f"{scenario}: time vs n_jobs ({n_cells} cells)")
    ax.set_xlabel("n_jobs")
    ax.set_ylabel(metric)
    ax.set_xticks(sorted(scenario_df["n_jobs"].unique()))
    ax.grid(True, alpha=0.3)
    ax.margins(x=0.25)
    label_lines_in_place(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_overview(
    df: pd.DataFrame,
    metric: str,
    output_path: Path,
) -> None:
    scenarios = sorted(df["scenario"].unique())
    n_cols = min(2, len(scenarios))
    n_rows = math.ceil(len(scenarios) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7 * n_cols, 4.5 * n_rows),
    )

    if hasattr(axes, "ravel"):
        axes_flat = list(axes.ravel())
    else:
        axes_flat = [axes]

    for ax, scenario in zip(axes_flat, scenarios, strict=False):
        scenario_df = df[df["scenario"] == scenario].copy()
        line_columns = ["label", "n_perms"]
        if scenario_df["source_csv"].nunique() > 1:
            line_columns.append("source_csv")

        for key, group in scenario_df.groupby(line_columns, sort=True):
            if not isinstance(key, tuple):
                key = (key,)
            label_parts = [str(key[0]), f"{key[1]} perms"]
            if len(key) > 2:
                label_parts.append(key[2])
            group = group.sort_values("n_jobs")
            ax.plot(
                group["n_jobs"],
                group[metric],
                marker="o",
                linewidth=2,
                label=" | ".join(label_parts),
            )

        ax.set_title(scenario)
        ax.set_xlabel("n_jobs")
        ax.set_ylabel(metric)
        ax.set_xticks(sorted(scenario_df["n_jobs"].unique()))
        ax.grid(True, alpha=0.3)
        ax.margins(x=0.3)
        label_lines_in_place(ax, fontsize=8)

    for ax in axes_flat[len(scenarios):]:
        ax.remove()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df, csv_paths = load_results(args.input_dir)
    output_paths = build_output_paths(df, args.output_dir)
    confirm_overwrite(output_paths, args.yes)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    overview_path = args.output_dir / "overview_time_vs_n_jobs.png"
    plot_overview(df, args.metric, overview_path)

    for scenario in sorted(df["scenario"].unique()):
        scenario_df = df[df["scenario"] == scenario].copy()
        plot_scenario(
            scenario_df,
            args.metric,
            args.output_dir
            / f"{sanitize_filename(scenario)}_time_vs_n_jobs.png",
        )

    print("Read CSV files:")
    for path in csv_paths:
        print(f"  - {path}")

    print("\nWrote plot files:")
    for path in output_paths:
        print(f"  - {path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        raise SystemExit(130)
