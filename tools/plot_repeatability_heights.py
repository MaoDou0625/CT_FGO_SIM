from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ANALYSIS_ROOT = Path(r"D:\Code\dataset\YuHangTuiChe\ct_fgo_sim_results\repeatability_analysis")
INPUT_CSV = ANALYSIS_ROOT / "common_position_group_heights.csv"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def plot_height_summary(rows: list[dict[str, str]], value_key: str, ylabel: str, title: str, output_path: Path) -> None:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    position_keys = sorted({(row["lat_q_rad"], row["lon_q_rad"]) for row in rows})
    position_index = {key: idx + 1 for idx, key in enumerate(position_keys)}

    for row in rows:
        grouped[row["segment_id"]].append(row)

    fig, ax = plt.subplots(figsize=(11, 6))
    for segment_id in sorted(grouped):
        segment_rows = sorted(
            grouped[segment_id],
            key=lambda row: position_index[(row["lat_q_rad"], row["lon_q_rad"])],
        )
        x = [position_index[(row["lat_q_rad"], row["lon_q_rad"])] for row in segment_rows]
        y = [float(row[value_key]) for row in segment_rows]
        ax.plot(x, y, marker="o", linewidth=1.2, markersize=3.5, label=segment_id)

    ax.set_xlabel("Common Position Index")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(range(1, len(position_keys) + 1))
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    rows = load_rows(INPUT_CSV)
    plot_height_summary(
        rows,
        value_key="nav_height_mean_m",
        ylabel="Navigation Height (m)",
        title="Navigation Height Summary at Common Positions",
        output_path=ANALYSIS_ROOT / "navigation_height_summary.png",
    )
    plot_height_summary(
        rows,
        value_key="rtk_height_mean_m",
        ylabel="RTK Height (m)",
        title="RTK Height Summary at Common Positions",
        output_path=ANALYSIS_ROOT / "rtk_height_summary.png",
    )


if __name__ == "__main__":
    main()
