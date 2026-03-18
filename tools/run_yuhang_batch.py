from __future__ import annotations

import csv
import subprocess
from pathlib import Path


REPO_ROOT = Path(r"D:\Code\CT_FGO_SIM")
EXE_PATH = REPO_ROOT / "build" / "Release" / "ct_fgo_sim_main.exe"
PLOT_SCRIPT = REPO_ROOT / "tools" / "plot_outputs.py"
INPUT_ROOT = Path(r"D:\Code\dataset\YuHangTuiChe\ct_fgo_sim_use")
OUTPUT_ROOT = Path(r"D:\Code\dataset\YuHangTuiChe\ct_fgo_sim_results")

CONFIG_TEMPLATE = """gnssfile: "{gnssfile}"
outputpath: "{outputpath}"

imu_main:
  file: "{imufile}"
  columns: 7
  rate_hz: 1000.0
  antlever: [0.0, 0.0, 0.0]

starttime: 0.0
endtime: 0.0
aligntime: 100.0
kf_interval_sec: 0.1
gnss_sigma_horizontal_m: 0.02
gnss_sigma_vertical_m: 0.03
imu_sigma_accel_mps2: 0.2
imu_sigma_gyro_rps: 0.01
gyro_bias_rw_sigma: 1.0e-4
accel_bias_rw_sigma: 1.0e-3
bias_tau_s: 3600.0
imu_stride: 10
solver_max_iterations: 15
"""


def find_segments(root: Path) -> list[Path]:
    segments: list[Path] = []
    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for segment_dir in sorted(dataset_dir.glob("transformed1cut*")):
            if not segment_dir.is_dir():
                continue
            if not (segment_dir / "rtk_ct_fgo_sim.txt").exists():
                continue
            if not (segment_dir / "imu_ct_fgo_sim.txt").exists():
                continue
            segments.append(segment_dir)
    return segments


def load_summary(path: Path) -> dict[str, str]:
    summary: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        summary[key.strip()] = value.strip()
    return summary


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    for segment_dir in find_segments(INPUT_ROOT):
        dataset = segment_dir.parent.name
        group = segment_dir.name
        output_dir = OUTPUT_ROOT / dataset / group
        output_dir.mkdir(parents=True, exist_ok=True)

        config_path = output_dir / "ct_fgo_sim.yaml"
        config_path.write_text(
            CONFIG_TEMPLATE.format(
                gnssfile=(segment_dir / "rtk_ct_fgo_sim.txt").as_posix(),
                imufile=(segment_dir / "imu_ct_fgo_sim.txt").as_posix(),
                outputpath=output_dir.as_posix(),
            ),
            encoding="utf-8",
        )

        subprocess.run([str(EXE_PATH), str(config_path)], check=True)
        subprocess.run(["python", str(PLOT_SCRIPT), "--output-dir", str(output_dir)], check=True)

        summary = load_summary(output_dir / "run_summary.txt")
        rows.append(
            {
                "dataset": dataset,
                "group": group,
                "matched_points": summary.get("matched_points", ""),
                "rmse_e_m": summary.get("rmse_e_m", ""),
                "rmse_n_m": summary.get("rmse_n_m", ""),
                "rmse_u_m": summary.get("rmse_u_m", ""),
                "rmse_h_m": summary.get("rmse_h_m", ""),
                "rmse_3d_m": summary.get("rmse_3d_m", ""),
            }
        )

    summary_csv = OUTPUT_ROOT / "yuhang_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "group", "matched_points", "rmse_e_m", "rmse_n_m", "rmse_u_m", "rmse_h_m", "rmse_3d_m"],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
