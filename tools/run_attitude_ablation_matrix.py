from __future__ import annotations

import csv
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


CT_REPO = Path(r"D:\Code\CT_FGO_SIM")
CT_EXE = CT_REPO / "build" / "Release" / "ct_fgo_sim_main.exe"
CT_PLOT = CT_REPO / "tools" / "plot_outputs.py"
KF_REPO = Path(r"D:\Code\kf_gins_used_in_paper")
KF_EXE = KF_REPO / "bin" / "Release" / "KF-GINS.exe"

SEGMENT_DIR = Path(r"D:\Code\dataset\YuHangTuiChe\ct_fgo_sim_use\20260122_121901_use\transformed1cut1")
KF_SEGMENT_DIR = Path(r"D:\Code\dataset\YuHangTuiChe\kf_gins_use\20260122_121901_use\transformed1cut1")
OUTPUT_ROOT = Path(r"D:\Code\dataset\YuHangTuiChe\attitude_ablation_2520")

START_TIME = 2387.9308
END_TIME = 2520.0


CT_CONFIG_TEMPLATE = """gnssfile: "{gnssfile}"
outputpath: "{outputpath}"

imu_main:
  file: "{imufile}"
  columns: 7
  rate_hz: 1000.0
  antlever: [0.0, 0.0, 0.0]

body_frame:
  q_body_imu_xyzw: [0.0, 0.0, 0.0, 1.0]
  q_body_imu_prior_sigma_rad: 0.1
  enable_nhc: {enable_nhc}
  estimate_q_body_imu: false
  nhc_enable_vx: false
  nhc_enable_vy: false
  nhc_enable_vz: true
  nhc_target_vx_mps: 0.0
  nhc_target_vy_mps: 0.0
  nhc_target_vz_mps: 0.0
  nhc_sigma_vx_mps: 1000000.0
  nhc_sigma_vy_mps: 1000000.0
  nhc_sigma_vz_mps: {nhc_sigma_vz_mps}

starttime: {starttime}
endtime: {endtime}
aligntime: 100.0
kf_interval_sec: 0.1
gnss_sigma_horizontal_m: {gnss_sigma_horizontal_m}
gnss_sigma_vertical_m: {gnss_sigma_vertical_m}
imu_sigma_accel_mps2: 0.2
imu_sigma_gyro_rps: 0.01
gyro_bias_rw_sigma: 1.0e-4
accel_bias_rw_sigma: 1.0e-3
bias_tau_s: 3600.0
imu_stride: 10
solver_max_iterations: 15
use_gnss_factors: {use_gnss_factors}
use_imu_factors: {use_imu_factors}
"""


KF_CONFIG_TEMPLATE = """imupath: "{imupath}"
gnsspath: "{gnsspath}"
outputpath: "{outputpath}"

imudatalen: 7
imudatarate: 1000

starttime: {starttime}
endtime: {endtime}

initpos: [ {init_lat_deg}, {init_lon_deg}, {init_h_m} ]
initvel: [ 0.0, 0.0, 0.0 ]
initatt: [ {init_roll_deg}, {init_pitch_deg}, {init_yaw_deg} ]

initgyrbias: [ 0, 0, 0 ]
initaccbias: [ 0, 0, 0 ]
initgyrscale: [ 0, 0, 0 ]
initaccscale: [ 0, 0, 0 ]

initposstd: [ 1.0, 1.0, 2.0 ]
initvelstd: [ 0.1, 0.1, 0.1 ]
initattstd: [ 0.5, 0.5, 2.0 ]

imunoise:
  arw: [0.24, 0.24, 0.24]
  vrw: [0.24, 0.24, 0.24]
  gbstd: [50.0, 50.0, 50.0]
  abstd: [250.0, 250.0, 250.0]
  gsstd: [1000.0, 1000.0, 1000.0]
  asstd: [1000.0, 1000.0, 1000.0]
  corrtime: 1.0

antlever: [ 0.0, 0.0, 0.0 ]
"""


@dataclass
class Case:
    name: str
    kind: str
    use_gnss_factors: bool = True
    use_imu_factors: bool = True
    enable_nhc: bool = False
    nhc_sigma_vz_mps: float = 0.2
    gnss_sigma_horizontal_m: float = 0.02
    gnss_sigma_vertical_m: float = 0.03


def quat_to_euler_enu_xyzw(qx: np.ndarray, qy: np.ndarray, qz: np.ndarray, qw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    roll = np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return np.degrees(roll), np.degrees(pitch), np.degrees(np.unwrap(yaw))


def compute_case_metrics(time_s: np.ndarray, roll_deg: np.ndarray, pitch_deg: np.ndarray, yaw_deg: np.ndarray, height_m: np.ndarray) -> dict[str, float]:
    def value_at(arr: np.ndarray, t: float) -> float:
        return float(np.interp(t, time_s, arr))

    static_mask = (time_s >= 2388.0) & (time_s <= 2488.0)
    metrics = {
        "yaw_delta_2490_2492_deg": value_at(yaw_deg, 2492.0) - value_at(yaw_deg, 2490.0),
        "yaw_delta_2490_2495_deg": value_at(yaw_deg, 2495.0) - value_at(yaw_deg, 2490.0),
        "yaw_delta_2490_2500_deg": value_at(yaw_deg, 2500.0) - value_at(yaw_deg, 2490.0),
        "roll_delta_2490_2500_deg": value_at(roll_deg, 2500.0) - value_at(roll_deg, 2490.0),
        "pitch_delta_2490_2500_deg": value_at(pitch_deg, 2500.0) - value_at(pitch_deg, 2490.0),
        "height_change_2490_2500_m": value_at(height_m, 2500.0) - value_at(height_m, 2490.0),
        "height_std_static_2388_2488_m": float(np.std(height_m[static_mask])) if np.any(static_mask) else float("nan"),
    }

    local_mask = (time_s >= 2490.0) & (time_s <= 2500.0)
    if np.count_nonzero(local_mask) >= 2:
        dt = np.diff(time_s[local_mask])
        dyaw = np.diff(yaw_deg[local_mask])
        metrics["max_abs_yaw_rate_2490_2500_degps"] = float(np.max(np.abs(dyaw / dt)))
    else:
        metrics["max_abs_yaw_rate_2490_2500_degps"] = float("nan")
    return metrics


def load_ct_case_metrics(output_dir: Path) -> dict[str, float]:
    traj = np.loadtxt(output_dir / "trajectory_enu.txt", comments="#")
    time_s = traj[:, 0]
    height_m = traj[:, 3]
    roll_deg, pitch_deg, yaw_deg = quat_to_euler_enu_xyzw(traj[:, 4], traj[:, 5], traj[:, 6], traj[:, 7])
    return compute_case_metrics(time_s, roll_deg, pitch_deg, yaw_deg, height_m)


def load_kf_case_metrics(output_dir: Path) -> dict[str, float]:
    nav = np.loadtxt(output_dir / "KF_GINS_Navresult.nav")
    time_s = nav[:, 1]
    roll_deg = nav[:, 8]
    pitch_deg = nav[:, 9]
    yaw_deg = np.degrees(np.unwrap(np.radians(nav[:, 10])))
    height_m = nav[:, 4]
    return compute_case_metrics(time_s, roll_deg, pitch_deg, yaw_deg, height_m)


def read_conversion_summary(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data


def run_ct_case(case: Case) -> dict[str, float]:
    output_dir = OUTPUT_ROOT / case.name
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "ct_fgo_sim.yaml"
    config_path.write_text(
        CT_CONFIG_TEMPLATE.format(
            gnssfile=(SEGMENT_DIR / "rtk_ct_fgo_sim.txt").as_posix(),
            imufile=(SEGMENT_DIR / "imu_ct_fgo_sim.txt").as_posix(),
            outputpath=output_dir.as_posix(),
            enable_nhc=str(case.enable_nhc).lower(),
            nhc_sigma_vz_mps=case.nhc_sigma_vz_mps,
            starttime=START_TIME,
            endtime=END_TIME,
            gnss_sigma_horizontal_m=case.gnss_sigma_horizontal_m,
            gnss_sigma_vertical_m=case.gnss_sigma_vertical_m,
            use_gnss_factors=str(case.use_gnss_factors).lower(),
            use_imu_factors=str(case.use_imu_factors).lower(),
        ),
        encoding="utf-8",
    )
    subprocess.run([str(CT_EXE), str(config_path)], check=True)
    subprocess.run(["python", str(CT_PLOT), "--output-dir", str(output_dir)], check=True)
    metrics = load_ct_case_metrics(output_dir)
    metrics["case"] = case.name
    metrics["kind"] = case.kind
    return metrics


def run_kf_case(case_name: str) -> dict[str, float]:
    output_dir = OUTPUT_ROOT / case_name
    output_dir.mkdir(parents=True, exist_ok=True)
    conv = read_conversion_summary(KF_SEGMENT_DIR / "conversion_summary.txt")
    lat_deg, lon_deg, h_m = [x.strip() for x in conv["init_pos_deg_m"].split(",")]
    roll_deg, pitch_deg, yaw_deg = [x.strip() for x in conv["init_att_deg_roll_pitch_yaw"].split(",")]
    config_path = output_dir / "kf_gins.yaml"
    config_path.write_text(
        KF_CONFIG_TEMPLATE.format(
            imupath=(KF_SEGMENT_DIR / "imu_kfgins.txt").as_posix(),
            gnsspath=(KF_SEGMENT_DIR / "rtk_kfgins.txt").as_posix(),
            outputpath=output_dir.as_posix(),
            starttime=START_TIME,
            endtime=END_TIME,
            init_lat_deg=lat_deg,
            init_lon_deg=lon_deg,
            init_h_m=h_m,
            init_roll_deg=roll_deg,
            init_pitch_deg=pitch_deg,
            init_yaw_deg=yaw_deg,
        ),
        encoding="utf-8",
    )
    subprocess.run([str(KF_EXE), str(config_path)], check=True)
    metrics = load_kf_case_metrics(output_dir)
    metrics["case"] = case_name
    metrics["kind"] = "kf_gins"
    return metrics


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    cases = [
        Case("ct_rtk_only", "ct_fgo", use_gnss_factors=True, use_imu_factors=False),
        Case("ct_imu_only", "ct_fgo", use_gnss_factors=False, use_imu_factors=True),
        Case("ct_rtk_imu", "ct_fgo", use_gnss_factors=True, use_imu_factors=True),
        Case("ct_rtk_imu_nhc_z", "ct_fgo", use_gnss_factors=True, use_imu_factors=True, enable_nhc=True, nhc_sigma_vz_mps=0.2),
        Case("ct_rtk_imu_weak_gnss_z", "ct_fgo", use_gnss_factors=True, use_imu_factors=True, gnss_sigma_vertical_m=0.10),
    ]

    rows: list[dict[str, float]] = [run_kf_case("kf_gins_baseline")]
    for case in cases:
        rows.append(run_ct_case(case))

    summary_path = OUTPUT_ROOT / "attitude_ablation_summary.csv"
    fieldnames = [
        "case",
        "kind",
        "yaw_delta_2490_2492_deg",
        "yaw_delta_2490_2495_deg",
        "yaw_delta_2490_2500_deg",
        "max_abs_yaw_rate_2490_2500_degps",
        "roll_delta_2490_2500_deg",
        "pitch_delta_2490_2500_deg",
        "height_change_2490_2500_m",
        "height_std_static_2388_2488_m",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
