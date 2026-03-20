from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_summary(path: Path) -> dict[str, str]:
    summary: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        summary[key.strip()] = value.strip()
    return summary


def write_summary(path: Path, summary: dict[str, str]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")


def load_trajectory(path: Path) -> np.ndarray:
    return np.loadtxt(path, comments="#")


def load_rtk(path: Path) -> np.ndarray:
    return np.loadtxt(path)


def load_imu(path: Path) -> np.ndarray:
    return np.loadtxt(path, comments="#")


def blh_rad_to_local_enu(blh_rad: np.ndarray, origin_blh_rad: np.ndarray) -> np.ndarray:
    lat0, lon0, h0 = origin_blh_rad
    lat = blh_rad[:, 1]
    lon = blh_rad[:, 2]
    h = blh_rad[:, 3]

    a = 6378137.0
    e2 = 0.0066943799901413156

    def rn(phi: np.ndarray) -> np.ndarray:
        return a / np.sqrt(1.0 - e2 * np.sin(phi) ** 2)

    rn0 = rn(np.array([lat0]))[0]
    x0 = (rn0 + h0) * np.cos(lat0) * np.cos(lon0)
    y0 = (rn0 + h0) * np.cos(lat0) * np.sin(lon0)
    z0 = (rn0 * (1.0 - e2) + h0) * np.sin(lat0)

    rn1 = rn(lat)
    x = (rn1 + h) * np.cos(lat) * np.cos(lon)
    y = (rn1 + h) * np.cos(lat) * np.sin(lon)
    z = (rn1 * (1.0 - e2) + h) * np.sin(lat)

    dx = x - x0
    dy = y - y0
    dz = z - z0

    sin_lat = np.sin(lat0)
    cos_lat = np.cos(lat0)
    sin_lon = np.sin(lon0)
    cos_lon = np.cos(lon0)

    ecef_to_enu = np.array(
        [
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ]
    )
    return np.column_stack([dx, dy, dz]) @ ecef_to_enu.T


def moving_average(signal: np.ndarray, window_samples: int) -> np.ndarray:
    window_samples = max(1, int(window_samples))
    kernel = np.ones(window_samples, dtype=np.float64) / float(window_samples)
    return np.convolve(signal, kernel, mode="same")


def detect_motion_onset(imu: np.ndarray) -> dict[str, float] | None:
    if imu.ndim != 2 or imu.shape[0] < 100 or imu.shape[1] < 7:
        return None

    imu_time = imu[:, 0]
    dt = float(np.median(np.diff(imu_time)))
    if dt <= 0.0:
        return None

    baseline_duration_s = min(30.0, max(5.0, 0.2 * float(imu_time[-1] - imu_time[0])))
    baseline_mask = imu_time <= imu_time[0] + baseline_duration_s
    if baseline_mask.sum() < 100:
        return None

    gyro = imu[:, 1:4]
    accel = imu[:, 4:7]
    gyro_mean = gyro[baseline_mask].mean(axis=0)
    accel_mean = accel[baseline_mask].mean(axis=0)

    gyro_dev = np.linalg.norm(gyro - gyro_mean, axis=1)
    accel_dev = np.linalg.norm(accel - accel_mean, axis=1)

    smooth_window = max(1, int(round(0.5 / dt)))
    gyro_dev_smooth = moving_average(gyro_dev, smooth_window)
    accel_dev_smooth = moving_average(accel_dev, smooth_window)

    gyro_base = gyro_dev_smooth[baseline_mask]
    accel_base = accel_dev_smooth[baseline_mask]
    gyro_threshold = float(gyro_base.mean() + 5.0 * gyro_base.std())
    accel_threshold = float(accel_base.mean() + 5.0 * accel_base.std())

    motion_mask = (gyro_dev_smooth > gyro_threshold) | (accel_dev_smooth > accel_threshold)
    persistence_samples = max(1, int(round(0.5 / dt)))
    run_length = 0
    for i, moving in enumerate(motion_mask):
        run_length = run_length + 1 if moving else 0
        if run_length >= persistence_samples:
            onset_index = i - persistence_samples + 1
            onset_time = float(imu_time[onset_index])
            return {
                "motion_onset_time_s": onset_time,
                "motion_onset_offset_s": onset_time - float(imu_time[0]),
                "motion_gyro_threshold_rps": gyro_threshold,
                "motion_accel_threshold_mps2": accel_threshold,
            }
    return None


def annotate_motion_onset(axes: list[plt.Axes] | np.ndarray, motion_onset_time_s: float) -> None:
    for ax in np.atleast_1d(axes):
        ax.axvline(motion_onset_time_s, color="tab:red", linestyle="--", linewidth=1.0, alpha=0.9)
        ylim = ax.get_ylim()
        y_text = ylim[0] + 0.92 * (ylim[1] - ylim[0])
        ax.text(
            motion_onset_time_s,
            y_text,
            f"motion start\n{motion_onset_time_s:.2f}s",
            color="tab:red",
            fontsize=8,
            ha="left",
            va="top",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.5},
        )


def compute_metrics(traj_time: np.ndarray, traj_e: np.ndarray, traj_n: np.ndarray, traj_u: np.ndarray,
                    rtk_time: np.ndarray, rtk_e: np.ndarray, rtk_n: np.ndarray, rtk_u: np.ndarray) -> dict[str, float]:
    rtk_e_interp = np.interp(traj_time, rtk_time, rtk_e)
    rtk_n_interp = np.interp(traj_time, rtk_time, rtk_n)
    rtk_u_interp = np.interp(traj_time, rtk_time, rtk_u)

    err_e = traj_e - rtk_e_interp
    err_n = traj_n - rtk_n_interp
    err_u = traj_u - rtk_u_interp
    err_h = np.sqrt(err_e ** 2 + err_n ** 2)
    err_3d = np.sqrt(err_h ** 2 + err_u ** 2)

    return {
        "matched_points": int(traj_time.shape[0]),
        "rmse_e_m": float(np.sqrt(np.mean(err_e ** 2))),
        "rmse_n_m": float(np.sqrt(np.mean(err_n ** 2))),
        "rmse_u_m": float(np.sqrt(np.mean(err_u ** 2))),
        "rmse_h_m": float(np.sqrt(np.mean(err_h ** 2))),
        "rmse_3d_m": float(np.sqrt(np.mean(err_3d ** 2))),
        "mean_e_m": float(np.mean(err_e)),
        "mean_n_m": float(np.mean(err_n)),
        "mean_u_m": float(np.mean(err_u)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    summary_path = output_dir / "run_summary.txt"
    summary = load_summary(summary_path)
    trajectory = load_trajectory(output_dir / "trajectory_enu.txt")
    rtk = load_rtk(Path(summary["gnss_file"]))
    imu = load_imu(Path(summary["imu_file"])) if "imu_file" in summary else None
    # Use the first RTK sample as the ENU origin to match the solver exactly.
    origin_blh_rad = rtk[0, 1:4]
    rtk_enu = blh_rad_to_local_enu(rtk, origin_blh_rad)
    motion_info = detect_motion_onset(imu) if imu is not None else None

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    traj_time = trajectory[:, 0]
    traj_e = trajectory[:, 1]
    traj_n = trajectory[:, 2]
    traj_u = trajectory[:, 3]
    rtk_time = rtk[:, 0]
    rtk_e = rtk_enu[:, 0]
    rtk_n = rtk_enu[:, 1]
    rtk_u = rtk_enu[:, 2]

    rtk_e_interp = np.interp(traj_time, rtk_time, rtk_e)
    rtk_n_interp = np.interp(traj_time, rtk_time, rtk_n)
    rtk_u_interp = np.interp(traj_time, rtk_time, rtk_u)
    err_e = traj_e - rtk_e_interp
    err_n = traj_n - rtk_n_interp
    err_u = traj_u - rtk_u_interp
    err_h = np.sqrt(err_e ** 2 + err_n ** 2)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rtk_e, rtk_n, label="RTK", linewidth=1.5)
    ax.plot(traj_e, traj_n, label="CT_FGO_SIM", linewidth=1.2)
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("Horizontal Comparison")
    ax.axis("equal")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "horizontal_compare.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(traj_time, err_e, label="East Error", linewidth=1.0)
    axes[0].plot(traj_time, err_n, label="North Error", linewidth=1.0)
    axes[0].set_ylabel("ENU Error (m)")
    axes[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    axes[0].legend()
    axes[1].plot(traj_time, err_h, label="Horizontal Error Norm", linewidth=1.0)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Horizontal Error (m)")
    axes[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    axes[1].legend()
    if motion_info is not None:
        annotate_motion_onset(axes, motion_info["motion_onset_time_s"])
    fig.suptitle("Horizontal Error")
    fig.tight_layout()
    fig.savefig(plot_dir / "horizontal_error.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rtk_time, rtk_u, label="RTK Height", linewidth=1.5)
    ax.plot(traj_time, traj_u, label="CT_FGO_SIM Height", linewidth=1.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Height / Up (m)")
    ax.set_title("Height Comparison")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "height_compare.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(traj_time, err_u, linewidth=1.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Height Error (m)")
    ax.set_title("Height Error")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    if motion_info is not None:
        annotate_motion_onset([ax], motion_info["motion_onset_time_s"])
    fig.tight_layout()
    fig.savefig(plot_dir / "height_error.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(traj_time, err_e, linewidth=1.0)
    axes[0].set_ylabel("East Error (m)")
    axes[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    axes[1].plot(traj_time, err_n, linewidth=1.0)
    axes[1].set_ylabel("North Error (m)")
    axes[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    axes[2].plot(traj_time, err_u, linewidth=1.0)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Up Error (m)")
    axes[2].grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    if motion_info is not None:
        annotate_motion_onset(axes, motion_info["motion_onset_time_s"])
    fig.suptitle("ENU Error Time Series")
    fig.tight_layout()
    fig.savefig(plot_dir / "enu_error_timeseries.png", dpi=180)
    plt.close(fig)

    metrics = compute_metrics(traj_time, traj_e, traj_n, traj_u, rtk_time, rtk_e, rtk_n, rtk_u)
    metrics_path = output_dir / "metrics_summary.txt"
    with metrics_path.open("w", encoding="utf-8", newline="\n") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
        if motion_info is not None:
            for key, value in motion_info.items():
                f.write(f"{key}: {value}\n")

    summary.update({key: str(value) for key, value in metrics.items()})
    if motion_info is not None:
        summary.update({key: str(value) for key, value in motion_info.items()})
    write_summary(summary_path, summary)


if __name__ == "__main__":
    main()
