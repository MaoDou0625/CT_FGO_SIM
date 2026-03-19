from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


INPUT_ROOT = Path(r"D:\Code\dataset\YuHangTuiChe\ct_fgo_sim_use")
OUTPUT_ROOT = Path(r"D:\Code\dataset\YuHangTuiChe\ct_fgo_sim_results\imu_accel_smoothing")
SMOOTH_WINDOW_S = 10.0


def find_segments(root: Path) -> list[tuple[str, str, Path]]:
    segments: list[tuple[str, str, Path]] = []
    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for segment_dir in sorted(dataset_dir.glob("transformed1cut*")):
            imu_path = segment_dir / "imu_ct_fgo_sim.txt"
            if imu_path.exists():
                segments.append((dataset_dir.name, segment_dir.name, imu_path))
    return segments


def moving_average(signal: np.ndarray, window_samples: int) -> np.ndarray:
    window_samples = max(1, int(window_samples))
    kernel = np.ones(window_samples, dtype=np.float64) / float(window_samples)
    return np.convolve(signal, kernel, mode="same")


def load_and_smooth(imu_path: Path) -> tuple[np.ndarray, np.ndarray]:
    imu = np.loadtxt(imu_path, comments="#")
    t = imu[:, 0]
    accel = imu[:, 4:7]
    dt = float(np.median(np.diff(t)))
    window_samples = max(1, int(round(SMOOTH_WINDOW_S / dt)))
    smoothed = np.column_stack([moving_average(accel[:, i], window_samples) for i in range(3)])
    return t, smoothed


def plot_single(dataset: str, group: str, time_s: np.ndarray, accel_smooth: np.ndarray, output_path: Path) -> None:
    rel_t = time_s - time_s[0]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(rel_t, accel_smooth[:, 0], label="acc_x", linewidth=1.2)
    ax.plot(rel_t, accel_smooth[:, 1], label="acc_y", linewidth=1.2)
    ax.plot(rel_t, accel_smooth[:, 2], label="acc_z", linewidth=1.2)
    ax.set_xlabel("Time Since Start (s)")
    ax.set_ylabel("Smoothed Accel (m/s^2)")
    ax.set_title(f"{dataset} / {group} IMU Accel Mean (10 s Smooth)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_summary(all_data: list[tuple[str, str, np.ndarray, np.ndarray]], output_path: Path) -> None:
    fig, axes = plt.subplots(len(all_data), 1, figsize=(12, 3.2 * len(all_data)), sharex=False)
    if len(all_data) == 1:
        axes = [axes]

    for ax, (dataset, group, time_s, accel_smooth) in zip(axes, all_data):
        rel_t = time_s - time_s[0]
        ax.plot(rel_t, accel_smooth[:, 0], label="acc_x", linewidth=1.0)
        ax.plot(rel_t, accel_smooth[:, 1], label="acc_y", linewidth=1.0)
        ax.plot(rel_t, accel_smooth[:, 2], label="acc_z", linewidth=1.0)
        ax.set_ylabel("m/s^2")
        ax.set_title(f"{dataset} / {group}")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time Since Start (s)")
    fig.suptitle("IMU Accelerometer Mean (10 s Smooth)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_data: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    for dataset, group, imu_path in find_segments(INPUT_ROOT):
        time_s, accel_smooth = load_and_smooth(imu_path)
        all_data.append((dataset, group, time_s, accel_smooth))
        single_name = f"{dataset}_{group}_accel_10s_smooth.png"
        plot_single(dataset, group, time_s, accel_smooth, OUTPUT_ROOT / single_name)

    plot_summary(all_data, OUTPUT_ROOT / "imu_accel_10s_smooth_summary.png")


if __name__ == "__main__":
    main()
