from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def quat_to_euler_enu_xyzw(
    qx: np.ndarray,
    qy: np.ndarray,
    qz: np.ndarray,
    qw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    roll = np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return np.degrees(roll), np.degrees(pitch), np.degrees(np.unwrap(yaw))


def load_ct_attitude(nav_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(nav_path, comments="#")
    if data.ndim != 2:
        raise RuntimeError(f"Unexpected CT nav shape in {nav_path}")

    time_s = data[:, 0]
    if nav_path.name == "nominal_nav.txt":
        qx, qy, qz, qw = data[:, 7], data[:, 8], data[:, 9], data[:, 10]
    else:
        qx, qy, qz, qw = data[:, 4], data[:, 5], data[:, 6], data[:, 7]

    roll_deg, pitch_deg, yaw_deg = quat_to_euler_enu_xyzw(qx, qy, qz, qw)
    return time_s, roll_deg, pitch_deg, yaw_deg


def load_kf_attitude(nav_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(nav_path)
    if data.ndim != 2 or data.shape[1] < 11:
        raise RuntimeError(f"Unexpected KF nav shape in {nav_path}")

    time_s = data[:, 1]
    roll_deg = data[:, 8]
    pitch_deg = data[:, 9]
    yaw_deg = np.degrees(np.unwrap(np.radians(data[:, 10])))
    return time_s, roll_deg, pitch_deg, yaw_deg


def maybe_trim(
    time_s: np.ndarray,
    roll_deg: np.ndarray,
    pitch_deg: np.ndarray,
    yaw_deg: np.ndarray,
    start_time: float | None,
    end_time: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mask = np.ones_like(time_s, dtype=bool)
    if start_time is not None:
        mask &= time_s >= start_time
    if end_time is not None:
        mask &= time_s <= end_time
    return time_s[mask], roll_deg[mask], pitch_deg[mask], yaw_deg[mask]


def save_fig_pickle(fig: plt.Figure, path: Path) -> None:
    with path.open("wb") as f:
        pickle.dump(fig, f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CT vs KF attitude comparison.")
    parser.add_argument("--ct-nav", required=True, type=Path, help="Path to trajectory_enu.txt or nominal_nav.txt")
    parser.add_argument("--kf-nav", required=True, type=Path, help="Path to KF_GINS_Navresult.nav")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory; defaults to CT nav parent")
    parser.add_argument("--start-time", type=float, default=None, help="Optional start time in seconds")
    parser.add_argument("--end-time", type=float, default=None, help="Optional end time in seconds")
    parser.add_argument("--ct-label", default="CT", help="Legend label for CT line")
    parser.add_argument("--kf-label", default="KF-GINS", help="Legend label for KF line")
    args = parser.parse_args()

    ct_time, ct_roll, ct_pitch, ct_yaw = load_ct_attitude(args.ct_nav)
    kf_time, kf_roll, kf_pitch, kf_yaw = load_kf_attitude(args.kf_nav)

    ct_time, ct_roll, ct_pitch, ct_yaw = maybe_trim(
        ct_time, ct_roll, ct_pitch, ct_yaw, args.start_time, args.end_time)
    kf_time, kf_roll, kf_pitch, kf_yaw = maybe_trim(
        kf_time, kf_roll, kf_pitch, kf_yaw, args.start_time, args.end_time)

    output_dir = args.output_dir or args.ct_nav.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    series = [
        ("Roll (deg)", ct_roll, kf_roll),
        ("Pitch (deg)", ct_pitch, kf_pitch),
        ("Yaw (deg)", ct_yaw, kf_yaw),
    ]

    for ax, (ylabel, ct_values, kf_values) in zip(axes, series):
        ax.plot(ct_time, ct_values, linewidth=1.2, label=args.ct_label)
        ax.plot(kf_time, kf_values, linewidth=1.0, label=args.kf_label)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(loc="best")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Attitude Comparison")
    fig.tight_layout()

    png_path = output_dir / "attitude_compare.png"
    fig_path = output_dir / "attitude_compare.fig"
    fig.savefig(png_path, dpi=180)
    save_fig_pickle(fig, fig_path)
    plt.close(fig)

    print(f"Wrote {png_path}")
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
