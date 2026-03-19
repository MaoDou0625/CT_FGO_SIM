from __future__ import annotations

import argparse
import subprocess
import tempfile
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def quat_to_rotmat_xyzw(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    q = np.array([qw, qx, qy, qz], dtype=float)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ])


def rotmat_to_rotvec(rot: np.ndarray) -> np.ndarray:
    trace = np.clip((np.trace(rot) - 1.0) * 0.5, -1.0, 1.0)
    angle = np.arccos(trace)
    if angle < 1.0e-12:
        return np.zeros(3)
    axis = np.array([
        rot[2, 1] - rot[1, 2],
        rot[0, 2] - rot[2, 0],
        rot[1, 0] - rot[0, 1],
    ]) / (2.0 * np.sin(angle))
    return axis * angle


def maybe_trim(time_s: np.ndarray, values: np.ndarray, start_time: float | None, end_time: float | None) -> tuple[np.ndarray, np.ndarray]:
    mask = np.ones_like(time_s, dtype=bool)
    if start_time is not None:
        mask &= time_s >= start_time
    if end_time is not None:
        mask &= time_s <= end_time
    return time_s[mask], values[mask]


def save_matlab_fig(
    output_fig_path: Path,
    dtheta_time_s: np.ndarray,
    dtheta_deg: np.ndarray,
    dbg_time_s: np.ndarray,
    dbg_rps: np.ndarray,
) -> None:
    output_fig_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ct_fgo_delta_fig_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        dtheta_csv = tmp_dir / "dtheta_estimates.csv"
        dbg_csv = tmp_dir / "dbg_estimates.csv"
        script_path = tmp_dir / "make_delta_estimates_fig.m"
        np.savetxt(dtheta_csv, np.column_stack((dtheta_time_s, dtheta_deg)), delimiter=",")
        np.savetxt(dbg_csv, np.column_stack((dbg_time_s, dbg_rps)), delimiter=",")

        script = textwrap.dedent(
            f"""
            dtheta = readmatrix('{dtheta_csv.as_posix()}');
            dbg = readmatrix('{dbg_csv.as_posix()}');
            fig = figure('Visible', 'off', 'Position', [100, 100, 1200, 780]);
            top_labels = {{'dtheta x (deg)', 'dtheta y (deg)', 'dtheta z (deg)'}};
            bottom_labels = {{'dbg x (rad/s)', 'dbg y (rad/s)', 'dbg z (rad/s)'}};
            for idx = 1:3
                subplot(2,3,idx);
                plot(dtheta(:,1), dtheta(:,idx+1), 'LineWidth', 1.1);
                grid on;
                ylabel(top_labels{{idx}});
            end
            for idx = 1:3
                subplot(2,3,idx+3);
                plot(dbg(:,1), dbg(:,idx+1), 'LineWidth', 1.1);
                grid on;
                ylabel(bottom_labels{{idx}});
            end
            xlabel('Time (s)');
            sgtitle('Estimated dtheta / dbg');
            savefig(fig, '{output_fig_path.as_posix()}');
            close(fig);
            exit;
            """
        ).strip()
        script_path.write_text(script, encoding="utf-8")
        subprocess.run(["matlab", "-batch", f"run('{script_path.as_posix()}')"], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot estimated dtheta and dbg.")
    parser.add_argument("--delta-file", type=Path, default=None, help="Path to delta_estimates.txt")
    parser.add_argument("--nominal-nav", type=Path, default=None, help="Fallback path to nominal_nav.txt")
    parser.add_argument("--trajectory", type=Path, default=None, help="Fallback path to trajectory_enu.txt")
    parser.add_argument("--bias-nodes", type=Path, default=None, help="Fallback path to bias_nodes.txt")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--start-time", type=float, default=None)
    parser.add_argument("--end-time", type=float, default=None)
    parser.add_argument("--matlab-fig", action="store_true", help="Also export a MATLAB-readable .fig")
    args = parser.parse_args()

    if args.delta_file is not None and args.delta_file.exists():
        data = np.loadtxt(args.delta_file, comments="#")
        if data.ndim != 2 or data.shape[1] < 7:
            raise RuntimeError(f"Unexpected delta estimate format in {args.delta_file}")
        time_s = data[:, 0]
        dtheta_deg = np.degrees(data[:, 1:4])
        dbg_rps = data[:, 4:7]
        stacked = np.column_stack((dtheta_deg, dbg_rps))
        time_s, stacked = maybe_trim(time_s, stacked, args.start_time, args.end_time)
        if time_s.size == 0:
            raise RuntimeError("No delta estimate samples remain after trimming")
        dtheta_deg = stacked[:, 0:3]
        dbg_rps = stacked[:, 3:6]
        output_dir = args.output_dir or args.delta_file.parent
    else:
        if args.nominal_nav is None or args.trajectory is None or args.bias_nodes is None:
            raise RuntimeError("Either --delta-file or all of --nominal-nav/--trajectory/--bias-nodes must be provided")
        nominal = np.loadtxt(args.nominal_nav, comments="#")
        traj = np.loadtxt(args.trajectory, comments="#")
        bias = np.loadtxt(args.bias_nodes, comments="#")
        if nominal.ndim != 2 or traj.ndim != 2 or bias.ndim != 2:
            raise RuntimeError("Unexpected input shape for fallback delta plotting")

        nominal_time = nominal[:, 0]
        traj_time = traj[:, 0]
        dtheta_deg = np.zeros((traj.shape[0], 3))
        for idx in range(traj.shape[0]):
            nominal_idx = int(np.argmin(np.abs(nominal_time - traj_time[idx])))
            rot_nom = quat_to_rotmat_xyzw(nominal[nominal_idx, 7], nominal[nominal_idx, 8], nominal[nominal_idx, 9], nominal[nominal_idx, 10])
            rot_full = quat_to_rotmat_xyzw(traj[idx, 4], traj[idx, 5], traj[idx, 6], traj[idx, 7])
            rot_delta = rot_nom.T @ rot_full
            dtheta_deg[idx, :] = np.degrees(rotmat_to_rotvec(rot_delta))
        dbg_time = bias[:, 0]
        dbg_rps = bias[:, 1:4]

        traj_time, dtheta_deg = maybe_trim(traj_time, dtheta_deg, args.start_time, args.end_time)
        dbg_time, dbg_rps = maybe_trim(dbg_time, dbg_rps, args.start_time, args.end_time)
        if traj_time.size == 0 or dbg_time.size == 0:
            raise RuntimeError("No fallback delta samples remain after trimming")

        output_dir = args.output_dir or args.nominal_nav.parent
        time_s = traj_time
        dbg_time = dbg_time

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    dtheta_labels = ["dtheta x (deg)", "dtheta y (deg)", "dtheta z (deg)"]
    dbg_labels = ["dbg x (rad/s)", "dbg y (rad/s)", "dbg z (rad/s)"]
    for idx in range(3):
        axes[0, idx].plot(time_s, dtheta_deg[:, idx], linewidth=1.0)
        axes[0, idx].set_ylabel(dtheta_labels[idx])
        axes[0, idx].grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        dbg_time_axis = time_s if dbg_rps.shape[0] == time_s.shape[0] else dbg_time
        axes[1, idx].plot(dbg_time_axis, dbg_rps[:, idx], linewidth=1.0)
        axes[1, idx].set_ylabel(dbg_labels[idx])
        axes[1, idx].grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        axes[1, idx].set_xlabel("Time (s)")

    fig.suptitle("Estimated dtheta / dbg")
    fig.tight_layout()

    png_path = output_dir / "delta_estimates.png"
    fig_path = output_dir / "delta_estimates.fig"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    if args.matlab_fig:
        dbg_time_for_fig = dbg_time if 'dbg_time' in locals() else time_s
        save_matlab_fig(fig_path, time_s, dtheta_deg, dbg_time_for_fig, dbg_rps)

    print(f"Wrote {png_path}")
    if args.matlab_fig:
        print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
