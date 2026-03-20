from __future__ import annotations

import argparse
import pickle
import subprocess
import tempfile
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


R_ENU_FROM_NED = np.array([
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
])


def wrap_degrees(angle_deg: np.ndarray) -> np.ndarray:
    return (angle_deg + 180.0) % 360.0 - 180.0


def heading_degrees_0_360(angle_deg: np.ndarray) -> np.ndarray:
    angle_deg = np.asarray(angle_deg, dtype=float)
    out = np.mod(angle_deg, 360.0)
    out[np.isnan(angle_deg)] = np.nan
    return out


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
    return np.degrees(roll), np.degrees(pitch), wrap_degrees(np.degrees(yaw))


def quat_to_rotmat_xyzw(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )


def euler_to_rotmat(roll_rad: float, pitch_rad: float, yaw_rad: float) -> np.ndarray:
    cr, sr = np.cos(roll_rad), np.sin(roll_rad)
    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=float)


def rotmat_to_euler(rot: np.ndarray) -> tuple[float, float, float]:
    roll = np.arctan2(rot[2, 1], rot[2, 2])
    pitch = -np.arcsin(np.clip(rot[2, 0], -1.0, 1.0))
    yaw = np.arctan2(rot[1, 0], rot[0, 0])
    return roll, pitch, yaw


def ned_velocity_to_enu(v_ned: np.ndarray) -> np.ndarray:
    # KF-GINS nav output stores velocity as north-east-down.
    return np.column_stack((v_ned[:, 1], v_ned[:, 0], -v_ned[:, 2]))


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


def save_fig_pickle(fig: plt.Figure, path: Path) -> None:
    with path.open("wb") as f:
        pickle.dump(fig, f)


def save_matlab_fig(output_fig_path: Path, arrays: list[np.ndarray], labels: list[str], title: str) -> None:
    output_fig_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ct_fgo_compare_fig_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        csv_paths = []
        for idx, array in enumerate(arrays):
            csv_path = tmp_dir / f"series_{idx}.csv"
            np.savetxt(csv_path, array, delimiter=",")
            csv_paths.append(csv_path)

        script_path = tmp_dir / "make_compare_fig.m"
        script_lines = [
            "fig = figure('Visible', 'off', 'Position', [100, 100, 1200, 780]);",
            "tiledlayout(3,1);",
        ]
        for axis_idx in range(3):
            script_lines.append(f"nexttile({axis_idx + 1});")
            for idx, (csv_path, label) in enumerate(zip(csv_paths, labels), start=1):
                script_lines.append(f"data{idx} = readmatrix('{csv_path.as_posix()}');")
                script_lines.append(
                    f"plot(data{idx}(:,1), data{idx}(:,{axis_idx + 2}), 'LineWidth', 1.1, 'DisplayName', '{label}'); hold on;")
            script_lines.append("grid on; legend('Location','best');")
        script_lines.append(f"sgtitle('{title}');")
        script_lines.append(f"savefig(fig, '{output_fig_path.as_posix()}');")
        script_lines.append("close(fig); exit;")
        script_path.write_text("\n".join(script_lines), encoding="utf-8")
        subprocess.run(["matlab", "-batch", f"run('{script_path.as_posix()}')"], check=True)


def load_ct_pose(traj_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(traj_path, comments="#")
    time_s = data[:, 0]
    pos_enu = data[:, 1:4]
    roll_out = np.empty_like(time_s)
    pitch_out = np.empty_like(time_s)
    yaw_out = np.empty_like(time_s)
    for idx in range(len(time_s)):
        rot_ct_enu_flu = quat_to_rotmat_xyzw(data[idx, 4], data[idx, 5], data[idx, 6], data[idx, 7])
        # CT output quaternions already use the vehicle/body convention expected here.
        # Only rotate the navigation basis ENU -> NED for direct NED plotting.
        rot_ct_ned_frd = R_ENU_FROM_NED.T @ rot_ct_enu_flu
        roll_rad, pitch_rad, yaw_rad = rotmat_to_euler(rot_ct_ned_frd)
        roll_out[idx] = np.degrees(roll_rad)
        pitch_out[idx] = np.degrees(pitch_rad)
        yaw_out[idx] = np.degrees(yaw_rad)
    yaw_out = heading_degrees_0_360(yaw_out)
    attitude_deg = np.column_stack((roll_out, pitch_out, yaw_out))
    return time_s, pos_enu, attitude_deg


def load_ct_velocity(delta_path: Path, nominal_nav_path: Path) -> tuple[np.ndarray, np.ndarray]:
    delta = np.loadtxt(delta_path, comments="#")
    nominal = np.loadtxt(nominal_nav_path, comments="#")
    time_s = delta[:, 0]
    nominal_time = nominal[:, 0]
    nominal_vel = nominal[:, 4:7]
    vel_nom_interp = np.column_stack([
        np.interp(time_s, nominal_time, nominal_vel[:, axis]) for axis in range(3)
    ])
    delta_vel = delta[:, 4:7]
    return time_s, vel_nom_interp + delta_vel


def load_kf_nav(nav_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(nav_path)
    time_s = data[:, 1]
    # KF_GINS_Navresult.nav columns 6~8 are velocity in north-east-down.
    vel_ned = data[:, 5:8]
    vel_enu = ned_velocity_to_enu(vel_ned)

    attitude_deg = np.column_stack((data[:, 8], data[:, 9], heading_degrees_0_360(data[:, 10])))
    return time_s, vel_enu, attitude_deg


def load_rtk_nav(rtk_path: Path) -> tuple[np.ndarray, np.ndarray]:
    rtk = np.loadtxt(rtk_path, comments="#")
    time_s = rtk[:, 0]
    pos_enu = blh_rad_to_local_enu(rtk, rtk[0, 1:4])
    return time_s, pos_enu


def intersect_window(time_a: np.ndarray, time_b: np.ndarray) -> tuple[float, float]:
    start = max(float(time_a[0]), float(time_b[0]))
    end = min(float(time_a[-1]), float(time_b[-1]))
    if start > end:
        raise RuntimeError(f"No common time window: [{start}, {end}]")
    return start, end


def trim_series(time_s: np.ndarray, values: np.ndarray, start: float, end: float) -> tuple[np.ndarray, np.ndarray]:
    mask = (time_s >= start) & (time_s <= end)
    return time_s[mask], values[mask]


def plot_compare(
    title: str,
    ylabels: list[str],
    output_prefix: Path,
    series: list[tuple[np.ndarray, np.ndarray, str]],
    matlab_fig: bool,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for axis_idx in range(3):
        ax = axes[axis_idx]
        for time_s, values, label in series:
            ax.plot(time_s, values[:, axis_idx], linewidth=1.1, label=label)
        ax.set_ylabel(ylabels[axis_idx])
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(loc="best")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    fig.tight_layout()

    png_path = output_prefix.with_suffix(".png")
    fig_path = output_prefix.with_suffix(".fig")
    fig.savefig(png_path, dpi=180)
    if matlab_fig:
        save_matlab_fig(fig_path, [np.column_stack((time_s, values)) for time_s, values, _ in series], [label for _, _, label in series], title)
    else:
        save_fig_pickle(fig, fig_path)
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {fig_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CT vs KF/RTK navigation comparisons.")
    parser.add_argument("--traj", required=True, type=Path, help="Path to trajectory_enu.txt")
    parser.add_argument("--nominal-nav", required=True, type=Path, help="Path to nominal_nav.txt")
    parser.add_argument("--delta", required=True, type=Path, help="Path to delta_estimates.txt")
    parser.add_argument("--kf-nav", required=True, type=Path, help="Path to KF_GINS_Navresult.nav")
    parser.add_argument("--rtk", required=True, type=Path, help="Path to rtk_ct_fgo_sim.txt")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--matlab-fig", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    ct_pose_time, ct_pos, ct_att = load_ct_pose(args.traj)
    ct_vel_time, ct_vel = load_ct_velocity(args.delta, args.nominal_nav)
    kf_time, kf_vel, kf_att = load_kf_nav(args.kf_nav)
    rtk_time, rtk_pos = load_rtk_nav(args.rtk)

    att_start, att_end = intersect_window(ct_pose_time, kf_time)
    ct_att_time, ct_att = trim_series(ct_pose_time, ct_att, att_start, att_end)
    kf_att_time, kf_att = trim_series(kf_time, kf_att, att_start, att_end)
    plot_compare(
        "Attitude Comparison (shared time window, NED)",
        ["Roll (deg)", "Pitch (deg)", "Yaw (deg)"],
        args.output_dir / "attitude_compare_full",
        [(ct_att_time, ct_att, "CT full"), (kf_att_time, kf_att, "KF-GINS")],
        args.matlab_fig,
    )

    vel_start, vel_end = intersect_window(ct_vel_time, kf_time)
    ct_vel_time_trim, ct_vel_trim = trim_series(ct_vel_time, ct_vel, vel_start, vel_end)
    kf_vel_time, kf_vel_trim = trim_series(kf_time, kf_vel, vel_start, vel_end)
    plot_compare(
        "Velocity Comparison (shared time window)",
        ["East vel (m/s)", "North vel (m/s)", "Up vel (m/s)"],
        args.output_dir / "velocity_compare_kf",
        [(ct_vel_time_trim, ct_vel_trim, "CT full"), (kf_vel_time, kf_vel_trim, "KF-GINS")],
        args.matlab_fig,
    )

    pos_start, pos_end = intersect_window(ct_pose_time, rtk_time)
    ct_pos_time, ct_pos_trim = trim_series(ct_pose_time, ct_pos, pos_start, pos_end)
    rtk_pos_time, rtk_pos_trim = trim_series(rtk_time, rtk_pos, pos_start, pos_end)
    plot_compare(
        "Position Comparison (shared time window)",
        ["East (m)", "North (m)", "Up (m)"],
        args.output_dir / "position_compare_rtk",
        [(ct_pos_time, ct_pos_trim, "CT full"), (rtk_pos_time, rtk_pos_trim, "RTK")],
        args.matlab_fig,
    )


if __name__ == "__main__":
    main()
