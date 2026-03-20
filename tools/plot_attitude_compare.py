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


def load_output_time_origin(output_dir: Path) -> float | None:
    summary_path = output_dir / "run_summary.txt"
    if not summary_path.exists():
        return None
    for line in summary_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("output_time_origin_s:"):
            return float(line.split(":", 1)[1].strip())
    return None


def maybe_to_relative_time(time_s: np.ndarray, output_time_origin_s: float | None) -> np.ndarray:
    if output_time_origin_s is None:
        return time_s
    if np.nanmin(time_s) > output_time_origin_s * 0.5:
        return time_s - output_time_origin_s
    return time_s


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


def load_rtk_velocity_heading(
    rtk_path: Path,
    speed_threshold_mps: float,
) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(rtk_path, comments="#")
    if data.ndim != 2 or data.shape[1] < 4:
        raise RuntimeError(f"Unexpected RTK shape in {rtk_path}")

    time_s = data[:, 0]
    pos_enu = blh_rad_to_local_enu(data, data[0, 1:4])
    vel_enu = np.empty_like(pos_enu)
    vel_enu[1:-1] = (pos_enu[2:] - pos_enu[:-2]) / (time_s[2:, None] - time_s[:-2, None])
    vel_enu[0] = (pos_enu[1] - pos_enu[0]) / (time_s[1] - time_s[0])
    vel_enu[-1] = (pos_enu[-1] - pos_enu[-2]) / (time_s[-1] - time_s[-2])

    speed_h = np.linalg.norm(vel_enu[:, :2], axis=1)
    # NED heading convention: yaw is measured from North toward East.
    yaw_deg = heading_degrees_0_360(np.degrees(np.arctan2(vel_enu[:, 0], vel_enu[:, 1])))
    yaw_deg[speed_h < speed_threshold_mps] = np.nan
    return time_s, yaw_deg


def load_ct_attitude(nav_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(nav_path, comments="#")
    if data.ndim != 2:
        raise RuntimeError(f"Unexpected CT nav shape in {nav_path}")

    time_s = data[:, 0]
    if nav_path.name == "nominal_nav.txt":
        qx, qy, qz, qw = data[:, 7], data[:, 8], data[:, 9], data[:, 10]
    else:
        qx, qy, qz, qw = data[:, 4], data[:, 5], data[:, 6], data[:, 7]

    roll_out = np.empty_like(time_s)
    pitch_out = np.empty_like(time_s)
    yaw_out = np.empty_like(time_s)
    for idx in range(len(time_s)):
        rot_ct_enu_flu = quat_to_rotmat_xyzw(qx[idx], qy[idx], qz[idx], qw[idx])
        # CT output quaternions are already expressed in the vehicle/body convention.
        # For NED plotting we only need to rotate the navigation basis ENU -> NED.
        rot_ct_ned_frd = R_ENU_FROM_NED.T @ rot_ct_enu_flu
        roll_rad, pitch_rad, yaw_rad = rotmat_to_euler(rot_ct_ned_frd)
        roll_out[idx] = np.degrees(roll_rad)
        pitch_out[idx] = np.degrees(pitch_rad)
        yaw_out[idx] = np.degrees(yaw_rad)
    yaw_out = heading_degrees_0_360(yaw_out)
    return time_s, roll_out, pitch_out, yaw_out


def load_kf_attitude(nav_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(nav_path)
    if data.ndim != 2 or data.shape[1] < 11:
        raise RuntimeError(f"Unexpected KF nav shape in {nav_path}")

    time_s = data[:, 1]
    roll_deg = data[:, 8]
    pitch_deg = data[:, 9]
    yaw_deg = heading_degrees_0_360(data[:, 10])
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


def compute_common_time_window(
    ct_time: np.ndarray,
    kf_time: np.ndarray,
    start_time: float | None,
    end_time: float | None,
) -> tuple[float, float]:
    common_start = max(ct_time[0], kf_time[0])
    common_end = min(ct_time[-1], kf_time[-1])
    if start_time is not None:
        common_start = max(common_start, start_time)
    if end_time is not None:
        common_end = min(common_end, end_time)
    if common_start > common_end:
        raise RuntimeError(
            f"No common time window between CT and KF after trimming: "
            f"[{common_start}, {common_end}]"
        )
    return common_start, common_end


def save_fig_pickle(fig: plt.Figure, path: Path) -> None:
    with path.open("wb") as f:
        pickle.dump(fig, f)


def require_nonempty(time_s: np.ndarray, label: str, start_time: float | None, end_time: float | None) -> None:
    if time_s.size == 0:
        raise RuntimeError(
            f"{label} has no samples in time window [{start_time}, {end_time}]. "
            "Check the selected navigation file and requested time range."
        )


def save_matlab_fig(
    output_fig_path: Path,
    ct_time: np.ndarray,
    ct_roll: np.ndarray,
    ct_pitch: np.ndarray,
    ct_yaw: np.ndarray,
    kf_time: np.ndarray,
    kf_roll: np.ndarray,
    kf_pitch: np.ndarray,
    kf_yaw: np.ndarray,
    ct_label: str,
    kf_label: str,
    rtk_time: np.ndarray | None = None,
    rtk_yaw: np.ndarray | None = None,
    rtk_label: str | None = None,
) -> None:
    output_fig_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ct_fgo_matlab_fig_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        ct_csv = tmp_dir / "ct_attitude.csv"
        kf_csv = tmp_dir / "kf_attitude.csv"
        rtk_csv = tmp_dir / "rtk_heading.csv"
        script_path = tmp_dir / "make_attitude_compare_fig.m"
        np.savetxt(ct_csv, np.column_stack((ct_time, ct_roll, ct_pitch, ct_yaw)), delimiter=",")
        np.savetxt(kf_csv, np.column_stack((kf_time, kf_roll, kf_pitch, kf_yaw)), delimiter=",")
        if rtk_time is not None and rtk_yaw is not None:
            np.savetxt(rtk_csv, np.column_stack((rtk_time, rtk_yaw)), delimiter=",")

        script = textwrap.dedent(
            f"""
            ct = readmatrix('{ct_csv.as_posix()}');
            kf = readmatrix('{kf_csv.as_posix()}');
            has_rtk = {1 if rtk_time is not None and rtk_yaw is not None else 0};
            if has_rtk
                rtk = readmatrix('{rtk_csv.as_posix()}');
            end
            fig = figure('Visible', 'off', 'Position', [100, 100, 1100, 780]);
            labels = {{'Roll (deg)', 'Pitch (deg)', 'Yaw (deg)'}};
            for idx = 1:3
                subplot(3,1,idx);
                plot(ct(:,1), ct(:,idx+1), 'LineWidth', 1.2, 'DisplayName', '{ct_label}');
                hold on;
                plot(kf(:,1), kf(:,idx+1), 'LineWidth', 1.0, 'DisplayName', '{kf_label}');
                if has_rtk && idx == 3
                    plot(rtk(:,1), rtk(:,2), 'LineWidth', 1.0, 'DisplayName', '{rtk_label or "RTK heading"}');
                end
                grid on;
                ylabel(labels{{idx}});
                legend('Location', 'best');
            end
            xlabel('Time (s)');
            sgtitle('Attitude Comparison (NED)');
            savefig(fig, '{output_fig_path.as_posix()}');
            close(fig);
            exit;
            """
        ).strip()
        script_path.write_text(script, encoding="utf-8")
        subprocess.run(["matlab", "-batch", f"run('{script_path.as_posix()}')"], check=True)


def plot_and_save(
    output_png_path: Path,
    output_fig_path: Path,
    title: str,
    ct_time: np.ndarray,
    ct_roll: np.ndarray,
    ct_pitch: np.ndarray,
    ct_yaw: np.ndarray,
    kf_time: np.ndarray,
    kf_roll: np.ndarray,
    kf_pitch: np.ndarray,
    kf_yaw: np.ndarray,
    ct_label: str,
    kf_label: str,
    matlab_fig: bool,
    rtk_time: np.ndarray | None = None,
    rtk_yaw: np.ndarray | None = None,
    rtk_label: str = "RTK diff heading",
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    series = [
        ("Roll (deg)", ct_roll, kf_roll),
        ("Pitch (deg)", ct_pitch, kf_pitch),
        ("Yaw (deg)", ct_yaw, kf_yaw),
    ]

    for ax, (ylabel, ct_values, kf_values) in zip(axes, series):
        ax.plot(ct_time, ct_values, linewidth=1.2, label=ct_label)
        ax.plot(kf_time, kf_values, linewidth=1.0, label=kf_label)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(loc="best")

    if rtk_time is not None and rtk_yaw is not None:
        axes[2].plot(rtk_time, rtk_yaw, linewidth=1.0, label=rtk_label)
        axes[2].legend(loc="best")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_png_path, dpi=180)
    if matlab_fig:
        save_matlab_fig(
            output_fig_path,
            ct_time, ct_roll, ct_pitch, ct_yaw,
            kf_time, kf_roll, kf_pitch, kf_yaw,
            ct_label, kf_label, rtk_time, rtk_yaw, rtk_label,
        )
    else:
        save_fig_pickle(fig, output_fig_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CT vs KF attitude comparison.")
    parser.add_argument("--ct-nav", required=True, type=Path, help="Path to trajectory_enu.txt or nominal_nav.txt")
    parser.add_argument("--kf-nav", required=True, type=Path, help="Path to KF_GINS_Navresult.nav")
    parser.add_argument("--rtk", type=Path, default=None, help="Optional RTK file for velocity-differenced heading overlay")
    parser.add_argument("--full-nav", type=Path, default=None, help="Optional path to trajectory_enu.txt for full-state plot")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory; defaults to CT nav parent")
    parser.add_argument("--start-time", type=float, default=None, help="Optional start time in seconds")
    parser.add_argument("--end-time", type=float, default=None, help="Optional end time in seconds")
    parser.add_argument("--ct-label", default="CT", help="Legend label for CT line")
    parser.add_argument("--kf-label", default="KF-GINS", help="Legend label for KF line")
    parser.add_argument("--rtk-label", default="RTK diff heading", help="Legend label for RTK-derived heading")
    parser.add_argument("--rtk-speed-threshold", type=float, default=0.5, help="Minimum horizontal speed for RTK heading")
    parser.add_argument("--matlab-fig", action="store_true", help="Also export a MATLAB-readable .fig via matlab -batch")
    args = parser.parse_args()
    output_dir = args.output_dir or args.ct_nav.parent
    output_time_origin_s = load_output_time_origin(output_dir)

    ct_time, ct_roll, ct_pitch, ct_yaw = load_ct_attitude(args.ct_nav)
    kf_time, kf_roll, kf_pitch, kf_yaw = load_kf_attitude(args.kf_nav)
    kf_time = maybe_to_relative_time(kf_time, output_time_origin_s)
    rtk_heading: tuple[np.ndarray, np.ndarray] | None = None
    if args.rtk is not None:
        rtk_time, rtk_yaw = load_rtk_velocity_heading(args.rtk, args.rtk_speed_threshold)
        rtk_heading = (maybe_to_relative_time(rtk_time, output_time_origin_s), rtk_yaw)

    common_start, common_end = compute_common_time_window(
        ct_time, kf_time, args.start_time, args.end_time)

    ct_time, ct_roll, ct_pitch, ct_yaw = maybe_trim(
        ct_time, ct_roll, ct_pitch, ct_yaw, common_start, common_end)
    kf_time, kf_roll, kf_pitch, kf_yaw = maybe_trim(
        kf_time, kf_roll, kf_pitch, kf_yaw, common_start, common_end)
    rtk_time = None
    rtk_yaw = None
    if rtk_heading is not None:
        rtk_time, _, _, rtk_yaw = maybe_trim(
            rtk_heading[0], np.zeros_like(rtk_heading[1]), np.zeros_like(rtk_heading[1]), rtk_heading[1],
            common_start, common_end)
    require_nonempty(ct_time, "CT attitude", common_start, common_end)
    require_nonempty(kf_time, "KF attitude", common_start, common_end)

    output_dir.mkdir(parents=True, exist_ok=True)

    nominal_png_path = output_dir / "attitude_compare.png"
    nominal_fig_path = output_dir / "attitude_compare.fig"
    plot_and_save(
        nominal_png_path,
        nominal_fig_path,
        "Attitude Comparison (Nominal vs KF, NED)",
        ct_time, ct_roll, ct_pitch, ct_yaw,
        kf_time, kf_roll, kf_pitch, kf_yaw,
        args.ct_label, args.kf_label, args.matlab_fig, rtk_time, rtk_yaw, args.rtk_label,
    )
    print(f"Wrote {nominal_png_path}")
    print(f"Wrote {nominal_fig_path}")

    full_nav_path = args.full_nav
    if full_nav_path is None:
        sibling_full = args.ct_nav.parent / "trajectory_enu.txt"
        if sibling_full != args.ct_nav and sibling_full.exists():
            full_nav_path = sibling_full

    if full_nav_path is not None:
        full_time, full_roll, full_pitch, full_yaw = load_ct_attitude(full_nav_path)
        full_common_start, full_common_end = compute_common_time_window(
            full_time, kf_time, args.start_time, args.end_time)
        full_time, full_roll, full_pitch, full_yaw = maybe_trim(
            full_time, full_roll, full_pitch, full_yaw, full_common_start, full_common_end)
        full_kf_time, full_kf_roll, full_kf_pitch, full_kf_yaw = maybe_trim(
            kf_time, kf_roll, kf_pitch, kf_yaw, full_common_start, full_common_end)
        full_rtk_time = None
        full_rtk_yaw = None
        if rtk_heading is not None:
            full_rtk_time, _, _, full_rtk_yaw = maybe_trim(
                rtk_heading[0], np.zeros_like(rtk_heading[1]), np.zeros_like(rtk_heading[1]), rtk_heading[1],
                full_common_start, full_common_end)
        require_nonempty(full_time, "CT full attitude", full_common_start, full_common_end)
        require_nonempty(full_kf_time, "KF attitude", full_common_start, full_common_end)

        full_png_path = output_dir / "attitude_compare_full.png"
        full_fig_path = output_dir / "attitude_compare_full.fig"
        plot_and_save(
            full_png_path,
            full_fig_path,
            "Attitude Comparison (Full vs KF, NED)",
            full_time, full_roll, full_pitch, full_yaw,
            full_kf_time, full_kf_roll, full_kf_pitch, full_kf_yaw,
            "CT full", args.kf_label, args.matlab_fig, full_rtk_time, full_rtk_yaw, args.rtk_label,
        )
        print(f"Wrote {full_png_path}")
        print(f"Wrote {full_fig_path}")


if __name__ == "__main__":
    main()
