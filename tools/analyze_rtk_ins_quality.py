from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CHI2_3D_99 = 11.344866730144373
CHI2_3D_999 = 16.26623619623813


def wrap_degrees(angle_deg: np.ndarray) -> np.ndarray:
    return (angle_deg + 180.0) % 360.0 - 180.0


def blh_rad_to_local_enu(blh_rad: np.ndarray, origin_blh_rad: np.ndarray) -> np.ndarray:
    lat0, lon0, h0 = origin_blh_rad
    lat = blh_rad[:, 0]
    lon = blh_rad[:, 1]
    h = blh_rad[:, 2]

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
        ],
        dtype=float,
    )
    return np.column_stack([dx, dy, dz]) @ ecef_to_enu.T


def load_nominal_nav(path: Path, origin_blh: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] < 7:
        raise RuntimeError(f"Unexpected nominal_nav format: {path}")
    time_s = data[:, 0]
    blh = data[:, 1:4]
    pos_enu = blh_rad_to_local_enu(blh, origin_blh)
    vel_enu = data[:, 4:7]
    return time_s, blh, pos_enu, vel_enu


def load_rtk(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] < 4:
        raise RuntimeError(f"Unexpected RTK format: {path}")
    time_s = data[:, 0]
    blh = data[:, 1:4]
    pos_enu = blh_rad_to_local_enu(blh, blh[0])
    std_enu = np.full_like(pos_enu, np.nan)
    if data.shape[1] >= 7:
        std_enu = data[:, 4:7]
    return time_s, blh, pos_enu, std_enu


def interpolate_rows(sample_time: np.ndarray, source_time: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [np.interp(sample_time, source_time, values[:, axis]) for axis in range(values.shape[1])]
    )


def central_difference(time_s: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.full_like(values, np.nan)
    if time_s.size < 2:
        return out
    out[0] = (values[1] - values[0]) / (time_s[1] - time_s[0])
    out[-1] = (values[-1] - values[-2]) / (time_s[-1] - time_s[-2])
    dt = time_s[2:] - time_s[:-2]
    good = dt > 0.0
    middle = np.full_like(values[1:-1], np.nan)
    middle[good] = (values[2:][good] - values[:-2][good]) / dt[good, None]
    out[1:-1] = middle
    return out


def robust_sigma(values: np.ndarray, min_sigma: float) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return min_sigma
    med = np.median(finite)
    mad = np.median(np.abs(finite - med))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma < min_sigma:
        return min_sigma
    return float(sigma)


def rolling_window_indices(time_s: np.ndarray, center_idx: int, window_s: float) -> np.ndarray:
    half = 0.5 * window_s
    return np.flatnonzero((time_s >= time_s[center_idx] - half) & (time_s <= time_s[center_idx] + half))


def rolling_cov_nis(
    time_s: np.ndarray,
    residual: np.ndarray,
    window_s: float,
    rtk_std: np.ndarray,
    min_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    nis = np.full(time_s.shape, np.nan)
    cov_diag = np.full((time_s.size, 3), np.nan)
    eye = np.eye(3)
    for idx in range(time_s.size):
        win = rolling_window_indices(time_s, idx, window_s)
        if win.size < 6:
            continue
        sample = residual[win]
        sample = sample[np.all(np.isfinite(sample), axis=1)]
        if sample.shape[0] < 6:
            continue
        cov = np.cov(sample.T)
        meas_var = np.where(np.isfinite(rtk_std[idx]), rtk_std[idx], min_std) ** 2
        cov = cov + np.diag(np.maximum(meas_var, min_std ** 2))
        cov = cov + eye * 1.0e-9
        cov_diag[idx] = np.diag(cov)
        centered = residual[idx] - np.mean(sample, axis=0)
        try:
            nis[idx] = float(centered.T @ np.linalg.solve(cov, centered))
        except np.linalg.LinAlgError:
            nis[idx] = np.nan
    return nis, cov_diag


def rolling_slope(time_s: np.ndarray, values: np.ndarray, window_s: float) -> np.ndarray:
    slopes = np.full(values.shape, np.nan)
    for idx in range(time_s.size):
        win = rolling_window_indices(time_s, idx, window_s)
        if win.size < 4:
            continue
        t = time_s[win] - np.mean(time_s[win])
        denom = float(np.dot(t, t))
        if denom <= 0.0:
            continue
        y = values[win] - np.mean(values[win], axis=0)
        slopes[idx] = t @ y / denom
    return slopes


def build_flags(
    res_h: np.ndarray,
    res_u: np.ndarray,
    delta_res_h: np.ndarray,
    drift_h: np.ndarray,
    vel_res_h: np.ndarray,
    heading_res_deg: np.ndarray,
    nis: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    jump_sigma = robust_sigma(delta_res_h, args.min_jump_sigma_m)
    drift_sigma = robust_sigma(drift_h, args.min_drift_sigma_mps)
    flags = {
        "jump": delta_res_h > max(args.jump_threshold_m, args.jump_sigma_factor * jump_sigma),
        "drift": drift_h > max(args.drift_threshold_mps, args.drift_sigma_factor * drift_sigma),
        "position": (res_h > args.position_threshold_m) | (np.abs(res_u) > args.vertical_threshold_m),
        "velocity": vel_res_h > args.velocity_threshold_mps,
        "heading": np.abs(heading_res_deg) > args.heading_threshold_deg,
        "nis99": nis > CHI2_3D_99,
        "nis999": nis > CHI2_3D_999,
    }
    flags["any"] = np.zeros(res_h.shape, dtype=bool)
    for value in flags.values():
        flags["any"] |= np.nan_to_num(value, nan=False).astype(bool)
    return flags


def save_diagnostics(
    output_path: Path,
    time_s: np.ndarray,
    residual_pos: np.ndarray,
    res_h: np.ndarray,
    delta_res_h: np.ndarray,
    drift: np.ndarray,
    drift_h: np.ndarray,
    residual_vel: np.ndarray,
    vel_res_h: np.ndarray,
    heading_res_deg: np.ndarray,
    nis: np.ndarray,
    cov_diag: np.ndarray,
    flags: dict[str, np.ndarray],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "time_s res_e_m res_n_m res_u_m res_h_m delta_res_h_m "
        "drift_e_mps drift_n_mps drift_u_mps drift_h_mps "
        "vel_res_e_mps vel_res_n_mps vel_res_u_mps vel_res_h_mps "
        "heading_res_deg nis_pos cov_e_m2 cov_n_m2 cov_u_m2 "
        "flag_jump flag_drift flag_position flag_velocity flag_heading flag_nis99 flag_nis999 flag_any"
    )
    data = np.column_stack(
        (
            time_s,
            residual_pos,
            res_h,
            delta_res_h,
            drift,
            drift_h,
            residual_vel,
            vel_res_h,
            heading_res_deg,
            nis,
            cov_diag,
            flags["jump"].astype(int),
            flags["drift"].astype(int),
            flags["position"].astype(int),
            flags["velocity"].astype(int),
            flags["heading"].astype(int),
            flags["nis99"].astype(int),
            flags["nis999"].astype(int),
            flags["any"].astype(int),
        )
    )
    np.savetxt(output_path, data, fmt="%.9g", header=header)


def save_plots(
    output_path: Path,
    time_s: np.ndarray,
    residual_pos: np.ndarray,
    res_h: np.ndarray,
    delta_res_h: np.ndarray,
    drift_h: np.ndarray,
    vel_res_h: np.ndarray,
    heading_res_deg: np.ndarray,
    nis: np.ndarray,
    flags: dict[str, np.ndarray],
) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(13, 11), sharex=True)

    axes[0].plot(time_s, residual_pos[:, 0], label="E")
    axes[0].plot(time_s, residual_pos[:, 1], label="N")
    axes[0].plot(time_s, residual_pos[:, 2], label="U")
    axes[0].plot(time_s, res_h, label="horizontal", color="black", linewidth=1.0)
    axes[0].set_ylabel("pos residual (m)")
    axes[0].legend(loc="best")

    axes[1].plot(time_s, delta_res_h, label="horizontal delta residual")
    axes[1].set_ylabel("jump score (m)")
    axes[1].legend(loc="best")

    axes[2].plot(time_s, drift_h, label="horizontal drift rate")
    axes[2].set_ylabel("drift (m/s)")
    axes[2].legend(loc="best")

    axes[3].plot(time_s, vel_res_h, label="velocity residual")
    axes[3].plot(time_s, np.abs(heading_res_deg), label="abs heading residual deg")
    axes[3].set_ylabel("vel / heading")
    axes[3].legend(loc="best")

    axes[4].plot(time_s, nis, label="position NIS")
    axes[4].axhline(CHI2_3D_99, color="tab:orange", linestyle="--", label="chi2 99%")
    axes[4].axhline(CHI2_3D_999, color="tab:red", linestyle="--", label="chi2 99.9%")
    axes[4].set_ylabel("NIS")
    axes[4].set_xlabel("time (s)")
    axes[4].legend(loc="best")

    flag_idx = np.flatnonzero(flags["any"])
    for ax in axes:
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        if flag_idx.size:
            ymin, ymax = ax.get_ylim()
            ax.scatter(time_s[flag_idx], np.full(flag_idx.shape, ymax), s=9, color="red", alpha=0.5)
            ax.set_ylim(ymin, ymax)

    fig.suptitle("RTK vs INS/nominal consistency diagnostics")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_position_error_blh_plot(
    output_path: Path,
    time_s: np.ndarray,
    residual_pos: np.ndarray,
    flags: dict[str, np.ndarray],
) -> None:
    # In the local ENU frame, latitude error maps to North, longitude error maps to East.
    lat_lon_hgt_error_m = np.column_stack((residual_pos[:, 1], residual_pos[:, 0], residual_pos[:, 2]))
    labels = ["Latitude / North error (m)", "Longitude / East error (m)", "Height / Up error (m)"]
    flag_idx = np.flatnonzero(flags["any"])

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    for axis_idx, ax in enumerate(axes):
        values = lat_lon_hgt_error_m[:, axis_idx]
        ax.plot(time_s, values, linewidth=1.0, label=labels[axis_idx])
        if flag_idx.size:
            ax.scatter(
                time_s[flag_idx],
                values[flag_idx],
                s=18,
                color="red",
                alpha=0.75,
                label="flagged RTK sample",
                zorder=3,
            )
        ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
        ax.set_ylabel(labels[axis_idx])
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(loc="best")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("RTK - INS position error with flagged samples")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_blh_compare_plot(
    output_png_path: Path,
    output_fig_path: Path,
    time_s: np.ndarray,
    rtk_blh: np.ndarray,
    nav_blh: np.ndarray,
) -> None:
    rtk_plot = np.column_stack((np.degrees(rtk_blh[:, 0]), np.degrees(rtk_blh[:, 1]), rtk_blh[:, 2]))
    nav_plot = np.column_stack((np.degrees(nav_blh[:, 0]), np.degrees(nav_blh[:, 1]), nav_blh[:, 2]))
    labels = ["Latitude (deg)", "Longitude (deg)", "Height (m)"]

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    for axis_idx, ax in enumerate(axes):
        ax.plot(time_s, rtk_plot[:, axis_idx], linewidth=1.0, label="RTK")
        ax.plot(time_s, nav_plot[:, axis_idx], linewidth=1.0, label="Nav nominal")
        ax.set_ylabel(labels[axis_idx])
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(loc="best")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("RTK and navigation BLH comparison")
    fig.tight_layout()
    fig.savefig(output_png_path, dpi=180)
    plt.close(fig)

    try:
        save_blh_compare_matlab_fig(output_fig_path, time_s, rtk_plot, nav_plot, labels)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"WARNING: failed to export MATLAB fig {output_fig_path}: {exc}")


def save_blh_compare_matlab_fig(
    output_fig_path: Path,
    time_s: np.ndarray,
    rtk_plot: np.ndarray,
    nav_plot: np.ndarray,
    labels: list[str],
) -> None:
    output_fig_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ct_fgo_blh_fig_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        rtk_csv = tmp_dir / "rtk_blh.csv"
        nav_csv = tmp_dir / "nav_blh.csv"
        np.savetxt(rtk_csv, np.column_stack((time_s, rtk_plot)), delimiter=",")
        np.savetxt(nav_csv, np.column_stack((time_s, nav_plot)), delimiter=",")

        script_path = tmp_dir / "make_blh_fig.m"
        lines = [
            "rtk = readmatrix('{rtk_csv}');".format(rtk_csv=rtk_csv.as_posix()),
            "nav = readmatrix('{nav_csv}');".format(nav_csv=nav_csv.as_posix()),
            "fig = figure('Visible', 'off', 'Position', [100, 100, 1200, 800]);",
            "tiledlayout(3,1);",
        ]
        for axis_idx, label in enumerate(labels, start=1):
            lines.extend(
                [
                    f"nexttile({axis_idx});",
                    f"plot(rtk(:,1), rtk(:,{axis_idx + 1}), 'LineWidth', 1.1, 'DisplayName', 'RTK'); hold on;",
                    f"plot(nav(:,1), nav(:,{axis_idx + 1}), 'LineWidth', 1.1, 'DisplayName', 'Nav nominal');",
                    "grid on; legend('Location','best');",
                    f"ylabel('{label}');",
                ]
            )
        lines.extend(
            [
                "xlabel('time (s)');",
                "sgtitle('RTK and navigation BLH comparison');",
                "savefig(fig, '{fig_path}');".format(fig_path=output_fig_path.as_posix()),
                "close(fig); exit;",
            ]
        )
        script_path.write_text("\n".join(lines), encoding="utf-8")
        subprocess.run(["matlab", "-batch", f"run('{script_path.as_posix()}')"], check=True)


def save_summary(
    output_path: Path,
    time_s: np.ndarray,
    res_h: np.ndarray,
    residual_pos: np.ndarray,
    delta_res_h: np.ndarray,
    drift_h: np.ndarray,
    vel_res_h: np.ndarray,
    heading_res_deg: np.ndarray,
    nis: np.ndarray,
    flags: dict[str, np.ndarray],
) -> None:
    def stats(values: np.ndarray) -> dict[str, float]:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return {"mean": float("nan"), "rms": float("nan"), "p95": float("nan"), "max": float("nan")}
        return {
            "mean": float(np.mean(finite)),
            "rms": float(np.sqrt(np.mean(finite ** 2))),
            "p95": float(np.percentile(np.abs(finite), 95.0)),
            "max": float(np.max(np.abs(finite))),
        }

    summary = {
        "time_start_s": float(time_s[0]),
        "time_end_s": float(time_s[-1]),
        "sample_count": int(time_s.size),
        "residual_e_m": stats(residual_pos[:, 0]),
        "residual_n_m": stats(residual_pos[:, 1]),
        "residual_u_m": stats(residual_pos[:, 2]),
        "residual_horizontal_m": stats(res_h),
        "delta_residual_horizontal_m": stats(delta_res_h),
        "drift_horizontal_mps": stats(drift_h),
        "velocity_residual_horizontal_mps": stats(vel_res_h),
        "heading_residual_deg": stats(heading_res_deg),
        "nis_position": stats(nis),
        "flag_counts": {name: int(np.count_nonzero(value)) for name, value in flags.items()},
        "flag_ratio": {name: float(np.count_nonzero(value) / time_s.size) for name, value in flags.items()},
    }
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def save_flagged_samples(
    output_path: Path,
    time_s: np.ndarray,
    res_h: np.ndarray,
    residual_pos: np.ndarray,
    delta_res_h: np.ndarray,
    drift_h: np.ndarray,
    vel_res_h: np.ndarray,
    heading_res_deg: np.ndarray,
    nis: np.ndarray,
    flags: dict[str, np.ndarray],
) -> None:
    rows = []
    for idx in np.flatnonzero(flags["any"]):
        reasons = [
            name
            for name in ("jump", "drift", "position", "velocity", "heading", "nis99", "nis999")
            if bool(flags[name][idx])
        ]
        rows.append(
            (
                time_s[idx],
                res_h[idx],
                residual_pos[idx, 2],
                delta_res_h[idx],
                drift_h[idx],
                vel_res_h[idx],
                heading_res_deg[idx],
                nis[idx],
                "|".join(reasons),
            )
        )

    with output_path.open("w", encoding="utf-8") as f:
        f.write(
            "time_s res_h_m res_u_m delta_res_h_m drift_h_mps "
            "vel_res_h_mps heading_res_deg nis_pos reasons\n"
        )
        for row in rows:
            f.write(
                f"{row[0]:.6f} {row[1]:.9g} {row[2]:.9g} {row[3]:.9g} "
                f"{row[4]:.9g} {row[5]:.9g} {row[6]:.9g} {row[7]:.9g} {row[8]}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze RTK consistency against CT nominal INS output.")
    parser.add_argument("--nominal-nav", required=True, type=Path, help="Path to nominal_nav.txt")
    parser.add_argument("--rtk", required=True, type=Path, help="Path to rtk_ct_fgo_sim.txt")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--window-s", type=float, default=20.0, help="Rolling statistics window length.")
    parser.add_argument("--speed-threshold-mps", type=float, default=1.0)
    parser.add_argument("--position-threshold-m", type=float, default=1.0)
    parser.add_argument("--vertical-threshold-m", type=float, default=1.5)
    parser.add_argument("--jump-threshold-m", type=float, default=0.4)
    parser.add_argument("--jump-sigma-factor", type=float, default=6.0)
    parser.add_argument("--min-jump-sigma-m", type=float, default=0.03)
    parser.add_argument("--drift-threshold-mps", type=float, default=0.02)
    parser.add_argument("--drift-sigma-factor", type=float, default=5.0)
    parser.add_argument("--min-drift-sigma-mps", type=float, default=0.002)
    parser.add_argument("--velocity-threshold-mps", type=float, default=0.3)
    parser.add_argument("--heading-threshold-deg", type=float, default=10.0)
    parser.add_argument("--min-std-e-m", type=float, default=0.03)
    parser.add_argument("--min-std-n-m", type=float, default=0.03)
    parser.add_argument("--min-std-u-m", type=float, default=0.08)
    args = parser.parse_args()

    rtk_time, rtk_blh, rtk_pos, rtk_std = load_rtk(args.rtk)
    nav_time, nav_blh, nav_pos, nav_vel = load_nominal_nav(args.nominal_nav, rtk_blh[0])

    start = max(float(rtk_time[0]), float(nav_time[0]))
    end = min(float(rtk_time[-1]), float(nav_time[-1]))
    if start >= end:
        raise RuntimeError(f"No common time span: [{start}, {end}]")
    mask = (rtk_time >= start) & (rtk_time <= end)
    time_s = rtk_time[mask]
    rtk_blh = rtk_blh[mask]
    rtk_pos = rtk_pos[mask]
    rtk_std = rtk_std[mask]

    nav_blh_interp = interpolate_rows(time_s, nav_time, nav_blh)
    nav_pos_interp = interpolate_rows(time_s, nav_time, nav_pos)
    nav_vel_interp = interpolate_rows(time_s, nav_time, nav_vel)
    rtk_vel = central_difference(time_s, rtk_pos)

    residual_pos = rtk_pos - nav_pos_interp
    residual_vel = rtk_vel - nav_vel_interp
    res_h = np.linalg.norm(residual_pos[:, :2], axis=1)
    res_u = residual_pos[:, 2]
    delta_res = np.vstack((np.full((1, 3), np.nan), np.diff(residual_pos, axis=0)))
    delta_res_h = np.linalg.norm(delta_res[:, :2], axis=1)
    slopes = rolling_slope(time_s, residual_pos, args.window_s)
    drift_h = np.linalg.norm(slopes[:, :2], axis=1)
    vel_res_h = np.linalg.norm(residual_vel[:, :2], axis=1)

    rtk_speed_h = np.linalg.norm(rtk_vel[:, :2], axis=1)
    nav_speed_h = np.linalg.norm(nav_vel_interp[:, :2], axis=1)
    rtk_heading = np.degrees(np.arctan2(rtk_vel[:, 0], rtk_vel[:, 1]))
    nav_heading = np.degrees(np.arctan2(nav_vel_interp[:, 0], nav_vel_interp[:, 1]))
    heading_res_deg = wrap_degrees(rtk_heading - nav_heading)
    heading_res_deg[(rtk_speed_h < args.speed_threshold_mps) | (nav_speed_h < args.speed_threshold_mps)] = np.nan

    min_std = np.array([args.min_std_e_m, args.min_std_n_m, args.min_std_u_m], dtype=float)
    nis, cov_diag = rolling_cov_nis(time_s, residual_pos, args.window_s, rtk_std, min_std)
    flags = build_flags(res_h, res_u, delta_res_h, drift_h, vel_res_h, heading_res_deg, nis, args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_diagnostics(
        args.output_dir / "rtk_ins_residual_diagnostics.txt",
        time_s,
        residual_pos,
        res_h,
        delta_res_h,
        slopes,
        drift_h,
        residual_vel,
        vel_res_h,
        heading_res_deg,
        nis,
        cov_diag,
        flags,
    )
    save_plots(
        args.output_dir / "rtk_ins_residual_diagnostics.png",
        time_s,
        residual_pos,
        res_h,
        delta_res_h,
        drift_h,
        vel_res_h,
        heading_res_deg,
        nis,
        flags,
    )
    save_position_error_blh_plot(
        args.output_dir / "rtk_position_error_blh_flags.png",
        time_s,
        residual_pos,
        flags,
    )
    save_blh_compare_plot(
        args.output_dir / "rtk_nav_blh_compare.png",
        args.output_dir / "rtk_nav_blh_compare.fig",
        time_s,
        rtk_blh,
        nav_blh_interp,
    )
    save_summary(
        args.output_dir / "rtk_ins_residual_summary.json",
        time_s,
        res_h,
        residual_pos,
        delta_res_h,
        drift_h,
        vel_res_h,
        heading_res_deg,
        nis,
        flags,
    )
    save_flagged_samples(
        args.output_dir / "rtk_quality_flags.txt",
        time_s,
        res_h,
        residual_pos,
        delta_res_h,
        drift_h,
        vel_res_h,
        heading_res_deg,
        nis,
        flags,
    )
    print(f"Wrote {args.output_dir / 'rtk_ins_residual_diagnostics.txt'}")
    print(f"Wrote {args.output_dir / 'rtk_quality_flags.txt'}")
    print(f"Wrote {args.output_dir / 'rtk_ins_residual_diagnostics.png'}")
    print(f"Wrote {args.output_dir / 'rtk_position_error_blh_flags.png'}")
    print(f"Wrote {args.output_dir / 'rtk_nav_blh_compare.png'}")
    print(f"Wrote {args.output_dir / 'rtk_nav_blh_compare.fig'}")
    print(f"Wrote {args.output_dir / 'rtk_ins_residual_summary.json'}")


if __name__ == "__main__":
    main()
