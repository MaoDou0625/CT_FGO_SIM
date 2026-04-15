#!/usr/bin/env python3
"""
Plot CT-FGO navigation trajectory vs RTK (lat/lon/h).

Reuses WGS84 constants (and optionally helpers) from chapter5_module2:
  run_chapter5_module2_kf_gins_pipeline.py

Trajectory file format matches CT-FGO output: trajectory_enu.txt
  # time_s east_m north_m up_m qx qy qz qw
RTK/GNSS file: time_s lat_rad lon_rad h_m [...]

ENU positions are converted to geodetic using the same WGS84 chain as
ct_fgo_sim Earth::LocalToGlobal(origin_blh, ned_from_enu).
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_chapter5_kf_pipeline(chapter5_root: Path):
    """Load chapter5 KF-GINS pipeline module for shared WGS84 constants."""
    path = chapter5_root / "run_chapter5_module2_kf_gins_pipeline.py"
    if not path.is_file():
        raise FileNotFoundError(f"chapter5 script not found: {path}")
    name = "ch5_kf_gins_pipeline"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    # Required for dataclasses (e.g. Python 3.14) when loading by file path.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def enu_to_ned(enu: np.ndarray) -> np.ndarray:
    """Match ct_fgo_sim Earth::EnuToNed."""
    x, y, z = enu[..., 0], enu[..., 1], enu[..., 2]
    return np.stack((y, x, -z), axis=-1)


def blh_to_ecef(blh: np.ndarray, wgs84_ra: float, wgs84_e1: float) -> np.ndarray:
    """Vectorized BlhToEcef; blh (..., 3) lat, lon rad, h m."""
    lat = blh[..., 0]
    lon = blh[..., 1]
    h = blh[..., 2]
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)
    rn = wgs84_ra / np.sqrt(1.0 - wgs84_e1 * sin_lat * sin_lat)
    rnh = rn + h
    x = rnh * cos_lat * cos_lon
    y = rnh * cos_lat * sin_lon
    z = (rnh - rn * wgs84_e1) * sin_lat
    return np.stack((x, y, z), axis=-1)


def ecef_to_blh(ecef: np.ndarray, wgs84_ra: float, wgs84_e1: float) -> np.ndarray:
    """Single-point EcefToBlh (loop matches C++ Earth::EcefToBlh)."""
    x, y, z = [float(v) for v in ecef.flatten()[:3]]
    p = math.sqrt(x * x + y * y)
    lat = math.atan(z / (p * (1.0 - wgs84_e1)))
    lon = 2.0 * math.atan2(y, x + p)
    h = 0.0
    h_prev = 0.0
    while True:
        h_prev = h
        sin_lat = math.sin(lat)
        rn = wgs84_ra / math.sqrt(1.0 - wgs84_e1 * sin_lat * sin_lat)
        h = p / math.cos(lat) - rn
        lat = math.atan(z / (p * (1.0 - wgs84_e1 * rn / (rn + h))))
        if abs(h - h_prev) <= 1.0e-4:
            break
    return np.array([lat, lon, h], dtype=float)


def cne(origin_blh: np.ndarray) -> np.ndarray:
    """Earth::Cne at origin (3x3)."""
    lat = float(origin_blh[0])
    lon = float(origin_blh[1])
    sin_lat = math.sin(lat)
    sin_lon = math.sin(lon)
    cos_lat = math.cos(lat)
    cos_lon = math.cos(lon)
    return np.array(
        [
            [-sin_lat * cos_lon, -sin_lon, -cos_lat * cos_lon],
            [-sin_lat * sin_lon, cos_lon, -cos_lat * sin_lon],
            [cos_lat, 0.0, -sin_lat],
        ],
        dtype=float,
    )


def local_ned_to_blh(origin_blh: np.ndarray, ned: np.ndarray, wgs84_ra: float, wgs84_e1: float) -> np.ndarray:
    """Earth::LocalToGlobal(origin_blh, local_ned)."""
    ecef0 = blh_to_ecef(origin_blh, wgs84_ra, wgs84_e1)
    d_ecef = cne(origin_blh) @ ned.reshape(3, 1)
    ecef1 = ecef0.reshape(3) + d_ecef.reshape(3)
    return ecef_to_blh(ecef1, wgs84_ra, wgs84_e1)


def local_ned_to_blh_vectorized(origin_blh: np.ndarray, ned: np.ndarray, wgs84_ra: float, wgs84_e1: float) -> np.ndarray:
    """ned (N,3) -> blh (N,3)."""
    ecef0 = blh_to_ecef(origin_blh, wgs84_ra, wgs84_e1)
    r = cne(origin_blh)
    out = np.zeros_like(ned)
    for i in range(ned.shape[0]):
        ecef1 = ecef0 + r @ ned[i]
        out[i] = ecef_to_blh(ecef1, wgs84_ra, wgs84_e1)
    return out


def read_origin_blh_rad(summary_path: Path) -> np.ndarray:
    text = summary_path.read_text(encoding="utf-8")
    m = re.search(r"^\s*origin_blh_rad:\s*([0-9eE\.\+\-]+)\s+([0-9eE\.\+\-]+)\s+([0-9eE\.\+\-]+)\s*$", text, re.MULTILINE)
    if not m:
        raise RuntimeError(f"origin_blh_rad not found in {summary_path}")
    return np.array([float(m.group(1)), float(m.group(2)), float(m.group(3))], dtype=float)


def read_gnss_path_from_summary(summary_path: Path) -> Path:
    text = summary_path.read_text(encoding="utf-8")
    m = re.search(r"^\s*gnss_file:\s*(.+)\s*$", text, re.MULTILINE)
    if not m:
        raise RuntimeError(f"gnss_file not found in {summary_path}")
    return Path(m.group(1).strip())


def load_numeric_table(path: Path) -> np.ndarray:
    return np.loadtxt(path, comments="#")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot CT-FGO trajectory vs RTK lat/lon/h.")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"D:\Code\dataset\output\YuHangTuiChe\20260122_121901_use__transformed1cut1_zAxisPro"),
        help="Run output folder containing trajectory_enu.txt and run_summary.txt",
    )
    p.add_argument(
        "--chapter5-root",
        type=Path,
        default=Path(r"D:\googleYun\30Code\chapter5_module2"),
        help="chapter5_module2 root (for WGS84 constants from run_chapter5_module2_kf_gins_pipeline.py)",
    )
    p.add_argument("--rtk-file", type=Path, default=None, help="Override RTK path (default: gnss_file from run_summary.txt)")
    p.add_argument(
        "--trajectory",
        type=Path,
        default=None,
        help="Override trajectory_enu.txt (default: <output-dir>/trajectory_enu.txt)",
    )
    p.add_argument("--save", type=Path, default=None, help="PNG path (default: <output-dir>/nav_vs_rtk_blh.png)")
    p.add_argument("--show", action="store_true", help="Show matplotlib window")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    traj_path = args.trajectory or (out_dir / "trajectory_enu.txt")
    summary_path = out_dir / "run_summary.txt"
    if not traj_path.is_file():
        raise FileNotFoundError(traj_path)
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)

    ch5 = load_chapter5_kf_pipeline(args.chapter5_root)
    wgs84_ra = float(ch5.WGS84_RA)
    wgs84_e1 = float(ch5.WGS84_E1)
    rad2deg = float(ch5.RAD2DEG)

    origin_blh = read_origin_blh_rad(summary_path)
    rtk_path = args.rtk_file or read_gnss_path_from_summary(summary_path)
    if not rtk_path.is_file():
        raise FileNotFoundError(rtk_path)

    traj = load_numeric_table(traj_path)
    if traj.shape[1] < 4:
        raise RuntimeError(f"Unexpected trajectory columns in {traj_path}")
    t_nav = traj[:, 0]
    enu = traj[:, 1:4]
    ned = enu_to_ned(enu)
    nav_blh = local_ned_to_blh_vectorized(origin_blh, ned, wgs84_ra, wgs84_e1)

    rtk = load_numeric_table(rtk_path)
    if rtk.shape[1] < 4:
        raise RuntimeError(f"Unexpected RTK columns in {rtk_path}")
    t_rtk = rtk[:, 0]
    rtk_lat = np.interp(t_nav, t_rtk, rtk[:, 1], left=np.nan, right=np.nan)
    rtk_lon = np.interp(t_nav, t_rtk, rtk[:, 2], left=np.nan, right=np.nan)
    rtk_h = np.interp(t_nav, t_rtk, rtk[:, 3], left=np.nan, right=np.nan)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    titles = ["Latitude (deg)", "Longitude (deg)", "Ellipsoidal height h (m)"]
    nav_deg_lat = nav_blh[:, 0] * rad2deg
    nav_deg_lon = nav_blh[:, 1] * rad2deg
    rtk_deg_lat = rtk_lat * rad2deg
    rtk_deg_lon = rtk_lon * rad2deg
    series = [
        (nav_deg_lat, rtk_deg_lat),
        (nav_deg_lon, rtk_deg_lon),
        (nav_blh[:, 2], rtk_h),
    ]
    for ax, (nav_y, rtk_y), title in zip(axes, series, titles):
        ax.plot(t_nav, rtk_y, label="RTK (interp)", color="C0", linewidth=1.0, alpha=0.85)
        ax.plot(t_nav, nav_y, label="CT-FGO nav", color="C1", linewidth=1.0, alpha=0.9)
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Navigation vs RTK — {out_dir.name}", fontsize=11)

    save_path = args.save or (out_dir / "nav_vs_rtk_blh.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"Wrote {save_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
