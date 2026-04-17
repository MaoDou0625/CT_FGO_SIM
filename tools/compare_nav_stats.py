#!/usr/bin/env python3
"""
Compare two CT-FGO simulation outputs and print navigation error stats.

Metrics:
  - Position (from `trajectory_enu.txt`): ENU component errors and 3D magnitude error
  - Velocity (from `nominal_nav.txt`): ENU magnitude error (ve/vn/vu columns)
  - Attitude: quaternion geodesic angle error (deg)

Usage:
  python tools/compare_nav_stats.py --ref <ref_dir> --other <other_dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_trajectory_enu(path: Path):
    t, e, n, u, qx, qy, qz, qw = [], [], [], [], [], [], [], []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            t.append(float(parts[0]))
            e.append(float(parts[1]))
            n.append(float(parts[2]))
            u.append(float(parts[3]))
            qx.append(float(parts[4]))
            qy.append(float(parts[5]))
            qz.append(float(parts[6]))
            qw.append(float(parts[7]))
    return (
        np.asarray(t),
        np.asarray(e),
        np.asarray(n),
        np.asarray(u),
        np.asarray(qx),
        np.asarray(qy),
        np.asarray(qz),
        np.asarray(qw),
    )


def load_nominal_nav(path: Path):
    # Columns: time lat lon h ve vn vu qx qy qz qw bg ba sg sa
    t, ve, vn, vu, qx, qy, qz, qw = [], [], [], [], [], [], [], []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 11:
                continue
            t.append(float(parts[0]))
            ve.append(float(parts[4]))
            vn.append(float(parts[5]))
            vu.append(float(parts[6]))
            qx.append(float(parts[7]))
            qy.append(float(parts[8]))
            qz.append(float(parts[9]))
            qw.append(float(parts[10]))
    return (
        np.asarray(t),
        np.asarray(ve),
        np.asarray(vn),
        np.asarray(vu),
        np.asarray(qx),
        np.asarray(qy),
        np.asarray(qz),
        np.asarray(qw),
    )


def quat_geodesic_angle_rad(qx0, qy0, qz0, qw0, qx1, qy1, qz1, qw1):
    q0 = np.stack([qx0, qy0, qz0, qw0], axis=1)
    q1 = np.stack([qx1, qy1, qz1, qw1], axis=1)
    q0 = q0 / np.linalg.norm(q0, axis=1, keepdims=True)
    q1 = q1 / np.linalg.norm(q1, axis=1, keepdims=True)
    dot = np.sum(q0 * q1, axis=1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    return 2.0 * np.arccos(dot)


def stats_abs(x):
    x = np.asarray(x)
    xabs = np.abs(x)
    rms = float(np.sqrt(np.mean(xabs * xabs)))
    p95 = float(np.percentile(xabs, 95))
    mx = float(np.max(xabs))
    return rms, p95, mx


def stats_mag(dx, dy, dz):
    mag = np.sqrt(dx * dx + dy * dy + dz * dz)
    rms = float(np.sqrt(np.mean(mag * mag)))
    p95 = float(np.percentile(mag, 95))
    mx = float(np.max(mag))
    return rms, p95, mx


def aligned_by_common_timestamps(t_ref: np.ndarray, t_other: np.ndarray):
    common = np.intersect1d(t_ref, t_other)
    if common.size < 2:
        raise RuntimeError("Not enough common timestamps to compare.")
    map_ref = {float(t): i for i, t in enumerate(t_ref)}
    map_other = {float(t): i for i, t in enumerate(t_other)}
    iref = [map_ref[float(t)] for t in common]
    iother = [map_other[float(t)] for t in common]
    return common, np.asarray(iref, dtype=int), np.asarray(iother, dtype=int)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True, type=Path, help="Reference output dir")
    p.add_argument("--other", required=True, type=Path, help="Other output dir")
    args = p.parse_args()

    ref = args.ref
    oth = args.other

    traj_ref = ref / "trajectory_enu.txt"
    traj_oth = oth / "trajectory_enu.txt"
    nav_ref = ref / "nominal_nav.txt"
    nav_oth = oth / "nominal_nav.txt"

    tr, er, nr, ur, qx0, qy0, qz0, qw0 = load_trajectory_enu(traj_ref)
    to, eo, no, uo, qx1, qy1, qz1, qw1 = load_trajectory_enu(traj_oth)
    _, iref, ioth = aligned_by_common_timestamps(tr, to)

    d_e = eo[ioth] - er[iref]
    d_n = no[ioth] - nr[iref]
    d_u = uo[ioth] - ur[iref]

    rms, p95, mx = stats_mag(d_e, d_n, d_u)
    print(f"Position ENU magnitude error (trajectory_enu): RMS={rms:.6g} P95={p95:.6g} Max={mx:.6g}")
    for comp_name, comp_err in [("east", d_e), ("north", d_n), ("up", d_u)]:
        r, p95c, mxc = stats_abs(comp_err)
        print(f"  {comp_name} |delta|: RMS={r:.6g} P95={p95c:.6g} Max={mxc:.6g}")

    ang_rad = quat_geodesic_angle_rad(
        qx0[iref],
        qy0[iref],
        qz0[iref],
        qw0[iref],
        qx1[ioth],
        qy1[ioth],
        qz1[ioth],
        qw1[ioth],
    )
    ang_deg = np.degrees(ang_rad)
    r, p95c, mxc = stats_abs(ang_deg)
    print(f"Attitude geodesic angle error (trajectory_enu quats, deg): RMS={r:.6g} P95={p95c:.6g} Max={mxc:.6g}")

    TN, veN, vnN, vuN, nq0x, nq0y, nq0z, nq0w = load_nominal_nav(nav_ref)
    T2, ve2, vn2, vu2, nq1x, nq1y, nq1z, nq1w = load_nominal_nav(nav_oth)
    _, iref2, ioth2 = aligned_by_common_timestamps(TN, T2)

    d_ve = ve2[ioth2] - veN[iref2]
    d_vn = vn2[ioth2] - vnN[iref2]
    d_vu = vu2[ioth2] - vuN[iref2]

    rms, p95, mx = stats_mag(d_ve, d_vn, d_vu)
    print(f"Velocity ENU magnitude error (nominal_nav): RMS={rms:.6g} P95={p95:.6g} Max={mx:.6g}")

    ang_rad2 = quat_geodesic_angle_rad(
        nq0x[iref2],
        nq0y[iref2],
        nq0z[iref2],
        nq0w[iref2],
        nq1x[ioth2],
        nq1y[ioth2],
        nq1z[ioth2],
        nq1w[ioth2],
    )
    ang_deg2 = np.degrees(ang_rad2)
    r, p95c, mxc = stats_abs(ang_deg2)
    print(f"Attitude geodesic angle error (nominal_nav quats, deg): RMS={r:.6g} P95={p95c:.6g} Max={mxc:.6g}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

