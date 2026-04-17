#!/usr/bin/env python3
"""
Diagnostic: nominal_nav ENU velocity vs composed (nominal + delta_vel from delta_estimates.txt)
vs RTK/GNSS ENU velocity from position differencing (reference only, not used in the solver).

Optional outage/recovery bands from app YAML (rtk_outage).

Prints horizontal-speed RMS for segments: clean (neither outage nor recovery band),
inside_outage, recovery_band.

Example:
  python tools/plot_velocity_diagnostic_outage.py \\
    --run-dir "D:/Code/dataset/output/.../rtk_outage_recover_40s20s" \\
    --rtk-file "D:/Code/dataset/.../rtk_ct_fgo_sim.txt" \\
    --outage-yaml "config/rtk_outage_recover_40s20s.yaml" \\
    -o "D:/Code/dataset/output/.../velocity_diagnostic.png"
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    print("matplotlib required: pip install matplotlib", file=sys.stderr)
    raise SystemExit(1) from e

try:
    from scipy.interpolate import interp1d
except ImportError:
    interp1d = None


def load_nominal_nav_enu_vel(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """nominal_nav.txt: time, ..., ve, vn, vu (ENU m/s), ..."""
    t, ve, vn, vu = [], [], [], []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 11:
                continue
            t.append(float(p[0]))
            ve.append(float(p[4]))
            vn.append(float(p[5]))
            vu.append(float(p[6]))
    return np.asarray(t), np.asarray(ve), np.asarray(vn), np.asarray(vu)


def load_delta_vel_enu(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """delta_estimates.txt: time, dtheta(3), dvx,dvy,dvz (ENU m/s), ..."""
    t, dvx, dvy, dvz = [], [], [], []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 7:
                continue
            t.append(float(p[0]))
            dvx.append(float(p[4]))
            dvy.append(float(p[5]))
            dvz.append(float(p[6]))
    return np.asarray(t), np.asarray(dvx), np.asarray(dvy), np.asarray(dvz)


def blh_rad_to_local_enu(blh_rad: np.ndarray, origin_blh_rad: np.ndarray) -> np.ndarray:
    lat0, lon0, h0 = origin_blh_rad
    lat = blh_rad[:, 1]
    lon = blh_rad[:, 2]
    h = blh_rad[:, 3]
    a = 6378137.0
    e2 = 0.0066943799901413156
    rn0 = a / np.sqrt(1.0 - e2 * np.sin(lat0) ** 2)
    x0 = (rn0 + h0) * np.cos(lat0) * np.cos(lon0)
    y0 = (rn0 + h0) * np.cos(lat0) * np.sin(lon0)
    z0 = (rn0 * (1.0 - e2) + h0) * np.sin(lat0)
    rn1 = a / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    x = (rn1 + h) * np.cos(lat) * np.cos(lon)
    y = (rn1 + h) * np.cos(lat) * np.sin(lon)
    z = (rn1 * (1.0 - e2) + h) * np.sin(lat)
    dx, dy, dz = x - x0, y - y0, z - z0
    sin_lat, cos_lat = np.sin(lat0), np.cos(lat0)
    sin_lon, cos_lon = np.sin(lon0), np.cos(lon0)
    ecef_to_enu = np.array(
        [
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ]
    )
    return np.column_stack([dx, dy, dz]) @ ecef_to_enu.T


def rtk_vel_enu(rtk_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rtk = np.loadtxt(rtk_path, comments="#")
    tt = rtk[:, 0]
    enu = blh_rad_to_local_enu(rtk, rtk[0, 1:4])
    vel = np.empty_like(enu)
    vel[1:-1] = (enu[2:] - enu[:-2]) / (tt[2:, None] - tt[:-2, None])
    vel[0] = (enu[1] - enu[0]) / max(1.0e-9, tt[1] - tt[0])
    vel[-1] = (enu[-1] - enu[-2]) / max(1.0e-9, tt[-1] - tt[-2])
    return tt, vel[:, 0], vel[:, 1], vel[:, 2]


def parse_rtk_outage_yaml(yaml_path: Path) -> tuple[list[tuple[float, float]], float]:
    text = yaml_path.read_text(encoding="utf-8", errors="replace")
    in_block = False
    ranges: list[tuple[float, float]] = []
    recovery = 20.0
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if line.strip().startswith("rtk_outage:"):
            in_block = True
            continue
        if not in_block:
            continue
        if line.strip() and not line.startswith(" ") and not line.startswith("\t"):
            break
        m = re.match(r"^\s+-\s*\[\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)\s*\]", line)
        if m:
            t0, t1 = float(m.group(1)), float(m.group(2))
            lo, hi = (t0, t1) if t0 <= t1 else (t1, t0)
            ranges.append((lo, hi))
        m2 = re.match(r"^\s*recovery_horizon_s:\s*([0-9.eE+-]+)", line)
        if m2:
            recovery = float(m2.group(1))
    return ranges, recovery


def segment_mask(
    t: np.ndarray,
    ranges: list[tuple[float, float]],
    recovery_h: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """outside: not in outage and not in recovery; inside: outage; recovery: after outage until +recovery_h."""
    in_out = np.zeros_like(t, dtype=bool)
    for lo, hi in ranges:
        in_out |= (t >= lo) & (t <= hi)
    in_rec = np.zeros_like(t, dtype=bool)
    for lo, hi in ranges:
        in_rec |= (t > hi) & (t <= hi + recovery_h)
    outside = ~(in_out | in_rec)
    return outside, in_out, in_rec


def horiz_speed(ve: np.ndarray, vn: np.ndarray) -> np.ndarray:
    return np.sqrt(ve * ve + vn * vn)


def rms_h(a_e: np.ndarray, a_n: np.ndarray, b_e: np.ndarray, b_n: np.ndarray, mask: np.ndarray) -> float:
    m = mask & np.isfinite(a_e) & np.isfinite(a_n) & np.isfinite(b_e) & np.isfinite(b_n)
    if not np.any(m):
        return float("nan")
    d = horiz_speed(a_e[m], a_n[m]) - horiz_speed(b_e[m], b_n[m])
    return float(np.sqrt(np.mean(d * d)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True, help="Output folder with nominal_nav.txt and delta_estimates.txt")
    ap.add_argument("--rtk-file", type=Path, required=True)
    ap.add_argument("--outage-yaml", type=Path, default=None)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--max-points", type=int, default=12000)
    args = ap.parse_args()

    if interp1d is None:
        print("scipy required: pip install scipy", file=sys.stderr)
        return 1

    run = args.run_dir.resolve()
    nom_p = run / "nominal_nav.txt"
    del_p = run / "delta_estimates.txt"
    if not nom_p.is_file() or not del_p.is_file():
        print(f"Missing {nom_p} or {del_p}", file=sys.stderr)
        return 1

    t_n, ve_n, vn_n, vu_n = load_nominal_nav_enu_vel(nom_p)
    t_d, dvx, dvy, dvz = load_delta_vel_enu(del_p)

    t0 = max(float(t_n.min()), float(t_d.min()))
    t1 = min(float(t_n.max()), float(t_d.max()))
    if t1 <= t0:
        print("No time overlap between nominal and delta.", file=sys.stderr)
        return 1

    n_grid = min(int(args.max_points), max(len(t_n), len(t_d)))
    t_g = np.linspace(t0, t1, num=max(200, n_grid))

    f_ve = interp1d(t_n, ve_n, kind="linear", bounds_error=False, fill_value=np.nan)
    f_vn = interp1d(t_n, vn_n, kind="linear", bounds_error=False, fill_value=np.nan)
    f_vu = interp1d(t_n, vu_n, kind="linear", bounds_error=False, fill_value=np.nan)
    f_dvx = interp1d(t_d, dvx, kind="linear", bounds_error=False, fill_value=np.nan)
    f_dvy = interp1d(t_d, dvy, kind="linear", bounds_error=False, fill_value=np.nan)
    f_dvz = interp1d(t_d, dvz, kind="linear", bounds_error=False, fill_value=np.nan)

    ve_i = f_ve(t_g)
    vn_i = f_vn(t_g)
    vu_i = f_vu(t_g)
    dvx_i = f_dvx(t_g)
    dvy_i = f_dvy(t_g)
    dvz_i = f_dvz(t_g)
    ve_c = ve_i + dvx_i
    vn_c = vn_i + dvy_i
    vu_c = vu_i + dvz_i

    tr, vre, vrn, vru = rtk_vel_enu(args.rtk_file.resolve())
    f_re = interp1d(tr, vre, kind="linear", bounds_error=False, fill_value=np.nan)
    f_rn = interp1d(tr, vrn, kind="linear", bounds_error=False, fill_value=np.nan)
    f_ru = interp1d(tr, vru, kind="linear", bounds_error=False, fill_value=np.nan)
    vre_i = f_re(t_g)
    vrn_i = f_rn(t_g)
    vru_i = f_ru(t_g)

    t_rel = t_g - t0
    ranges: list[tuple[float, float]] = []
    recovery_h = 20.0
    if args.outage_yaml is not None:
        ranges, recovery_h = parse_rtk_outage_yaml(args.outage_yaml.resolve())

    outside_m, inside_m, rec_m = segment_mask(t_g, ranges, recovery_h) if ranges else (
        np.ones_like(t_g, dtype=bool),
        np.zeros_like(t_g, dtype=bool),
        np.zeros_like(t_g, dtype=bool),
    )

    print("=== Horizontal speed RMS |composed - nominal| (m/s) ===")
    for name, m in [("clean_neither_outage_nor_recovery", outside_m), ("inside_outage", inside_m), ("recovery_band", rec_m)]:
        d = horiz_speed(ve_c, vn_c) - horiz_speed(ve_i, vn_i)
        if np.any(m & np.isfinite(d)):
            print(f"  {name}: RMS={float(np.sqrt(np.nanmean((d[m]) ** 2))):.6g}")
        else:
            print(f"  {name}: n/a")

    print("=== Horizontal speed RMS |composed - RTK| (m/s) ===")
    for name, m in [("clean_neither_outage_nor_recovery", outside_m), ("inside_outage", inside_m), ("recovery_band", rec_m)]:
        r = rms_h(ve_c, vn_c, vre_i, vrn_i, m)
        print(f"  {name}: RMS={r:.6g}")

    print("=== Horizontal speed RMS |nominal - RTK| (m/s) ===")
    for name, m in [("clean_neither_outage_nor_recovery", outside_m), ("inside_outage", inside_m), ("recovery_band", rec_m)]:
        r = rms_h(ve_i, vn_i, vre_i, vrn_i, m)
        print(f"  {name}: RMS={r:.6g}")

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, constrained_layout=True)

    def shade(ax):
        if not ranges:
            return
        for lo, hi in ranges:
            ax.axvspan(lo - t0, hi - t0, color="salmon", alpha=0.2, linewidth=0)
            ax.axvspan(hi - t0, hi + recovery_h - t0, color="lightgreen", alpha=0.15, linewidth=0)

    pairs = [
        ("East vel (m/s)", ve_i, ve_c, vre_i),
        ("North vel (m/s)", vn_i, vn_c, vrn_i),
        ("Up vel (m/s)", vu_i, vu_c, vru_i),
    ]
    for ax, (yl, a0, a1, a2) in zip(axes[:3], pairs):
        shade(ax)
        ax.plot(t_rel, a0, lw=0.8, label="nominal")
        ax.plot(t_rel, a1, lw=0.8, label="composed (nom+delta)")
        ax.plot(t_rel, a2, lw=0.75, ls="--", label="RTK diff (ref)")
        ax.set_ylabel(yl)
        ax.grid(True, alpha=0.35)
        ax.legend(loc="upper right", fontsize=7)
    shade(axes[3])
    axes[3].plot(t_rel, horiz_speed(ve_i, vn_i), lw=0.8, label="nominal |v_h|")
    axes[3].plot(t_rel, horiz_speed(ve_c, vn_c), lw=0.8, label="composed |v_h|")
    axes[3].plot(t_rel, horiz_speed(vre_i, vrn_i), lw=0.75, ls="--", label="RTK |v_h|")
    axes[3].set_ylabel("Horizontal speed (m/s)")
    axes[3].legend(loc="upper right", fontsize=7)
    axes[3].grid(True, alpha=0.35)
    axes[-1].set_xlabel("time − t0 (s)")
    fig.suptitle(f"Velocity diagnostic — {run.name}", fontsize=11)
    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
