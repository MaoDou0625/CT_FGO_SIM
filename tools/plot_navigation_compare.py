#!/usr/bin/env python3
"""
Plot navigation comparison: ENU position and attitude (roll/pitch/yaw, deg) from two
`dense_trajectory_enu.txt` exports (e.g. batch vs sliding).

Requires: numpy, matplotlib, scipy; optional pandas (duplicate timestamps).

Example:
  python tools/plot_navigation_compare.py \\
    D:/Code/dataset/output/.../compare_batch_FGO/dense_trajectory_enu.txt \\
    D:/Code/dataset/output/.../compare_sw_marg_FGO/dense_trajectory_enu.txt \\
    -o D:/Code/dataset/output/.../nav_compare.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    print("matplotlib is required: pip install matplotlib", file=sys.stderr)
    raise SystemExit(1) from e

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from scipy.interpolate import interp1d
    from scipy.spatial.transform import Rotation as SciRotation
except ImportError:
    interp1d = None
    SciRotation = None


def load_dense(path: Path) -> tuple[np.ndarray, ...]:
    cols = ["time", "east", "north", "up", "qx", "qy", "qz", "qw"]
    if pd is not None:
        df = pd.read_csv(
            path,
            sep=r"\s+",
            comment="#",
            header=None,
            names=cols,
            engine="python",
        )
        df = df.groupby("time", as_index=False).mean(numeric_only=True)
        t = df["time"].to_numpy()
        e = df["east"].to_numpy()
        n = df["north"].to_numpy()
        u = df["up"].to_numpy()
        qx = df["qx"].to_numpy()
        qy = df["qy"].to_numpy()
        qz = df["qz"].to_numpy()
        qw = df["qw"].to_numpy()
        return t, e, n, u, qx, qy, qz, qw

    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 8:
                continue
            rows.append([float(x) for x in p[:8]])
    if not rows:
        raise ValueError(f"No data in {path}")
    arr = np.array(rows, dtype=float)
    t, e, n, u, qx, qy, qz, qw = [arr[:, i] for i in range(8)]
    t_key = np.round(t, 6)
    uniq = np.unique(t_key)
    out = []
    for tk in uniq:
        m = t_key == tk
        out.append(
            [
                float(tk),
                np.mean(e[m]),
                np.mean(n[m]),
                np.mean(u[m]),
                np.mean(qx[m]),
                np.mean(qy[m]),
                np.mean(qz[m]),
                np.mean(qw[m]),
            ]
        )
    arr = np.array(out, dtype=float)
    return tuple(arr[:, i] for i in range(8))


def quat_xyzw_to_rpy_deg(qx: np.ndarray, qy: np.ndarray, qz: np.ndarray, qw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if SciRotation is None:
        raise SystemExit("scipy is required for attitude: pip install scipy")
    q = np.stack([qx, qy, qz, qw], axis=1)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    rot = SciRotation.from_quat(q)
    euler = rot.as_euler("ZYX", degrees=True)
    yaw = euler[:, 0]
    pitch = euler[:, 1]
    roll = euler[:, 2]
    return roll, pitch, yaw


def wrap_deg(d: np.ndarray) -> np.ndarray:
    return (d + 180.0) % 360.0 - 180.0


def quat_angle_deg(qx0, qy0, qz0, qw0, qx1, qy1, qz1, qw1) -> np.ndarray:
    """Geodesic angle between matched quaternions (deg)."""
    q0 = np.stack([qx0, qy0, qz0, qw0], axis=1)
    q1 = np.stack([qx1, qy1, qz1, qw1], axis=1)
    q0 /= np.linalg.norm(q0, axis=1, keepdims=True)
    q1 /= np.linalg.norm(q1, axis=1, keepdims=True)
    dot = np.sum(q0 * q1, axis=1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    return 2.0 * np.degrees(np.arccos(dot))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("reference", type=Path, help="Reference dense_trajectory_enu.txt (e.g. batch)")
    p.add_argument("other", type=Path, help="Other trajectory (e.g. sliding)")
    p.add_argument("-o", "--output", type=Path, default=None, help="Output image path (.png / .pdf)")
    p.add_argument("--max-points", type=int, default=8000, help="Max points on common time grid")
    args = p.parse_args()

    if interp1d is None or SciRotation is None:
        print("scipy is required: pip install scipy", file=sys.stderr)
        return 1

    t0, e0, n0, u0, qx0, qy0, qz0, qw0 = load_dense(args.reference)
    t1, e1, n1, u1, qx1, qy1, qz1, qw1 = load_dense(args.other)

    t_min = max(float(t0.min()), float(t1.min()))
    t_max = min(float(t0.max()), float(t1.max()))
    if t_max <= t_min:
        print("No overlapping time between trajectories.", file=sys.stderr)
        return 1

    n_grid = min(int(args.max_points), max(len(t0), len(t1)))
    t_grid = np.linspace(t_min, t_max, num=max(100, n_grid))

    def interp_vec(t_src: np.ndarray, *series: np.ndarray) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for s in series:
            out.append(
                interp1d(
                    t_src,
                    s,
                    kind="linear",
                    bounds_error=False,
                    fill_value=np.nan,
                )(t_grid)
            )
        return out

    e0i, n0i, u0i, qx0i, qy0i, qz0i, qw0i = interp_vec(t0, e0, n0, u0, qx0, qy0, qz0, qw0)
    e1i, n1i, u1i, qx1i, qy1i, qz1i, qw1i = interp_vec(t1, e1, n1, u1, qx1, qy1, qz1, qw1)

    r0, p0, y0 = quat_xyzw_to_rpy_deg(qx0i, qy0i, qz0i, qw0i)
    r1, p1, y1 = quat_xyzw_to_rpy_deg(qx1i, qy1i, qz1i, qw1i)
    ang = quat_angle_deg(qx0i, qy0i, qz0i, qw0i, qx1i, qy1i, qz1i, qw1i)

    t_rel = t_grid - t_min

    fig, axes = plt.subplots(4, 3, figsize=(14, 12), sharex=True, constrained_layout=True)
    ref_label = "reference"
    oth_label = "other"

    # Row 0–2: position
    for row, (name, a0, a1) in enumerate(
        [
            ("East (m)", e0i, e1i),
            ("North (m)", n0i, n1i),
            ("Up (m)", u0i, u1i),
        ]
    ):
        ax0 = axes[row, 0]
        ax1 = axes[row, 1]
        ax2 = axes[row, 2]
        ax0.plot(t_rel, a0, color="C0", lw=0.8, label=ref_label)
        ax0.set_ylabel(name)
        ax0.legend(loc="upper right", fontsize=8)
        ax0.grid(True, alpha=0.3)
        ax1.plot(t_rel, a1, color="C1", lw=0.8, label=oth_label)
        ax1.legend(loc="upper right", fontsize=8)
        ax1.grid(True, alpha=0.3)
        err = a1 - a0
        ax2.plot(t_rel, err, color="C2", lw=0.8)
        ax2.set_ylabel("Δ " + name.split()[0])
        ax2.grid(True, alpha=0.3)

    # Row 3: attitude
    for col, (name, a0, a1) in enumerate(
        [
            ("Roll (deg)", r0, r1),
            ("Pitch (deg)", p0, p1),
            ("Yaw (deg)", y0, y1),
        ]
    ):
        ax0 = axes[3, col]
        ax0.plot(t_rel, a0, color="C0", lw=0.7, alpha=0.9, label=ref_label)
        ax0.plot(t_rel, a1, color="C1", lw=0.7, alpha=0.9, label=oth_label)
        ax0.set_ylabel(name)
        ax0.legend(loc="upper right", fontsize=7)
        ax0.grid(True, alpha=0.3)

    axes[3, 0].set_xlabel("time − t₀ (s)")
    axes[3, 1].set_xlabel("time − t₀ (s)")
    axes[3, 2].set_xlabel("time − t₀ (s)")

    fig.suptitle(
        f"Navigation compare\nref: {args.reference.name}\nother: {args.other.name}",
        fontsize=11,
    )

    # Extra small figure: quaternion geodesic angle
    fig2, ax = plt.subplots(figsize=(10, 3), constrained_layout=True)
    ax.plot(t_rel, ang, color="C3", lw=0.8)
    ax.set_ylabel("|q_err| (deg)")
    ax.set_xlabel("time − t₀ (s)")
    ax.set_title("Attitude difference: 2·arccos(|⟨q_ref,q_other⟩|)")
    ax.grid(True, alpha=0.3)

    out = args.output
    if out is None:
        out = args.other.parent / "nav_compare_position_attitude.png"
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    ang_path = out.with_name(out.stem + "_quat_angle" + out.suffix)
    fig2.savefig(ang_path, dpi=160)
    plt.close(fig)
    plt.close(fig2)

    print(f"Wrote {out}")
    print(f"Wrote {ang_path}")
    rms = lambda x: float(np.sqrt(np.nanmean(x**2)))
    print(f"RMS pos d_east/d_north/d_up (m): {rms(e1i-e0i):.6g}, {rms(n1i-n0i):.6g}, {rms(u1i-u0i):.6g}")
    print(f"RMS |q_angle| (deg): {rms(ang):.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
