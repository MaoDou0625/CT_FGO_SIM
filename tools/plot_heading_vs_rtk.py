from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

R_ENU_FROM_NED = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])


def heading_deg_0_360(angle_deg: np.ndarray) -> np.ndarray:
    return np.mod(angle_deg, 360.0)


def wrap_deg(angle_deg: np.ndarray) -> np.ndarray:
    return (angle_deg + 180.0) % 360.0 - 180.0


def quat_to_rotmat_xyzw(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )


def yaw_deg_from_ct_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray]:
    traj = np.loadtxt(path, comments="#")
    t = traj[:, 0]
    qx, qy, qz, qw = traj[:, 4], traj[:, 5], traj[:, 6], traj[:, 7]
    yaw = np.empty_like(t)
    for i in range(t.size):
        # trajectory_enu quaternion is stored in ENU convention; map to NED before extracting yaw.
        r_enu = quat_to_rotmat_xyzw(float(qx[i]), float(qy[i]), float(qz[i]), float(qw[i]))
        r_ned = R_ENU_FROM_NED.T @ r_enu
        yaw[i] = np.degrees(np.arctan2(r_ned[1, 0], r_ned[0, 0]))
    return t, heading_deg_0_360(yaw)


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
        [[-sin_lon, cos_lon, 0.0], [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat], [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]]
    )
    return np.column_stack([dx, dy, dz]) @ ecef_to_enu.T


def heading_deg_from_rtk(path: Path, speed_thresh: float) -> tuple[np.ndarray, np.ndarray]:
    rtk = np.loadtxt(path)
    t = rtk[:, 0]
    enu = blh_rad_to_local_enu(rtk, rtk[0, 1:4])
    vel = np.empty_like(enu)
    vel[1:-1] = (enu[2:] - enu[:-2]) / (t[2:, None] - t[:-2, None])
    vel[0] = (enu[1] - enu[0]) / max(1.0e-6, t[1] - t[0])
    vel[-1] = (enu[-1] - enu[-2]) / max(1.0e-6, t[-1] - t[-2])
    speed = np.linalg.norm(vel[:, :2], axis=1)
    yaw = heading_deg_0_360(np.degrees(np.arctan2(vel[:, 0], vel[:, 1])))
    yaw[speed < speed_thresh] = np.nan
    return t, yaw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--rtk", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--speed-threshold", type=float, default=0.8)
    args = ap.parse_args()

    ct_t, ct_yaw = yaw_deg_from_ct_trajectory(Path(args.traj))
    rtk_t, rtk_yaw = heading_deg_from_rtk(Path(args.rtk), args.speed_threshold)
    rtk_interp = np.interp(ct_t, rtk_t, np.nan_to_num(rtk_yaw, nan=0.0))
    valid = ~np.isnan(np.interp(ct_t, rtk_t, rtk_yaw, left=np.nan, right=np.nan))
    err = np.full_like(ct_t, np.nan, dtype=float)
    err[valid] = wrap_deg(ct_yaw[valid] - rtk_interp[valid])
    rms = float(np.sqrt(np.nanmean(err[valid] ** 2))) if np.any(valid) else float("nan")

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(ct_t, ct_yaw, label="CT yaw", linewidth=1.1)
    axes[0].plot(rtk_t, rtk_yaw, label="RTK heading", linewidth=1.0)
    axes[0].set_ylabel("Heading (deg)")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()
    axes[1].plot(ct_t, err, label=f"Yaw error (RMS={rms:.3f} deg)", linewidth=1.1)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("CT-RTK (deg)")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()
    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Wrote {out_path}")
    print(f"heading_rms_deg: {rms}")


if __name__ == "__main__":
    main()
