from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULT_ROOT = Path(r"D:\Code\dataset\YuHangTuiChe\ct_fgo_sim_results")
ANALYSIS_ROOT = RESULT_ROOT / "common_trajectory_analysis"
DIST_THRESHOLD_M = 2.0
S_BIN_M = 1.0


@dataclass
class SegmentData:
    dataset: str
    group: str
    segment_id: str
    rtk_time: np.ndarray
    rtk_blh: np.ndarray
    nav_blh: np.ndarray
    rtk_xyz: np.ndarray
    nav_xyz: np.ndarray


def load_summary(path: Path) -> dict[str, str]:
    summary: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        summary[key.strip()] = value.strip()
    return summary


def blh_to_ecef(blh: np.ndarray) -> np.ndarray:
    lat = blh[:, 0]
    lon = blh[:, 1]
    h = blh[:, 2]
    a = 6378137.0
    e2 = 0.0066943799901413156
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    cos_lon = np.cos(lon)
    sin_lon = np.sin(lon)
    rn = a / np.sqrt(1.0 - e2 * sin_lat * sin_lat)
    x = (rn + h) * cos_lat * cos_lon
    y = (rn + h) * cos_lat * sin_lon
    z = (rn * (1.0 - e2) + h) * sin_lat
    return np.column_stack([x, y, z])


def ecef_to_blh(ecef: np.ndarray) -> np.ndarray:
    a = 6378137.0
    e2 = 0.0066943799901413156
    out = np.zeros_like(ecef)
    for i, xyz in enumerate(ecef):
        x, y, z = xyz
        p = np.hypot(x, y)
        lat = np.arctan2(z, p * (1.0 - e2))
        lon = 2.0 * np.arctan2(y, x + p)
        h = 0.0
        h_prev = 1.0
        while abs(h - h_prev) > 1.0e-4:
            h_prev = h
            sin_lat = np.sin(lat)
            rn = a / np.sqrt(1.0 - e2 * sin_lat * sin_lat)
            h = p / np.cos(lat) - rn
            lat = np.arctan2(z, p * (1.0 - e2 * rn / (rn + h)))
        out[i] = [lat, lon, h]
    return out


def cne(origin_blh: np.ndarray) -> np.ndarray:
    lat, lon, _ = origin_blh
    sin_lat = np.sin(lat)
    sin_lon = np.sin(lon)
    cos_lat = np.cos(lat)
    cos_lon = np.cos(lon)
    return np.array(
        [
            [-sin_lat * cos_lon, -sin_lon, -cos_lat * cos_lon],
            [-sin_lat * sin_lon, cos_lon, -cos_lat * sin_lon],
            [cos_lat, 0.0, -sin_lat],
        ]
    )


def local_enu_to_blh(origin_blh: np.ndarray, local_enu: np.ndarray) -> np.ndarray:
    ecef0 = blh_to_ecef(origin_blh.reshape(1, 3))[0]
    ecef = ecef0 + local_enu @ cne(origin_blh).T
    return ecef_to_blh(ecef)


def ecef_to_enu(origin_blh: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    ecef0 = blh_to_ecef(origin_blh.reshape(1, 3))[0]
    return (xyz - ecef0) @ cne(origin_blh)


def load_segments(root: Path) -> list[SegmentData]:
    segments: list[SegmentData] = []
    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for group_dir in sorted(dataset_dir.glob("transformed1cut*")):
            if not group_dir.is_dir() or "rtk_only" in group_dir.name:
                continue
            summary_path = group_dir / "run_summary.txt"
            traj_path = group_dir / "trajectory_enu.txt"
            if not summary_path.exists() or not traj_path.exists():
                continue

            summary = load_summary(summary_path)
            rtk_blh = np.loadtxt(summary["gnss_file"])[:, 1:4]
            traj = np.loadtxt(traj_path, comments="#")
            origin_blh = np.fromstring(summary["origin_blh_rad"], sep=" ")
            nav_blh = local_enu_to_blh(origin_blh, traj[:, 1:4])

            segments.append(
                SegmentData(
                    dataset=dataset_dir.name,
                    group=group_dir.name,
                    segment_id=f"{dataset_dir.name}/{group_dir.name}",
                    rtk_time=np.loadtxt(summary["gnss_file"])[:, 0],
                    rtk_blh=rtk_blh,
                    nav_blh=nav_blh,
                    rtk_xyz=blh_to_ecef(rtk_blh),
                    nav_xyz=blh_to_ecef(nav_blh),
                )
            )
    return segments


def compute_path_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def cumulative_s(points: np.ndarray) -> np.ndarray:
    ds = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(ds)])


def nearest_reference_indices(ref_points: np.ndarray, query_points: np.ndarray, chunk_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    idx_out = np.empty(query_points.shape[0], dtype=int)
    dist_out = np.empty(query_points.shape[0], dtype=float)
    for start in range(0, query_points.shape[0], chunk_size):
        stop = min(start + chunk_size, query_points.shape[0])
        q = query_points[start:stop]
        diff = q[:, None, :] - ref_points[None, :, :]
        dist2 = np.einsum("ijk,ijk->ij", diff, diff)
        idx = np.argmin(dist2, axis=1)
        idx_out[start:stop] = idx
        dist_out[start:stop] = np.sqrt(dist2[np.arange(idx.size), idx])
    return idx_out, dist_out


def write_csv(path: Path, rows: list[dict[str, float | str | int]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, summary: dict[str, float | int | str]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")


def main() -> None:
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    segments = load_segments(RESULT_ROOT)
    if len(segments) < 2:
        raise RuntimeError("Need at least two segments.")

    ref_segment = max(segments, key=lambda segment: compute_path_length(segment.rtk_xyz))
    ref_origin = ref_segment.rtk_blh[0]
    ref_xyz = ref_segment.rtk_xyz
    ref_enu = ecef_to_enu(ref_origin, ref_xyz)
    ref_s = cumulative_s(ref_enu[:, :2])

    coverage_rows: list[dict[str, float | str]] = []
    projection_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    s_starts: list[float] = []
    s_ends: list[float] = []

    for segment in segments:
        query_enu = ecef_to_enu(ref_origin, segment.rtk_xyz)
        nearest_idx, nearest_dist = nearest_reference_indices(ref_enu[:, :2], query_enu[:, :2])
        projected_s = ref_s[nearest_idx]
        valid = nearest_dist <= DIST_THRESHOLD_M
        if valid.sum() == 0:
            raise RuntimeError(f"No valid overlap for {segment.segment_id}")
        s_start = float(projected_s[valid].min())
        s_end = float(projected_s[valid].max())
        s_starts.append(s_start)
        s_ends.append(s_end)
        projection_data[segment.segment_id] = (projected_s, nearest_dist)
        coverage_rows.append(
            {
                "segment_id": segment.segment_id,
                "s_start_m": s_start,
                "s_end_m": s_end,
                "valid_ratio": float(valid.mean()),
                "mean_nearest_dist_m": float(nearest_dist[valid].mean()),
                "max_nearest_dist_m": float(nearest_dist[valid].max()),
            }
        )

    common_s_start = max(s_starts)
    common_s_end = min(s_ends)
    if common_s_end <= common_s_start:
        raise RuntimeError("No common trajectory interval found across all segments.")

    trimmed_rows: list[dict[str, float | str | int]] = []
    binned_by_segment: dict[str, dict[int, tuple[float, float, int]]] = {}
    for segment in segments:
        projected_s, nearest_dist = projection_data[segment.segment_id]
        valid = (
            (nearest_dist <= DIST_THRESHOLD_M)
            & (projected_s >= common_s_start)
            & (projected_s <= common_s_end)
        )
        s_local = projected_s[valid] - common_s_start
        rtk_h = segment.rtk_blh[valid, 2]
        nav_h = segment.nav_blh[valid, 2]
        bins = np.floor(s_local / S_BIN_M).astype(int)
        per_bin: dict[int, tuple[float, float, int]] = {}
        for bin_id in np.unique(bins):
            idxs = np.where(bins == bin_id)[0]
            per_bin[bin_id] = (
                float(rtk_h[idxs].mean()),
                float(nav_h[idxs].mean()),
                int(idxs.size),
            )
            trimmed_rows.append(
                {
                    "segment_id": segment.segment_id,
                    "s_bin": int(bin_id),
                    "s_center_m": float((bin_id + 0.5) * S_BIN_M),
                    "sample_count": int(idxs.size),
                    "rtk_height_mean_m": float(rtk_h[idxs].mean()),
                    "nav_height_mean_m": float(nav_h[idxs].mean()),
                }
            )
        binned_by_segment[segment.segment_id] = per_bin

    common_bins = set.intersection(*(set(per_bin.keys()) for per_bin in binned_by_segment.values()))
    common_bin_rows: list[dict[str, float | int]] = []
    for bin_id in sorted(common_bins):
        rtk_heights = np.array([binned_by_segment[segment.segment_id][bin_id][0] for segment in segments], dtype=float)
        nav_heights = np.array([binned_by_segment[segment.segment_id][bin_id][1] for segment in segments], dtype=float)
        common_bin_rows.append(
            {
                "s_bin": int(bin_id),
                "s_center_m": float((bin_id + 0.5) * S_BIN_M),
                "rtk_height_mean_m": float(rtk_heights.mean()),
                "rtk_height_std_m": float(rtk_heights.std()),
                "nav_height_mean_m": float(nav_heights.mean()),
                "nav_height_std_m": float(nav_heights.std()),
            }
        )

    summary = {
        "segment_count": len(segments),
        "reference_segment": ref_segment.segment_id,
        "distance_threshold_m": DIST_THRESHOLD_M,
        "s_bin_m": S_BIN_M,
        "common_s_start_m": common_s_start,
        "common_s_end_m": common_s_end,
        "common_s_length_m": common_s_end - common_s_start,
        "common_bin_count": len(common_bins),
        "rtk_repeatability_mean_std_m": float(np.mean([row["rtk_height_std_m"] for row in common_bin_rows])),
        "nav_repeatability_mean_std_m": float(np.mean([row["nav_height_std_m"] for row in common_bin_rows])),
    }

    write_csv(
        ANALYSIS_ROOT / "segment_coverage.csv",
        coverage_rows,
        ["segment_id", "s_start_m", "s_end_m", "valid_ratio", "mean_nearest_dist_m", "max_nearest_dist_m"],
    )
    write_csv(
        ANALYSIS_ROOT / "trimmed_group_heights_by_s.csv",
        trimmed_rows,
        ["segment_id", "s_bin", "s_center_m", "sample_count", "rtk_height_mean_m", "nav_height_mean_m"],
    )
    write_csv(
        ANALYSIS_ROOT / "repeatability_by_s_bin.csv",
        common_bin_rows,
        ["s_bin", "s_center_m", "rtk_height_mean_m", "rtk_height_std_m", "nav_height_mean_m", "nav_height_std_m"],
    )
    write_summary(ANALYSIS_ROOT / "common_trajectory_summary.txt", summary)

    ref_csv_rows = [
        {"ref_index": int(i), "s_m": float(s), "east_m": float(p[0]), "north_m": float(p[1]), "up_m": float(p[2])}
        for i, (s, p) in enumerate(zip(ref_s, ref_enu))
        if common_s_start <= s <= common_s_end
    ]
    write_csv(
        ANALYSIS_ROOT / "reference_common_trajectory.csv",
        ref_csv_rows,
        ["ref_index", "s_m", "east_m", "north_m", "up_m"],
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    for segment in segments:
        per_bin = binned_by_segment[segment.segment_id]
        xs = [(bin_id + 0.5) * S_BIN_M for bin_id in sorted(common_bins)]
        ys = [per_bin[bin_id][1] for bin_id in sorted(common_bins)]
        ax.plot(xs, ys, linewidth=1.2, label=segment.segment_id)
    ax.set_xlabel("Common Trajectory s (m)")
    ax.set_ylabel("Navigation Height (m)")
    ax.set_title("Navigation Height Along Common Trajectory")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ANALYSIS_ROOT / "navigation_height_vs_common_s.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    for segment in segments:
        per_bin = binned_by_segment[segment.segment_id]
        xs = [(bin_id + 0.5) * S_BIN_M for bin_id in sorted(common_bins)]
        ys = [per_bin[bin_id][0] for bin_id in sorted(common_bins)]
        ax.plot(xs, ys, linewidth=1.2, label=segment.segment_id)
    ax.set_xlabel("Common Trajectory s (m)")
    ax.set_ylabel("RTK Height (m)")
    ax.set_title("RTK Height Along Common Trajectory")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ANALYSIS_ROOT / "rtk_height_vs_common_s.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    xs = [row["s_center_m"] for row in common_bin_rows]
    ax.plot(xs, [row["rtk_height_std_m"] for row in common_bin_rows], linewidth=1.2, label="RTK height std")
    ax.plot(xs, [row["nav_height_std_m"] for row in common_bin_rows], linewidth=1.2, label="Navigation height std")
    ax.set_xlabel("Common Trajectory s (m)")
    ax.set_ylabel("Height Repeatability Std (m)")
    ax.set_title("Height Repeatability Along Common Trajectory")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ANALYSIS_ROOT / "height_repeatability_std_vs_common_s.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
