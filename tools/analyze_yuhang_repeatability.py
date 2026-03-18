from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


RESULT_ROOT = Path(r"D:\Code\dataset\YuHangTuiChe\ct_fgo_sim_results")
MIN_DECIMALS = 4
MAX_DECIMALS = 10


@dataclass
class SegmentData:
    dataset: str
    group: str
    segment_id: str
    rtk_time: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    rtk_h: np.ndarray
    nav_h: np.ndarray


def load_summary(path: Path) -> dict[str, str]:
    summary: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        summary[key.strip()] = value.strip()
    return summary


def load_segments(root: Path) -> list[SegmentData]:
    segments: list[SegmentData] = []
    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for group_dir in sorted(dataset_dir.glob("transformed1cut*")):
            if not group_dir.is_dir():
                continue
            if "rtk_only" in group_dir.name:
                continue
            summary_path = group_dir / "run_summary.txt"
            traj_path = group_dir / "trajectory_enu.txt"
            if not summary_path.exists() or not traj_path.exists():
                continue

            summary = load_summary(summary_path)
            rtk_path = Path(summary["gnss_file"])
            rtk = np.loadtxt(rtk_path)
            traj = np.loadtxt(traj_path, comments="#")
            origin_blh = np.fromstring(summary["origin_blh_rad"], sep=" ")
            nav_blh = local_enu_to_blh(origin_blh, traj[:, 1:4])
            nav_h = np.interp(rtk[:, 0], traj[:, 0], nav_blh[:, 2])

            segments.append(
                SegmentData(
                    dataset=dataset_dir.name,
                    group=group_dir.name,
                    segment_id=f"{dataset_dir.name}/{group_dir.name}",
                    rtk_time=rtk[:, 0],
                    lat=rtk[:, 1],
                    lon=rtk[:, 2],
                    rtk_h=rtk[:, 3],
                    nav_h=nav_h,
                )
            )
    return segments


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


def quantized_position_keys(segment: SegmentData, decimals: int) -> list[tuple[float, float]]:
    lat_q = np.round(segment.lat, decimals)
    lon_q = np.round(segment.lon, decimals)
    return list(zip(lat_q.tolist(), lon_q.tolist()))


def find_finest_common_decimals(segments: list[SegmentData]) -> tuple[int, set[tuple[float, float]]]:
    for decimals in range(MAX_DECIMALS, MIN_DECIMALS - 1, -1):
        key_sets = [set(quantized_position_keys(segment, decimals)) for segment in segments]
        common = set.intersection(*key_sets)
        if common:
            return decimals, common
    raise RuntimeError("No common RTK positions found across all segments in the configured precision range.")


def build_group_records(
    segments: list[SegmentData], decimals: int, common_keys: set[tuple[float, float]]
) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    for segment in segments:
        lat_q = np.round(segment.lat, decimals)
        lon_q = np.round(segment.lon, decimals)
        per_key_indices: dict[tuple[float, float], list[int]] = {}
        for idx, key in enumerate(zip(lat_q.tolist(), lon_q.tolist())):
            if key not in common_keys:
                continue
            per_key_indices.setdefault(key, []).append(idx)

        for key in sorted(per_key_indices):
            idxs = np.asarray(per_key_indices[key], dtype=int)
            rows.append(
                {
                    "segment_id": segment.segment_id,
                    "dataset": segment.dataset,
                    "group": segment.group,
                    "lat_q_rad": key[0],
                    "lon_q_rad": key[1],
                    "sample_count": int(idxs.size),
                    "time_mean_s": float(segment.rtk_time[idxs].mean()),
                    "rtk_height_mean_m": float(segment.rtk_h[idxs].mean()),
                    "rtk_height_std_within_group_m": float(segment.rtk_h[idxs].std()),
                    "nav_height_mean_m": float(segment.nav_h[idxs].mean()),
                    "nav_height_std_within_group_m": float(segment.nav_h[idxs].std()),
                }
            )
    return rows


def build_repeatability_rows(group_rows: list[dict[str, float | str | int]]) -> list[dict[str, float | str | int]]:
    grouped: dict[tuple[float, float], list[dict[str, float | str | int]]] = {}
    for row in group_rows:
        key = (float(row["lat_q_rad"]), float(row["lon_q_rad"]))
        grouped.setdefault(key, []).append(row)

    rows: list[dict[str, float | str | int]] = []
    for key in sorted(grouped):
        items = grouped[key]
        rtk_heights = np.array([float(item["rtk_height_mean_m"]) for item in items], dtype=float)
        nav_heights = np.array([float(item["nav_height_mean_m"]) for item in items], dtype=float)
        rows.append(
            {
                "lat_q_rad": key[0],
                "lon_q_rad": key[1],
                "segment_count": len(items),
                "rtk_height_mean_m": float(rtk_heights.mean()),
                "rtk_height_std_m": float(rtk_heights.std()),
                "rtk_height_range_m": float(rtk_heights.max() - rtk_heights.min()),
                "nav_height_mean_m": float(nav_heights.mean()),
                "nav_height_std_m": float(nav_heights.std()),
                "nav_height_range_m": float(nav_heights.max() - nav_heights.min()),
                "repeatability_gain_m": float(nav_heights.std() - rtk_heights.std()),
            }
        )
    return rows


def summarize_repeatability(
    segments: list[SegmentData],
    decimals: int,
    repeatability_rows: list[dict[str, float | str | int]],
) -> dict[str, float | int]:
    rtk_stds = np.array([float(row["rtk_height_std_m"]) for row in repeatability_rows], dtype=float)
    nav_stds = np.array([float(row["nav_height_std_m"]) for row in repeatability_rows], dtype=float)
    return {
        "segment_count": len(segments),
        "common_position_count": len(repeatability_rows),
        "lat_lon_round_decimals": decimals,
        "approx_position_bin_m": float(6378137.0 * (10.0 ** (-decimals))),
        "rtk_repeatability_mean_std_m": float(rtk_stds.mean()),
        "rtk_repeatability_median_std_m": float(np.median(rtk_stds)),
        "rtk_repeatability_max_std_m": float(rtk_stds.max()),
        "nav_repeatability_mean_std_m": float(nav_stds.mean()),
        "nav_repeatability_median_std_m": float(np.median(nav_stds)),
        "nav_repeatability_max_std_m": float(nav_stds.max()),
        "nav_minus_rtk_mean_std_m": float(nav_stds.mean() - rtk_stds.mean()),
    }


def write_csv(path: Path, rows: list[dict[str, float | str | int]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, summary: dict[str, float | int]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")


def main() -> None:
    analysis_root = RESULT_ROOT / "repeatability_analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)

    segments = load_segments(RESULT_ROOT)
    decimals, common_keys = find_finest_common_decimals(segments)
    group_rows = build_group_records(segments, decimals, common_keys)
    repeatability_rows = build_repeatability_rows(group_rows)
    summary = summarize_repeatability(segments, decimals, repeatability_rows)

    write_csv(
        analysis_root / "common_position_group_heights.csv",
        group_rows,
        [
            "segment_id",
            "dataset",
            "group",
            "lat_q_rad",
            "lon_q_rad",
            "sample_count",
            "time_mean_s",
            "rtk_height_mean_m",
            "rtk_height_std_within_group_m",
            "nav_height_mean_m",
            "nav_height_std_within_group_m",
        ],
    )
    write_csv(
        analysis_root / "common_position_repeatability.csv",
        repeatability_rows,
        [
            "lat_q_rad",
            "lon_q_rad",
            "segment_count",
            "rtk_height_mean_m",
            "rtk_height_std_m",
            "rtk_height_range_m",
            "nav_height_mean_m",
            "nav_height_std_m",
            "nav_height_range_m",
            "repeatability_gain_m",
        ],
    )
    write_summary(analysis_root / "repeatability_summary.txt", summary)


if __name__ == "__main__":
    main()
