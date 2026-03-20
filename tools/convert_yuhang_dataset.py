from __future__ import annotations

import argparse
import math
from pathlib import Path


KF_GNSS_STD = (0.02, 0.02, 0.03)


def convert_rtk(src_path: Path, dst_path: Path) -> None:
    deg_to_rad = math.pi / 180.0
    with src_path.open("r", encoding="utf-8") as src, dst_path.open("w", encoding="utf-8", newline="\n") as dst:
        dst.write("# time_s lat_rad lon_rad h_m\n")
        for line in src:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            time_s = float(parts[0])
            lon_deg = float(parts[1])
            lat_deg = float(parts[2])
            h_m = float(parts[3])
            dst.write(f"{time_s:.10f} {lat_deg * deg_to_rad:.12f} {lon_deg * deg_to_rad:.12f} {h_m:.6f}\n")


def convert_rtk_kfgins(src_path: Path, dst_path: Path) -> None:
    with src_path.open("r", encoding="utf-8") as src, dst_path.open("w", encoding="utf-8", newline="\n") as dst:
        for line in src:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            time_s = float(parts[0])
            lon_deg = float(parts[1])
            lat_deg = float(parts[2])
            h_m = float(parts[3])
            dst.write(
                f"{time_s:.12e} {lat_deg:.12e} {lon_deg:.12e} {h_m:.12e} "
                f"{KF_GNSS_STD[0]:.12e} {KF_GNSS_STD[1]:.12e} {KF_GNSS_STD[2]:.12e}\n"
            )


def convert_imu(src_path: Path, dst_path: Path) -> None:
    degph_to_radps = math.pi / (180.0 * 3600.0)
    with src_path.open("r", encoding="utf-8") as src, dst_path.open("w", encoding="utf-8", newline="\n") as dst:
        dst.write("# time_s gyro_x_radps gyro_y_radps gyro_z_radps accel_x_mps2 accel_y_mps2 accel_z_mps2\n")
        for line in src:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            time_s = float(parts[0])
            gx = float(parts[1]) * degph_to_radps
            gy = float(parts[2]) * degph_to_radps
            gz = float(parts[3]) * degph_to_radps
            ax = float(parts[4])
            ay = float(parts[5])
            az = float(parts[6])
            dst.write(f"{time_s:.10f} {gx:.12e} {gy:.12e} {gz:.12e} {ax:.9f} {ay:.9f} {az:.9f}\n")


def convert_imu_kfgins(src_path: Path, dst_path: Path) -> None:
    degph_to_radps = math.pi / (180.0 * 3600.0)
    rows: list[tuple[float, float, float, float, float, float, float]] = []
    with src_path.open("r", encoding="utf-8") as src:
        for line in src:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            rows.append(tuple(float(parts[i]) for i in range(7)))

    if not rows:
        dst_path.write_text("", encoding="utf-8")
        return

    if len(rows) >= 2:
        nominal_dt = rows[1][0] - rows[0][0]
    else:
        nominal_dt = 0.0

    with dst_path.open("w", encoding="utf-8", newline="\n") as dst:
        for idx, row in enumerate(rows):
            time_s, gx_degph, gy_degph, gz_degph, ax, ay, az = row
            dt = nominal_dt if idx == 0 else time_s - rows[idx - 1][0]

            # Match the existing KF-GINS dataset convention:
            # raw BRU-like source -> FRD increments via diag(1, -1, -1).
            dtheta_x = gx_degph * degph_to_radps * dt
            dtheta_y = -gy_degph * degph_to_radps * dt
            dtheta_z = -gz_degph * degph_to_radps * dt
            dvel_x = ax * dt
            dvel_y = -ay * dt
            dvel_z = -az * dt

            dst.write(
                f"{time_s:.12e} {dtheta_x:.12e} {dtheta_y:.12e} {dtheta_z:.12e} "
                f"{dvel_x:.12e} {dvel_y:.12e} {dvel_z:.12e}\n"
            )


def convert_one(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    convert_rtk_kfgins(src_dir / "rtk_cut.txt", dst_dir / "rtk_kfgins.txt")
    convert_imu_kfgins(src_dir / "imu_cut.txt", dst_dir / "imu_kfgins.txt")


def iter_source_segments(src_root: Path) -> list[Path]:
    segments: list[Path] = []
    for dataset_dir in sorted(src_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        if dataset_dir.name.startswith("ct_fgo_"):
            continue
        for segment_dir in sorted(dataset_dir.glob("transformed1cut*")):
            if not segment_dir.is_dir():
                continue
            if not (segment_dir / "rtk_cut.txt").exists():
                continue
            if not (segment_dir / "imu_cut.txt").exists():
                continue
            segments.append(segment_dir)
    return segments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir")
    parser.add_argument("--dst-dir")
    parser.add_argument("--src-root")
    parser.add_argument("--dst-root")
    args = parser.parse_args()

    if args.src_dir and args.dst_dir:
        convert_one(Path(args.src_dir), Path(args.dst_dir))
        return

    if args.src_root and args.dst_root:
        src_root = Path(args.src_root)
        dst_root = Path(args.dst_root)
        for src_dir in iter_source_segments(src_root):
            relative = src_dir.relative_to(src_root)
            convert_one(src_dir, dst_root / relative)
        return

    raise SystemExit("Use either --src-dir/--dst-dir or --src-root/--dst-root")


if __name__ == "__main__":
    main()
