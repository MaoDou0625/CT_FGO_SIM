from __future__ import annotations

import argparse
import math
from pathlib import Path


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


def write_schema(dst_dir: Path) -> None:
    schema = """# CT_FGO_SIM Input Schema

## rtk_ct_fgo_sim.txt

Columns:
1. `time_s`
2. `lat_rad`
3. `lon_rad`
4. `h_m`

## imu_ct_fgo_sim.txt

Columns:
1. `time_s`
2. `gyro_x_radps`
3. `gyro_y_radps`
4. `gyro_z_radps`
5. `accel_x_mps2`
6. `accel_y_mps2`
7. `accel_z_mps2`
"""
    (dst_dir / "schema.md").write_text(schema, encoding="utf-8")


def convert_one(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    convert_rtk(src_dir / "rtk_cut.txt", dst_dir / "rtk_ct_fgo_sim.txt")
    convert_imu(src_dir / "imu_cut.txt", dst_dir / "imu_ct_fgo_sim.txt")
    write_schema(dst_dir)


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
