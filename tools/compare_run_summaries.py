#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_summary(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare CT-FGO run_summary files.")
    parser.add_argument("baseline", type=Path, help="Baseline run_summary.txt")
    parser.add_argument("candidate", type=Path, help="Candidate run_summary.txt")
    args = parser.parse_args()

    b = parse_summary(args.baseline)
    c = parse_summary(args.candidate)

    print("== Key navigation consistency fields ==")
    nav_keys = [
        "gnss_count",
        "imu_count",
        "control_point_count",
        "initial_q_nb_xyzw",
        "initial_bg0_rps",
        "initial_ba0_mps2",
        "origin_blh_rad",
    ]
    inconsistent = False
    for key in nav_keys:
        vb = b.get(key, "<missing>")
        vc = c.get(key, "<missing>")
        same = vb == vc
        print(f"{key}: {'same' if same else 'DIFF'}")
        if not same:
            inconsistent = True
            print(f"  baseline : {vb}")
            print(f"  candidate: {vc}")

    impl = c.get("graph_backend_impl", "<unknown>")
    req = c.get("graph_backend_requested", c.get("graph_backend", "<unknown>"))
    print("\n== Backend info (candidate) ==")
    print(f"requested: {req}")
    print(f"impl: {impl}")

    print("\n== Course vs RTK (GNSS diff) heading (candidate) ==")
    for key in (
        "course_vs_rtk_yaw_rms_deg",
        "course_vs_rtk_yaw_weighted_median_deg_est",
        "course_vs_rtk_yaw_sample_count",
        "course_vs_rtk_yaw_min_speed_mps",
    ):
        if key in c:
            print(f"{key}: {c[key]}")

    print("\n== Time coverage (candidate) ==")
    for key in (
        "config_start_time_s",
        "config_end_time_s",
        "last_gnss_time_s",
        "last_control_point_time_s",
    ):
        if key in c:
            print(f"{key}: {c[key]}")

    print("\n== Runtime breakdown (candidate) ==")
    build_solve = float(c.get("sliding_window_total_build_solve_s", "0"))
    marginal = float(c.get("sliding_window_total_marginalization_s", "0"))
    reprop = float(c.get("sliding_window_total_reprop_s", "0"))
    total = build_solve + marginal + reprop
    if total <= 0:
        print("No runtime totals found.")
        return 1 if inconsistent else 0

    print(f"total_s: {total:.6f}")
    print(f"build_solve_s: {build_solve:.6f} ({100.0 * build_solve / total:.3f}%)")
    print(f"marginal_s: {marginal:.6f} ({100.0 * marginal / total:.3f}%)")
    print(f"reprop_s: {reprop:.6f} ({100.0 * reprop / total:.3f}%)")

    return 1 if inconsistent else 0


if __name__ == "__main__":
    raise SystemExit(main())
