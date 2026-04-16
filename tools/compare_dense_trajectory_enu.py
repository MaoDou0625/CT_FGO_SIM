#!/usr/bin/env python3
"""
Compare two dense_trajectory_enu.txt files (same time column) and report position RMSE.

Usage:
  python tools/compare_dense_trajectory_enu.py path/to/batch.txt path/to/sliding.txt
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


def load_dense(path: Path) -> dict[float, tuple[float, float, float]]:
    rows: dict[float, tuple[float, float, float]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            t = float(parts[0])
            rows[t] = (float(parts[1]), float(parts[2]), float(parts[3]))
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference", type=Path)
    p.add_argument("other", type=Path)
    args = p.parse_args()
    ref = load_dense(args.reference)
    oth = load_dense(args.other)
    common = sorted(set(ref) & set(oth))
    if len(common) < 2:
        print("Need at least two common timestamps.", file=sys.stderr)
        return 1
    se = 0.0
    sn = 0.0
    su = 0.0
    for t in common:
        r = ref[t]
        o = oth[t]
        de = o[0] - r[0]
        dn = o[1] - r[1]
        du = o[2] - r[2]
        se += de * de
        sn += dn * dn
        su += du * du
    n = float(len(common))
    rmse_e = math.sqrt(se / n)
    rmse_n = math.sqrt(sn / n)
    rmse_u = math.sqrt(su / n)
    horiz = math.sqrt((se + sn) / n)
    print(f"Compared {len(common)} samples.")
    print(f"RMSE east_m: {rmse_e:.6f}  north_m: {rmse_n:.6f}  up_m: {rmse_u:.6f}")
    print(f"Horizontal RMSE (m): {horiz:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
