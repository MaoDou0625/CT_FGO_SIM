#pragma once

#include "ct_fgo_sim/spline/control_point.h"

#include <algorithm>

namespace ct_fgo_sim {

/// B-spline window start index for time `t`, assuming uniformly spaced knot timestamps.
/// Uses mean spacing `(t_last - t_first) / (n - 1)` so it matches grids built by
/// `BuildKnotGridFromNominal` (uniform in time, not necessarily equal to config `spline_dt_s`).
int FindSplineWindowStart(const spline::ControlPointArray& control_points, double t);

/// Piecewise-linear node interval for spline interpolation at time `t`.
int FindNodeIntervalStart(const spline::ControlPointArray& control_points, double t);

}  // namespace ct_fgo_sim
