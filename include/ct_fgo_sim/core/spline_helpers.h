#pragma once

#include "ct_fgo_sim/spline/control_point.h"

#include <algorithm>

namespace ct_fgo_sim {

/// Piecewise-linear node interval for error-state interpolation at time `t`.
int FindNodeIntervalStart(const spline::ControlPointArray& control_points, double t);

}  // namespace ct_fgo_sim
