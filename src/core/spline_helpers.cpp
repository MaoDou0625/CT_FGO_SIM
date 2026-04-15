#include "ct_fgo_sim/core/spline_helpers.h"

#include <cmath>

namespace ct_fgo_sim {

int FindSplineWindowStart(const spline::ControlPointArray& control_points, double t) {
    if (control_points.size() < 4) {
        return -1;
    }

    const double t_first = control_points.front().Timestamp();
    const double t_last = control_points.back().Timestamp();
    const double span = t_last - t_first;
    if (span <= 0.0) {
        return -1;
    }

    const double dt_uniform = span / static_cast<double>(control_points.size() - 1);
    if (dt_uniform <= 0.0) {
        return -1;
    }

    const int raw_index = static_cast<int>(std::floor((t - t_first) / dt_uniform + 1.0e-12));
    return std::clamp(raw_index, 0, static_cast<int>(control_points.size()) - 4);
}

int FindNodeIntervalStart(const spline::ControlPointArray& control_points, double t) {
    if (control_points.size() < 2) {
        return -1;
    }
    if (t <= control_points.front().Timestamp()) {
        return 0;
    }
    if (t >= control_points.back().Timestamp()) {
        return static_cast<int>(control_points.size()) - 2;
    }
    const auto upper = std::lower_bound(
        control_points.begin(),
        control_points.end(),
        t,
        [](const spline::ControlPoint& control_point, double time) { return control_point.Timestamp() < time; });
    return std::max(0, static_cast<int>(std::distance(control_points.begin(), upper)) - 1);
}

}  // namespace ct_fgo_sim
