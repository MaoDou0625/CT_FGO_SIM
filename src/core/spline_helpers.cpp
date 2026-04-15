#include "ct_fgo_sim/core/spline_helpers.h"

namespace ct_fgo_sim {

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
