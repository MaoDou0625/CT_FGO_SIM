#pragma once

#include "ct_fgo_sim/spline/control_point.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace ct_fgo_sim::spline {

class SplineInitializer {
public:
    using PathPoint = std::pair<double, Sophus::SE3d>;
    using Path = std::vector<PathPoint, Eigen::aligned_allocator<PathPoint>>;

    static ControlPointArray InitializeFromPath(
        const Path& path,
        double dt) {
        if (path.empty() || dt <= 0.0) {
            return {};
        }

        const double t_min = path.front().first;
        const double t_max = path.back().first;
        const double t_start = t_min - dt;
        const int count = static_cast<int>(std::ceil((t_max - t_min) / dt)) + 5;

        ControlPointArray cps;
        cps.reserve(static_cast<size_t>(count));
        for (int i = 0; i < count; ++i) {
            const double t = t_start + i * dt;
            cps.emplace_back(t, Interpolate(path, t));
        }
        return cps;
    }

private:
    static Sophus::SE3d Interpolate(
        const Path& path,
        double t) {
        if (t <= path.front().first) {
            return path.front().second;
        }
        if (t >= path.back().first) {
            return path.back().second;
        }
        auto it = std::lower_bound(
            path.begin(),
            path.end(),
            t,
            [](const PathPoint& lhs, double rhs) { return lhs.first < rhs; });
        const auto& p1 = *(it - 1);
        const auto& p2 = *it;
        const double a = (t - p1.first) / (p2.first - p1.first);
        return p1.second * Sophus::SE3d::exp((p1.second.inverse() * p2.second).log() * a);
    }
};

}  // namespace ct_fgo_sim::spline
