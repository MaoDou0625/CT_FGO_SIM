#pragma once

#include <sophus/se3.hpp>

namespace ct_fgo_sim::spline {

class ControlPoint {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ControlPoint() = default;
    ControlPoint(double timestamp, const Sophus::SE3d& pose) : timestamp_(timestamp), pose_(pose) {}

    double* PoseData() { return pose_.data(); }
    const double* PoseData() const { return pose_.data(); }

    Sophus::SE3d& Pose() { return pose_; }
    const Sophus::SE3d& Pose() const { return pose_; }

    double Timestamp() const { return timestamp_; }

private:
    double timestamp_ = 0.0;
    Sophus::SE3d pose_ = Sophus::SE3d();
};

using ControlPointArray = std::vector<ControlPoint, Eigen::aligned_allocator<ControlPoint>>;

}  // namespace ct_fgo_sim::spline
