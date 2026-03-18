#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace ct_fgo_sim {

using Eigen::Matrix3d;
using Eigen::Quaterniond;
using Eigen::Vector3d;

struct GnssMeasurement {
    double time = 0.0;
    Vector3d blh = Vector3d::Zero();
    Vector3d std = Vector3d::Zero();
};

struct ImuMeasurement {
    double time = 0.0;
    double dt = 0.0;
    Vector3d dtheta = Vector3d::Zero();
    Vector3d dvel = Vector3d::Zero();
};

struct Pose {
    Matrix3d R = Matrix3d::Identity();
    Vector3d t = Vector3d::Zero();
};

}  // namespace ct_fgo_sim
