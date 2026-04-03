#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>

namespace ct_fgo_sim {

using Eigen::Matrix3d;
using Eigen::Quaterniond;
using Eigen::Vector3d;
using AlignedVec3Array = std::vector<Vector3d, Eigen::aligned_allocator<Vector3d>>;

struct GnssMeasurement {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double time = 0.0;
    Vector3d blh = Vector3d::Zero();
    Vector3d std = Vector3d::Zero();
};

struct ImuMeasurement {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double time = 0.0;
    double dt = 0.0;
    Vector3d dtheta = Vector3d::Zero();
    Vector3d dvel = Vector3d::Zero();
};

struct NhcMeasurement {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double time = 0.0;
    Vector3d vel_body_mps = Vector3d::Zero();
};

struct Pose {
    Matrix3d R = Matrix3d::Identity();
    Vector3d t = Vector3d::Zero();
};

using GnssMeasurementArray = std::vector<GnssMeasurement, Eigen::aligned_allocator<GnssMeasurement>>;
using ImuMeasurementArray = std::vector<ImuMeasurement, Eigen::aligned_allocator<ImuMeasurement>>;
using NhcMeasurementArray = std::vector<NhcMeasurement, Eigen::aligned_allocator<NhcMeasurement>>;

}  // namespace ct_fgo_sim
