#pragma once

#include "ct_fgo_sim/types.h"

#include <optional>
#include <vector>

namespace ct_fgo_sim {

struct StaticAlignmentResult {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double window_start_time = 0.0;
    double window_end_time = 0.0;
    double reference_time = 0.0;
    Vector3d vel0_ned = Vector3d::Zero();
    Quaterniond q_nb = Quaterniond::Identity();
    Vector3d bg0 = Vector3d::Zero();
    Vector3d ba0 = Vector3d::Zero();
};

struct NominalNavState {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double time = 0.0;
    Vector3d blh = Vector3d::Zero();
    Vector3d vel_ned = Vector3d::Zero();
    Quaterniond q_nb = Quaterniond::Identity();
    Vector3d bg = Vector3d::Zero();
    Vector3d ba = Vector3d::Zero();
};

using NominalNavStates = std::vector<NominalNavState, Eigen::aligned_allocator<NominalNavState>>;

StaticAlignmentResult EstimateInitialAlignment(
    const ImuMeasurementArray& imu,
    const Vector3d& origin_blh,
    double align_time_s);

NominalNavStates PropagateNominalTrajectory(
    const ImuMeasurementArray& imu,
    const Vector3d& initial_blh,
    const StaticAlignmentResult& alignment,
    const std::vector<double>& bias_times,
    const AlignedVec3Array& gyro_biases,
    const AlignedVec3Array& accel_biases);

std::optional<NominalNavState> EvaluateNominalState(
    const NominalNavStates& states,
    double time);

}  // namespace ct_fgo_sim
