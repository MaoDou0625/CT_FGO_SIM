#pragma once

#include "ct_fgo_sim/types.h"

#include <vector>

namespace ct_fgo_sim {

struct StaticAlignmentResult {
    Quaterniond q_nb = Quaterniond::Identity();
    Vector3d bg0 = Vector3d::Zero();
    Vector3d ba0 = Vector3d::Zero();
};

struct NominalNavState {
    double time = 0.0;
    Vector3d blh = Vector3d::Zero();
    Vector3d vel_enu = Vector3d::Zero();
    Quaterniond q_nb = Quaterniond::Identity();
    Vector3d bg = Vector3d::Zero();
    Vector3d ba = Vector3d::Zero();
};

StaticAlignmentResult EstimateInitialAlignment(
    const std::vector<ImuMeasurement>& imu,
    const Vector3d& origin_blh,
    double align_time_s);

std::vector<NominalNavState> PropagateNominalTrajectory(
    const std::vector<ImuMeasurement>& imu,
    const Vector3d& initial_blh,
    const StaticAlignmentResult& alignment,
    const std::vector<double>& bias_times,
    const std::vector<Vector3d>& gyro_biases,
    const std::vector<Vector3d>& accel_biases);

}  // namespace ct_fgo_sim
