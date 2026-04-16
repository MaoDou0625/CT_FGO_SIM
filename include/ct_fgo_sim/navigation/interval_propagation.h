#pragma once

#include "ct_fgo_sim/navigation/mechanization.h"
#include "ct_fgo_sim/spline/control_point.h"

#include <Eigen/Core>

#include <optional>
#include <vector>

namespace ct_fgo_sim {

struct NominalImuInterval {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double start_time = 0.0;
    double end_time = 0.0;
    double mid_time = 0.0;
    double dt = 0.0;
    size_t imu_index = 0;
    Vector3d omega_ib_b_nom = Vector3d::Zero();
    Vector3d accel_n_mid = Vector3d::Zero();
};

using NominalImuIntervals = std::vector<NominalImuInterval>;

struct KnotIntervalPropagation {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double start_time = 0.0;
    double end_time = 0.0;
    size_t begin_imu_index = 0;
    size_t end_imu_index = 0;
    bool valid = false;
    MatrixErrorState phi = MatrixErrorState::Identity();
    MatrixErrorState q = MatrixErrorState::Zero();
    MatrixErrorState sqrt_info = MatrixErrorState::Identity();
};

using KnotIntervalPropagations = std::vector<KnotIntervalPropagation>;

struct IntervalPropagationCache {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    NominalImuIntervals imu_intervals;
    KnotIntervalPropagations knot_intervals;
};

void BuildIntervalPropagationCache(
    const ImuMeasurementArray& imu,
    const NominalNavStates& nominal_states,
    const spline::ControlPointArray& control_points,
    double sigma_gyro_rps,
    double sigma_accel_mps2,
    double sigma_bg_std,
    double sigma_ba_std,
    double sigma_sg_std,
    double sigma_sa_std,
    double bias_tau_s,
    IntervalPropagationCache& cache);

/// Incrementally extends `cache` for appended nominal states/control points.
/// Falls back to full rebuild when input/cache prefixes are inconsistent.
void AppendIntervalPropagationCache(
    const ImuMeasurementArray& imu,
    const NominalNavStates& nominal_states,
    const spline::ControlPointArray& control_points,
    double sigma_gyro_rps,
    double sigma_accel_mps2,
    double sigma_bg_std,
    double sigma_ba_std,
    double sigma_sg_std,
    double sigma_sa_std,
    double bias_tau_s,
    IntervalPropagationCache& cache);

std::optional<Vector3d> EvaluateNominalGyroCenterAtTime(
    const IntervalPropagationCache& cache,
    double time);

std::optional<Vector3d> EvaluateNominalAccelAtTime(
    const IntervalPropagationCache& cache,
    double time);

}  // namespace ct_fgo_sim
