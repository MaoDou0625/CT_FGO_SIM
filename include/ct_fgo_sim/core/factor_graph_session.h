#pragma once

#include "ct_fgo_sim/navigation/interval_propagation.h"
#include "ct_fgo_sim/navigation/mechanization.h"
#include "ct_fgo_sim/spline/control_point.h"
#include "ct_fgo_sim/types.h"

#include <Eigen/Geometry>

namespace ct_fgo_sim {

struct AppConfig;

/// Pointers to `System` state used to assemble and solve the Ceres factor graph.
struct FactorGraphSession {
    AppConfig* config = nullptr;
    Vector3d* origin_blh = nullptr;
    GnssMeasurementArray* gnss = nullptr;
    ImuMeasurementArray* imu = nullptr;
    NhcMeasurementArray* nhc = nullptr;
    spline::ControlPointArray* control_points = nullptr;
    AlignedVec3Array* delta_theta_nodes = nullptr;
    AlignedVec3Array* delta_vel_nodes = nullptr;
    AlignedVec3Array* delta_pos_nodes = nullptr;
    AlignedVec3Array* delta_bg_nodes = nullptr;
    AlignedVec3Array* delta_ba_nodes = nullptr;
    Vector3d* lever_arm = nullptr;
    double* time_offset_s = nullptr;
    Eigen::Quaterniond* q_body_imu = nullptr;
    NominalNavStates* nominal_nav = nullptr;
    IntervalPropagationCache* interval_cache = nullptr;
};

bool BuildAndSolveFactorGraph(FactorGraphSession& session);

}  // namespace ct_fgo_sim
