#pragma once

#include "ct_fgo_sim/navigation/interval_propagation.h"
#include "ct_fgo_sim/navigation/mechanization.h"
#include "ct_fgo_sim/spline/control_point.h"
#include "ct_fgo_sim/types.h"

#include <Eigen/Geometry>

namespace ct_fgo_sim {

struct AppConfig;
struct MarginalizationFrontier;

/// Optional statistics from the last windowed Ceres solve (filled when `window_solver_stats_out` is non-null).
struct WindowSolverStats {
    int num_successful_steps = 0;
    double initial_cost = 0.0;
    double final_cost = 0.0;
    int termination_type = 0;
};

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
    AlignedVec3Array* delta_sg_nodes = nullptr;
    AlignedVec3Array* delta_sa_nodes = nullptr;
    Vector3d* lever_arm = nullptr;
    double* time_offset_s = nullptr;
    double* yaw_bias_rad = nullptr;
    Eigen::Quaterniond* q_body_imu = nullptr;
    NominalNavStates* nominal_nav = nullptr;
    IntervalPropagationCache* interval_cache = nullptr;
    /// Inclusive knot index bounds; if `window_knot_lo < 0`, build the full trajectory graph.
    int window_knot_lo = -1;
    int window_knot_hi = -1;
    /// When the left window knot is > 0, add this linearized prior on `window_knot_lo`.
    const MarginalizationFrontier* marginalization_frontier = nullptr;
    bool has_yaw_bias_step_limit = false;
    double yaw_bias_center_rad = 0.0;
    double yaw_bias_step_limit_rad = 0.0;
    /// If non-null and the graph is windowed, filled after `ceres::Solve`.
    WindowSolverStats* window_solver_stats_out = nullptr;
    /// Per-window Ceres `max_num_iterations` when >= 1; otherwise use `AppConfig::solver_max_iterations_window`.
    int sliding_solver_max_iterations_override = -1;
};

bool BuildAndSolveFactorGraph(FactorGraphSession& session);

}  // namespace ct_fgo_sim
