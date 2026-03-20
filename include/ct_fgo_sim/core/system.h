#pragma once

#include "ct_fgo_sim/navigation/interval_propagation.h"
#include "ct_fgo_sim/navigation/mechanization.h"
#include "ct_fgo_sim/navigation/earth.h"
#include "ct_fgo_sim/spline/control_point.h"
#include "ct_fgo_sim/types.h"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace ct_fgo_sim {

struct ImuConfig {
    std::string file;
    int columns = 7;
    double rate_hz = 0.0;
    bool values_are_increments = true;
    Vector3d antlever = Vector3d::Zero();
};

struct BodyFrameConfig {
    Eigen::Quaterniond q_body_imu = Eigen::Quaterniond::Identity();
    double q_body_imu_prior_sigma_rad = 0.1;
    bool enable_nhc = false;
    bool estimate_q_body_imu = true;
    bool nhc_enable_vx = false;
    bool nhc_enable_vy = true;
    bool nhc_enable_vz = true;
    double nhc_target_vx_mps = 0.0;
    double nhc_target_vy_mps = 0.0;
    double nhc_target_vz_mps = 0.0;
    double nhc_sigma_vx_mps = 1.0e6;
    double nhc_sigma_vy_mps = 0.2;
    double nhc_sigma_vz_mps = 0.2;
};

struct AppConfig {
    std::string gnss_file;
    std::filesystem::path output_path;
    ImuConfig imu_main;
    BodyFrameConfig body_frame;
    double spline_dt_s = 0.1;
    double start_time = 0.0;
    double end_time = 0.0;
    double align_time_s = 30.0;
    double gnss_sigma_horizontal_m = 0.03;
    double gnss_sigma_vertical_m = 0.05;
    double imu_sigma_accel_mps2 = 0.2;
    double imu_sigma_gyro_rps = 0.01;
    double gyro_bias_rw_sigma = 1.0e-4;
    double accel_bias_rw_sigma = 1.0e-3;
    double bias_tau_s = 3600.0;
    bool enable_initial_yaw_feedback = false;
    double initial_yaw_feedback_window_s = 20.0;
    double initial_yaw_feedback_min_speed_mps = 0.5;
    int initial_yaw_feedback_min_pairs = 10;
    double initial_yaw_feedback_max_abs_rad = 0.7853981633974483;
    int imu_stride = 10;
    int outer_iterations = 1;
    int solver_max_iterations = 20;
    bool use_gnss_factors = true;
    bool use_imu_factors = true;
    bool use_explicit_init_state = false;
    Vector3d init_pos_blh = Vector3d::Zero();
    Vector3d init_vel_ned = Vector3d::Zero();
    Vector3d init_att_rpy_rad = Vector3d::Zero();
    Vector3d init_bg_rps = Vector3d::Zero();
    Vector3d init_ba_mps2 = Vector3d::Zero();
};

struct ComposedState {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double time = 0.0;
    NominalNavState nominal;
    Sophus::SE3d full_pose = Sophus::SE3d();
    Vector3d full_vel_ned = Vector3d::Zero();
    Vector3d full_vel_body = Vector3d::Zero();
    Vector3d full_omega_body = Vector3d::Zero();
    Vector3d full_accel_ned = Vector3d::Zero();
    Vector3d full_alpha_body = Vector3d::Zero();
    Vector3d full_bg = Vector3d::Zero();
    Vector3d full_ba = Vector3d::Zero();
    Vector3d delta_theta = Vector3d::Zero();
    Vector3d delta_vel_ned = Vector3d::Zero();
    Vector3d delta_pos_ned = Vector3d::Zero();
    Vector3d delta_bg = Vector3d::Zero();
    Vector3d delta_ba = Vector3d::Zero();
};

class System {
public:
    bool LoadConfig(const std::filesystem::path& config_path);
    bool Run();
    void Describe() const;
    std::optional<ComposedState> EvaluateComposedState(double time) const;

private:
    bool IsPureInertialReplay() const;
    bool LoadMeasurements();
    void TrimMeasurementsToTimeWindow();
    bool InitializeControlPoints();
    bool ResetControlPointsFromNominalTrajectory(bool reset_biases);
    bool BuildAndSolveProblem();
    bool SaveOutputs() const;
    bool ApplyInitialYawFeedbackFromGnss();
    std::optional<Vector3d> EvaluateNominalGyroCenterAtTime(double time) const;
    std::optional<Vector3d> EvaluateNominalAccelAtTime(double time) const;
    std::optional<Vector3d> EvaluateNodeValueAtTime(
        double time,
        const AlignedVec3Array& nodes) const;
    std::optional<Vector3d> EvaluateNodeDerivativeAtTime(
        double time,
        const AlignedVec3Array& nodes) const;
    void UpdateNominalTrajectoryFromCurrentBiases();

    AppConfig config_;
    GnssMeasurementArray gnss_;
    ImuMeasurementArray imu_;
    spline::ControlPointArray control_points_;
    AlignedVec3Array delta_theta_nodes_;
    AlignedVec3Array delta_vel_nodes_;
    AlignedVec3Array delta_pos_nodes_;
    AlignedVec3Array delta_bg_nodes_;
    AlignedVec3Array delta_ba_nodes_;
    Vector3d lever_arm_ = Vector3d::Zero();
    Eigen::Quaterniond initial_q_body_imu_ = Eigen::Quaterniond::Identity();
    Eigen::Quaterniond q_body_imu_ = Eigen::Quaterniond::Identity();
    double time_offset_s_ = 0.0;
    Vector3d origin_blh_ = Vector3d::Zero();
    Eigen::Quaterniond initial_q_nb_ = Eigen::Quaterniond::Identity();
    StaticAlignmentResult initial_alignment_;
    NominalNavStates nominal_nav_;
    IntervalPropagationCache interval_cache_;
    bool initial_yaw_feedback_applied_ = false;
    double initial_yaw_feedback_total_rad_ = 0.0;
};

}  // namespace ct_fgo_sim
