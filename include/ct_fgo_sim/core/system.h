#pragma once

#include "ct_fgo_sim/navigation/mechanization.h"
#include "ct_fgo_sim/navigation/earth.h"
#include "ct_fgo_sim/spline/spline_initializer.h"
#include "ct_fgo_sim/spline/control_point.h"
#include "ct_fgo_sim/types.h"

#include <filesystem>
#include <string>
#include <vector>

namespace ct_fgo_sim {

struct ImuConfig {
    std::string file;
    int columns = 7;
    double rate_hz = 0.0;
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
    int imu_stride = 10;
    int solver_max_iterations = 20;
    bool use_gnss_factors = true;
    bool use_imu_factors = true;
};

class System {
public:
    bool LoadConfig(const std::filesystem::path& config_path);
    bool Run();
    void Describe() const;

private:
    bool LoadMeasurements();
    void TrimMeasurementsToTimeWindow();
    bool InitializeControlPoints();
    bool BuildAndSolveProblem();
    bool SaveOutputs() const;
    void UpdateNominalTrajectoryFromCurrentBiases();

    AppConfig config_;
    std::vector<GnssMeasurement> gnss_;
    std::vector<ImuMeasurement> imu_;
    std::vector<spline::ControlPoint> control_points_;
    std::vector<Vector3d> gyro_biases_;
    std::vector<Vector3d> accel_biases_;
    Vector3d lever_arm_ = Vector3d::Zero();
    Eigen::Quaterniond initial_q_body_imu_ = Eigen::Quaterniond::Identity();
    Eigen::Quaterniond q_body_imu_ = Eigen::Quaterniond::Identity();
    double time_offset_s_ = 0.0;
    Vector3d origin_blh_ = Vector3d::Zero();
    Eigen::Quaterniond initial_q_nb_ = Eigen::Quaterniond::Identity();
    StaticAlignmentResult initial_alignment_;
    NominalNavStates nominal_nav_;
};

}  // namespace ct_fgo_sim
