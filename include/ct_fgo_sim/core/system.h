#pragma once

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

struct AppConfig {
    std::string gnss_file;
    std::filesystem::path output_path;
    ImuConfig imu_main;
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
    Eigen::Quaterniond EstimateInitialAttitude() const;
    bool InitializeControlPoints();
    bool BuildAndSolveProblem();
    bool SaveOutputs() const;

    AppConfig config_;
    std::vector<GnssMeasurement> gnss_;
    std::vector<ImuMeasurement> imu_;
    std::vector<spline::ControlPoint> control_points_;
    std::vector<Vector3d> gyro_biases_;
    std::vector<Vector3d> accel_biases_;
    Vector3d lever_arm_ = Vector3d::Zero();
    double time_offset_s_ = 0.0;
    Vector3d origin_blh_ = Vector3d::Zero();
    Eigen::Quaterniond initial_q_nb_ = Eigen::Quaterniond::Identity();
};

}  // namespace ct_fgo_sim
