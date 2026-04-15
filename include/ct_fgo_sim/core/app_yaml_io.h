#pragma once

#include "ct_fgo_sim/types.h"

#include <Eigen/Geometry>
#include <filesystem>

namespace ct_fgo_sim {

struct AppConfig;

/// IMU lever arm and body–IMU rotation read with the YAML application config.
struct ImuExtrinsicDefaults {
    Vector3d lever_arm = Vector3d::Zero();
    Eigen::Quaterniond q_body_imu = Eigen::Quaterniond::Identity();
};

bool LoadAppConfigYaml(
    const std::filesystem::path& config_path,
    AppConfig& config,
    ImuExtrinsicDefaults& imu_extrinsics);

bool LoadMeasurementBundle(
    AppConfig& config,
    GnssMeasurementArray& gnss,
    ImuMeasurementArray& imu,
    NhcMeasurementArray& nhc,
    bool fill_default_time_window_if_zero);

void TrimNavMeasurementsToConfigWindow(
    const AppConfig& config,
    GnssMeasurementArray& gnss,
    ImuMeasurementArray& imu,
    NhcMeasurementArray& nhc);

}  // namespace ct_fgo_sim
