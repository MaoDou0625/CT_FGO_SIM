#include "ct_fgo_sim/core/system.h"

#include <glog/logging.h>
#include <yaml-cpp/yaml.h>

namespace ct_fgo_sim {

bool System::LoadConfig(const std::filesystem::path& config_path) {
    const YAML::Node cfg = YAML::LoadFile(config_path.string());
    if (!cfg["gnssfile"] || !cfg["imu_main"]) {
        LOG(ERROR) << "Missing required nodes: gnssfile or imu_main";
        return false;
    }

    config_.gnss_file = cfg["gnssfile"].as<std::string>();
    if (cfg["outputpath"]) {
        config_.output_path = cfg["outputpath"].as<std::string>();
    }
    if (cfg["kf_interval_sec"]) {
        config_.spline_dt_s = cfg["kf_interval_sec"].as<double>();
    }
    if (cfg["starttime"]) {
        config_.start_time = cfg["starttime"].as<double>();
    }
    if (cfg["endtime"]) {
        config_.end_time = cfg["endtime"].as<double>();
    }
    if (cfg["aligntime"]) {
        config_.align_time_s = cfg["aligntime"].as<double>();
    }

    const YAML::Node imu = cfg["imu_main"];
    config_.imu_main.file = imu["file"].as<std::string>();
    config_.imu_main.columns = imu["columns"] ? imu["columns"].as<int>() : 7;
    config_.imu_main.rate_hz = imu["rate_hz"] ? imu["rate_hz"].as<double>() : 0.0;
    if (imu["antlever"]) {
        const auto v = imu["antlever"].as<std::vector<double>>();
        if (v.size() == 3) {
            config_.imu_main.antlever = Vector3d(v[0], v[1], v[2]);
        }
    }

    return true;
}

void System::Describe() const {
    LOG(INFO) << "CT_FGO_SIM skeleton loaded";
    LOG(INFO) << "GNSS file: " << config_.gnss_file;
    LOG(INFO) << "IMU file: " << config_.imu_main.file;
    LOG(INFO) << "IMU rate: " << config_.imu_main.rate_hz;
    LOG(INFO) << "Spline dt: " << config_.spline_dt_s;
    LOG(INFO) << "Time window: [" << config_.start_time << ", " << config_.end_time << "]";
}

}  // namespace ct_fgo_sim
