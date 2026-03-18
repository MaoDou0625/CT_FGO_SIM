#pragma once

#include "ct_fgo_sim/navigation/earth.h"
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
};

class System {
public:
    bool LoadConfig(const std::filesystem::path& config_path);
    void Describe() const;

private:
    AppConfig config_;
};

}  // namespace ct_fgo_sim
