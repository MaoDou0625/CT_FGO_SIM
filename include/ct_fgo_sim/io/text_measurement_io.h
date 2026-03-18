#pragma once

#include "ct_fgo_sim/types.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace ct_fgo_sim::io {

inline std::vector<double> ParseNumericRow(const std::string& line) {
    std::string normalized = line;
    for (char& c : normalized) {
        if (c == ',') {
            c = ' ';
        }
    }

    std::stringstream ss(normalized);
    std::vector<double> values;
    double value = 0.0;
    while (ss >> value) {
        values.push_back(value);
    }
    return values;
}

inline std::vector<GnssMeasurement> LoadGnssFile(const std::filesystem::path& path) {
    std::ifstream ifs(path);
    std::vector<GnssMeasurement> rows;
    std::string line;
    while (std::getline(ifs, line)) {
        const auto values = ParseNumericRow(line);
        if (values.size() < 4) {
            continue;
        }
        GnssMeasurement meas;
        meas.time = values[0];
        // Expected format:
        // time_s lat_rad lon_rad h_m [sigma_e_m sigma_n_m sigma_u_m]
        meas.blh = Vector3d(values[1], values[2], values[3]);
        if (values.size() >= 7) {
            meas.std = Vector3d(values[4], values[5], values[6]);
        }
        rows.push_back(meas);
    }
    return rows;
}

inline std::vector<ImuMeasurement> LoadImuFile(const std::filesystem::path& path) {
    std::ifstream ifs(path);
    std::vector<ImuMeasurement> rows;
    std::string line;
    double prev_time = 0.0;
    bool first = true;
    while (std::getline(ifs, line)) {
        const auto values = ParseNumericRow(line);
        if (values.size() < 7) {
            continue;
        }
        ImuMeasurement meas;
        meas.time = values[0];
        meas.dt = first ? 0.0 : meas.time - prev_time;
        // Expected format:
        // time_s gyro_x_radps gyro_y_radps gyro_z_radps accel_x_mps2 accel_y_mps2 accel_z_mps2
        meas.dtheta = Vector3d(values[1], values[2], values[3]);
        meas.dvel = Vector3d(values[4], values[5], values[6]);
        rows.push_back(meas);
        prev_time = meas.time;
        first = false;
    }
    return rows;
}

}  // namespace ct_fgo_sim::io
