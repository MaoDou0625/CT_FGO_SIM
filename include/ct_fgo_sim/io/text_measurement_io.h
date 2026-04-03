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

inline GnssMeasurementArray LoadGnssFile(const std::filesystem::path& path) {
    std::ifstream ifs(path);
    GnssMeasurementArray rows;
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

inline ImuMeasurementArray LoadImuFile(const std::filesystem::path& path, bool values_are_increments = true) {
    std::ifstream ifs(path);
    ImuMeasurementArray rows;
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
        // Default format follows KF-GINS:
        // time_s dtheta_x_rad dtheta_y_rad dtheta_z_rad dvel_x_mps dvel_y_mps dvel_z_mps
        // When values_are_increments=false, the six IMU columns are treated as rates/specific force
        // and converted to increments using the current sample dt.
        meas.dtheta = Vector3d(values[1], values[2], values[3]);
        meas.dvel = Vector3d(values[4], values[5], values[6]);
        if (!values_are_increments) {
            meas.dtheta *= meas.dt;
            meas.dvel *= meas.dt;
        }
        rows.push_back(meas);
        prev_time = meas.time;
        first = false;
    }
    return rows;
}

inline NhcMeasurementArray LoadNhcFile(const std::filesystem::path& path) {
    std::ifstream ifs(path);
    NhcMeasurementArray rows;
    std::string line;
    while (std::getline(ifs, line)) {
        const auto values = ParseNumericRow(line);
        if (values.size() < 3) {
            continue;
        }
        NhcMeasurement meas;
        meas.time = values[0];
        if (values.size() == 3) {
            // time_s vy_mps vz_mps
            meas.vel_body_mps = Vector3d(0.0, values[1], values[2]);
        } else if (values.size() >= 4) {
            // time_s vx_mps vy_mps vz_mps
            meas.vel_body_mps = Vector3d(values[1], values[2], values[3]);
        } else {
            meas.vel_body_mps = Vector3d::Zero();
        }
        rows.push_back(meas);
    }
    return rows;
}

}  // namespace ct_fgo_sim::io
