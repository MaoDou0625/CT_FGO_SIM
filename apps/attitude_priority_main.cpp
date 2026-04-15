// Standalone demo: same local NED conventions as `mechanization.cpp` (vel_ned = vn,ve,vd;
// q_nb body->NED; gravity along +Down as [0,0,+g] in the velocity equation; static alignment
// uses down triad + Earth rate). IMU file is increments: dtheta/dt, dvel/dt with bg/ba in
// rad/s and m/s^2; propagation uses dtheta−bg*dt and dvel−ba*dt like a minimal strapdown.

#include "ct_fgo_sim/io/text_measurement_io.h"
#include "ct_fgo_sim/navigation/earth.h"
#include "ct_fgo_sim/types.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <glog/logging.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

namespace ct_fgo_sim {
namespace {

struct AttitudePriorityConfig {
    std::filesystem::path gnss_file;
    std::filesystem::path imu_file;
    std::filesystem::path output_path;
    double start_time = 0.0;
    double end_time = 0.0;
    double align_time_s = 100.0;
    bool fix_height = true;
    bool clamp_vertical_velocity = true;
};

struct NavSample {
    double time = 0.0;
    Vector3d blh = Vector3d::Zero();
    Vector3d vel_ned = Vector3d::Zero();
    Quaterniond q_nb = Quaterniond::Identity();
};

struct StaticAlignmentResult {
    Quaterniond q_nb = Quaterniond::Identity();
    Vector3d bg0 = Vector3d::Zero();
    Vector3d ba0 = Vector3d::Zero();
};

Eigen::Matrix3d BuildNedTriad(const Vector3d& down_axis, const Vector3d& earth_rate_axis) {
    Vector3d down = down_axis.normalized();
    Vector3d east = down.cross(earth_rate_axis);
    if (east.norm() < 1.0e-12) {
        east = down.unitOrthogonal();
    } else {
        east.normalize();
    }
    Vector3d north = east.cross(down);
    if (north.norm() < 1.0e-12) {
        north = east.unitOrthogonal();
    } else {
        north.normalize();
    }

    Eigen::Matrix3d frame;
    frame.col(0) = north;
    frame.col(1) = east;
    frame.col(2) = down;
    return frame;
}

StaticAlignmentResult EstimateInitialAlignment(
    const ImuMeasurementArray& imu,
    const Vector3d& origin_blh,
    double align_time_s) {
    StaticAlignmentResult result;
    if (imu.empty()) {
        return result;
    }

    const double t_end = imu.front().time + align_time_s;
    Vector3d gyro_rate_sum = Vector3d::Zero();
    Vector3d accel_rate_sum = Vector3d::Zero();
    int count = 0;
    for (const auto& m : imu) {
        if (m.time > t_end) {
            break;
        }
        if (m.dt <= 1.0e-9) {
            continue;
        }
        gyro_rate_sum += m.dtheta / m.dt;
        accel_rate_sum += m.dvel / m.dt;
        ++count;
    }
    if (count < 10 || accel_rate_sum.norm() < 1.0e-6 || gyro_rate_sum.norm() < 1.0e-9) {
        return result;
    }

    const Vector3d gyro_mean = gyro_rate_sum / static_cast<double>(count);
    const Vector3d accel_mean = accel_rate_sum / static_cast<double>(count);

    const Vector3d down_b = -accel_mean.normalized();
    const Vector3d down_n = Vector3d::UnitZ();
    const Vector3d wie_n = Earth::Iewn(origin_blh.x());
    const Eigen::Matrix3d triad_b = BuildNedTriad(down_b, gyro_mean);
    const Eigen::Matrix3d triad_n = BuildNedTriad(down_n, wie_n);
    result.q_nb = Quaterniond(triad_n * triad_b.transpose()).normalized();

    const Eigen::Matrix3d c_nb = result.q_nb.toRotationMatrix();
    const Vector3d expected_gyro_b = c_nb.transpose() * wie_n;
    const Vector3d expected_accel_b = c_nb.transpose() * Vector3d(0.0, 0.0, -Earth::Gravity(origin_blh));
    result.bg0 = gyro_mean - expected_gyro_b;
    result.ba0 = accel_mean - expected_accel_b;
    return result;
}

Eigen::Matrix3d ExpRot(const Vector3d& rotvec) {
    const double angle = rotvec.norm();
    if (angle < 1.0e-12) {
        return Eigen::Matrix3d::Identity();
    }
    return Eigen::AngleAxisd(angle, rotvec / angle).toRotationMatrix();
}

Vector3d QuaternionToRpyDeg(const Quaterniond& q_nb) {
    const Eigen::Matrix3d r = q_nb.toRotationMatrix();
    const double roll = std::atan2(r(2, 1), r(2, 2));
    const double pitch = -std::asin(std::clamp(r(2, 0), -1.0, 1.0));
    const double yaw = std::atan2(r(1, 0), r(0, 0));
    return Vector3d(roll, pitch, yaw) * 180.0 / M_PI;
}

bool LoadConfig(const std::filesystem::path& path, AttitudePriorityConfig& config) {
    const YAML::Node cfg = YAML::LoadFile(path.string());
    const auto dir = path.parent_path();
    if (!cfg["gnssfile"] || !cfg["imu_main"]) {
        LOG(ERROR) << "Missing gnssfile or imu_main in config";
        return false;
    }

    config.gnss_file = cfg["gnssfile"].as<std::string>();
    if (config.gnss_file.is_relative()) {
        config.gnss_file = (dir / config.gnss_file).lexically_normal();
    }

    config.imu_file = cfg["imu_main"]["file"].as<std::string>();
    if (config.imu_file.is_relative()) {
        config.imu_file = (dir / config.imu_file).lexically_normal();
    }

    if (cfg["outputpath"]) {
        config.output_path = cfg["outputpath"].as<std::string>();
        if (config.output_path.is_relative()) {
            config.output_path = (dir / config.output_path).lexically_normal();
        }
    } else {
        config.output_path = (dir / "../output/attitude_priority").lexically_normal();
    }

    if (cfg["starttime"]) {
        config.start_time = cfg["starttime"].as<double>();
    }
    if (cfg["endtime"]) {
        config.end_time = cfg["endtime"].as<double>();
    }
    if (cfg["aligntime"]) {
        config.align_time_s = cfg["aligntime"].as<double>();
    }
    if (cfg["attitude_priority"]) {
        const auto node = cfg["attitude_priority"];
        if (node["fix_height"]) {
            config.fix_height = node["fix_height"].as<bool>();
        }
        if (node["clamp_vertical_velocity"]) {
            config.clamp_vertical_velocity = node["clamp_vertical_velocity"].as<bool>();
        }
    }
    return true;
}

GnssMeasurementArray TrimGnss(const GnssMeasurementArray& gnss, double start_time, double end_time) {
    GnssMeasurementArray out;
    for (const auto& m : gnss) {
        if (m.time >= start_time && m.time <= end_time) {
            out.push_back(m);
        }
    }
    return out;
}

ImuMeasurementArray TrimImu(const ImuMeasurementArray& imu, double start_time, double end_time) {
    ImuMeasurementArray out;
    for (const auto& m : imu) {
        if (m.time >= start_time && m.time <= end_time) {
            out.push_back(m);
        }
    }
    return out;
}

void SaveOutputs(
    const AttitudePriorityConfig& config,
    const Vector3d& origin_blh,
    const StaticAlignmentResult& alignment,
    const std::vector<NavSample>& nav) {
    std::filesystem::create_directories(config.output_path);

    std::ofstream nav_ofs(config.output_path / "attitude_priority_nav.txt");
    nav_ofs << "# time_s lat_rad lon_rad h_m vn_mps ve_mps vd_mps roll_deg pitch_deg yaw_deg qx qy qz qw\n";
    for (const auto& sample : nav) {
        const Vector3d rpy_deg = QuaternionToRpyDeg(sample.q_nb);
        nav_ofs << std::setprecision(17)
                << sample.time << ' '
                << sample.blh.x() << ' '
                << sample.blh.y() << ' '
                << sample.blh.z() << ' '
                << sample.vel_ned.x() << ' '
                << sample.vel_ned.y() << ' '
                << sample.vel_ned.z() << ' '
                << rpy_deg.x() << ' '
                << rpy_deg.y() << ' '
                << rpy_deg.z() << ' '
                << sample.q_nb.x() << ' '
                << sample.q_nb.y() << ' '
                << sample.q_nb.z() << ' '
                << sample.q_nb.w() << '\n';
    }

    std::ofstream summary_ofs(config.output_path / "attitude_priority_summary.txt");
    summary_ofs << std::setprecision(17);
    summary_ofs << "gnss_file: " << config.gnss_file.string() << '\n';
    summary_ofs << "imu_file: " << config.imu_file.string() << '\n';
    summary_ofs << "start_time: " << config.start_time << '\n';
    summary_ofs << "end_time: " << config.end_time << '\n';
    summary_ofs << "align_time_s: " << config.align_time_s << '\n';
    summary_ofs << "fix_height: " << config.fix_height << '\n';
    summary_ofs << "clamp_vertical_velocity: " << config.clamp_vertical_velocity << '\n';
    summary_ofs << "origin_blh_rad: " << origin_blh.transpose() << '\n';
    summary_ofs << "initial_q_nb_xyzw: "
                << alignment.q_nb.x() << ' '
                << alignment.q_nb.y() << ' '
                << alignment.q_nb.z() << ' '
                << alignment.q_nb.w() << '\n';
    summary_ofs << "initial_bg0_rps: " << alignment.bg0.transpose() << '\n';
    summary_ofs << "initial_ba0_mps2: " << alignment.ba0.transpose() << '\n';
    if (!nav.empty()) {
        const Vector3d start_rpy = QuaternionToRpyDeg(nav.front().q_nb);
        const Vector3d end_rpy = QuaternionToRpyDeg(nav.back().q_nb);
        summary_ofs << "start_rpy_deg: " << start_rpy.transpose() << '\n';
        summary_ofs << "end_rpy_deg: " << end_rpy.transpose() << '\n';
        summary_ofs << "sample_count: " << nav.size() << '\n';
    }

    std::ofstream alignment_ofs(config.output_path / "attitude_priority_alignment.txt");
    alignment_ofs << std::setprecision(17);
    alignment_ofs << "initial_q_nb_xyzw: "
                  << alignment.q_nb.x() << ' '
                  << alignment.q_nb.y() << ' '
                  << alignment.q_nb.z() << ' '
                  << alignment.q_nb.w() << '\n';
    alignment_ofs << "initial_bg0_rps: " << alignment.bg0.transpose() << '\n';
    alignment_ofs << "initial_bg0_degph: "
                  << (alignment.bg0 * (180.0 / M_PI * 3600.0)).transpose() << '\n';
    alignment_ofs << "initial_ba0_mps2: " << alignment.ba0.transpose() << '\n';
    alignment_ofs << "initial_ba0_mgal: " << (alignment.ba0 / 1.0e-5).transpose() << '\n';
}

}  // namespace
}  // namespace ct_fgo_sim

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);

    if (argc != 2) {
        std::cerr << "usage: attitude_priority_main <config.yaml>\n";
        return 1;
    }

    ct_fgo_sim::AttitudePriorityConfig config;
    if (!ct_fgo_sim::LoadConfig(argv[1], config)) {
        return 1;
    }

    auto gnss = ct_fgo_sim::io::LoadGnssFile(config.gnss_file);
    auto imu = ct_fgo_sim::io::LoadImuFile(config.imu_file, true);
    if (gnss.empty() || imu.empty()) {
        LOG(ERROR) << "No input measurements loaded";
        return 1;
    }

    if (config.start_time == 0.0 && config.end_time == 0.0) {
        config.start_time = std::max(gnss.front().time, imu.front().time);
        config.end_time = std::min(gnss.back().time, imu.back().time);
    }

    gnss = ct_fgo_sim::TrimGnss(gnss, config.start_time, config.end_time);
    imu = ct_fgo_sim::TrimImu(imu, config.start_time, config.end_time);
    if (gnss.empty() || imu.size() < 2) {
        LOG(ERROR) << "Insufficient measurements after trimming";
        return 1;
    }

    const ct_fgo_sim::Vector3d initial_blh = gnss.front().blh;
    const double fixed_height = initial_blh.z();
    ct_fgo_sim::Vector3d blh = initial_blh;
    const ct_fgo_sim::StaticAlignmentResult alignment =
        ct_fgo_sim::EstimateInitialAlignment(imu, initial_blh, config.align_time_s);
    ct_fgo_sim::Quaterniond q_nb = alignment.q_nb;
    ct_fgo_sim::Vector3d vel_ned = ct_fgo_sim::Vector3d::Zero();

    std::vector<ct_fgo_sim::NavSample> nav;
    nav.reserve(imu.size());

    nav.push_back({imu.front().time, blh, vel_ned, q_nb});

    for (size_t i = 1; i < imu.size(); ++i) {
        const auto& meas = imu[i];
        if (meas.dt <= 0.0) {
            continue;
        }

        if (config.fix_height) {
            blh.z() = fixed_height;
        }

        const ct_fgo_sim::Vector3d dtheta_corr = meas.dtheta - alignment.bg0 * meas.dt;
        const ct_fgo_sim::Vector3d dvel_corr = meas.dvel - alignment.ba0 * meas.dt;

        const ct_fgo_sim::Vector3d wie_n = ct_fgo_sim::Earth::Iewn(blh.x());
        const ct_fgo_sim::Vector3d wen_n = ct_fgo_sim::Earth::Wnen(blh, vel_ned);
        const Eigen::Matrix3d c_nb_old = q_nb.toRotationMatrix();
        const Eigen::Matrix3d c_nn = ct_fgo_sim::ExpRot(-(wie_n + wen_n) * meas.dt);
        const Eigen::Matrix3d c_bb = ct_fgo_sim::ExpRot(dtheta_corr);
        const Eigen::Matrix3d c_nb_new = c_nn * c_nb_old * c_bb;
        q_nb = ct_fgo_sim::Quaterniond(c_nb_new).normalized();

        const ct_fgo_sim::Vector3d gravity_n(0.0, 0.0, ct_fgo_sim::Earth::Gravity(blh));
        const ct_fgo_sim::Vector3d coriolis = (2.0 * wie_n + wen_n).cross(vel_ned);
        vel_ned += c_nb_old * dvel_corr + (gravity_n - coriolis) * meas.dt;
        if (config.clamp_vertical_velocity) {
            vel_ned.z() = 0.0;
        }

        nav.push_back({meas.time, blh, vel_ned, q_nb});
    }

    ct_fgo_sim::SaveOutputs(config, initial_blh, alignment, nav);
    LOG(INFO) << "Wrote attitude-priority outputs to " << config.output_path.string();
    return 0;
}
