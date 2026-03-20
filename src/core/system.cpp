#include "ct_fgo_sim/core/system.h"

#include "ct_fgo_sim/factors/error_state_attitude_factor.h"
#include "ct_fgo_sim/factors/error_state_gnss_factor.h"
#include "ct_fgo_sim/factors/error_state_process_factor.h"
#include "ct_fgo_sim/factors/quaternion_prior_factor.h"
#include "ct_fgo_sim/io/text_measurement_io.h"

#include <ceres/ceres.h>
#include <glog/logging.h>
#include <sophus/so3.hpp>
#include <sophus/ceres_manifold.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <thread>

namespace ct_fgo_sim {

namespace {

constexpr double kDegToRad = M_PI / 180.0;

int FindSplineWindowStart(const spline::ControlPointArray& control_points, double spline_dt_s, double t) {
    if (control_points.size() < 4 || spline_dt_s <= 0.0) {
        return -1;
    }

    const double t_first = control_points.front().Timestamp();
    const int raw_index = static_cast<int>(std::floor((t - t_first) / spline_dt_s));
    return std::clamp(raw_index, 0, static_cast<int>(control_points.size()) - 4);
}

int FindNodeIntervalStart(const spline::ControlPointArray& control_points, double t) {
    if (control_points.size() < 2) {
        return -1;
    }
    if (t <= control_points.front().Timestamp()) {
        return 0;
    }
    if (t >= control_points.back().Timestamp()) {
        return static_cast<int>(control_points.size()) - 2;
    }
    const auto upper = std::lower_bound(
        control_points.begin(),
        control_points.end(),
        t,
        [](const spline::ControlPoint& control_point, double time) { return control_point.Timestamp() < time; });
    return std::max(0, static_cast<int>(std::distance(control_points.begin(), upper)) - 1);
}

Eigen::Matrix3d BuildGnssSqrtInfo(double sigma_horizontal_m, double sigma_vertical_m) {
    Eigen::Matrix3d sqrt_info = Eigen::Matrix3d::Zero();
    sqrt_info(0, 0) = 1.0 / std::max(1.0e-6, sigma_horizontal_m);
    sqrt_info(1, 1) = 1.0 / std::max(1.0e-6, sigma_horizontal_m);
    sqrt_info(2, 2) = 1.0 / std::max(1.0e-6, sigma_vertical_m);
    return sqrt_info;
}

Vector3d InterpolateNodeValue(
    double time,
    const spline::ControlPointArray& control_points,
    const AlignedVec3Array& nodes) {
    if (control_points.empty() || nodes.empty() || control_points.size() != nodes.size()) {
        return Vector3d::Zero();
    }
    if (time <= control_points.front().Timestamp()) {
        return nodes.front();
    }
    if (time >= control_points.back().Timestamp()) {
        return nodes.back();
    }

    const auto upper = std::lower_bound(
        control_points.begin(),
        control_points.end(),
        time,
        [](const spline::ControlPoint& control_point, double t) { return control_point.Timestamp() < t; });
    const size_t j = static_cast<size_t>(std::distance(control_points.begin(), upper));
    const size_t i = j - 1;
    const double dt = control_points[j].Timestamp() - control_points[i].Timestamp();
    if (dt <= 1.0e-9) {
        return nodes[i];
    }
    const double u = std::clamp((time - control_points[i].Timestamp()) / dt, 0.0, 1.0);
    return nodes[i] * (1.0 - u) + nodes[j] * u;
}

Vector3d NedToEnu(const Vector3d& ned) {
    return Earth::NedToEnu(ned);
}

Eigen::Quaterniond QnbNedToQebEnu(const Eigen::Quaterniond& q_nb_ned) {
    return Eigen::Quaterniond(Earth::RenuNed() * q_nb_ned.toRotationMatrix()).normalized();
}

Eigen::Quaterniond EulerNedToQuaternion(const Vector3d& rpy_rad) {
    return Eigen::Quaterniond(
        Eigen::AngleAxisd(rpy_rad.z(), Vector3d::UnitZ()) *
        Eigen::AngleAxisd(rpy_rad.y(), Vector3d::UnitY()) *
        Eigen::AngleAxisd(rpy_rad.x(), Vector3d::UnitX())).normalized();
}

}  // namespace

bool System::LoadConfig(const std::filesystem::path& config_path) {
    const YAML::Node cfg = YAML::LoadFile(config_path.string());
    const std::filesystem::path config_dir = config_path.parent_path();
    if (!cfg["gnssfile"] || !cfg["imu_main"]) {
        LOG(ERROR) << "Missing required nodes: gnssfile or imu_main";
        return false;
    }

    std::filesystem::path gnss_path = cfg["gnssfile"].as<std::string>();
    if (gnss_path.is_relative()) {
        gnss_path = config_dir / gnss_path;
    }
    config_.gnss_file = gnss_path.lexically_normal().string();

    if (cfg["outputpath"]) {
        std::filesystem::path output_path = cfg["outputpath"].as<std::string>();
        if (output_path.is_relative()) {
            output_path = config_dir / output_path;
        }
        config_.output_path = output_path.lexically_normal();
    } else {
        config_.output_path = (config_dir / "../output").lexically_normal();
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
    if (cfg["gnss_sigma_horizontal_m"]) {
        config_.gnss_sigma_horizontal_m = cfg["gnss_sigma_horizontal_m"].as<double>();
    }
    if (cfg["gnss_sigma_vertical_m"]) {
        config_.gnss_sigma_vertical_m = cfg["gnss_sigma_vertical_m"].as<double>();
    }
    if (cfg["imu_sigma_accel_mps2"]) {
        config_.imu_sigma_accel_mps2 = cfg["imu_sigma_accel_mps2"].as<double>();
    }
    if (cfg["imu_sigma_gyro_rps"]) {
        config_.imu_sigma_gyro_rps = cfg["imu_sigma_gyro_rps"].as<double>();
    }
    if (cfg["gyro_bias_rw_sigma"]) {
        config_.gyro_bias_rw_sigma = cfg["gyro_bias_rw_sigma"].as<double>();
    }
    if (cfg["accel_bias_rw_sigma"]) {
        config_.accel_bias_rw_sigma = cfg["accel_bias_rw_sigma"].as<double>();
    }
    if (cfg["bias_tau_s"]) {
        config_.bias_tau_s = cfg["bias_tau_s"].as<double>();
    }
    if (cfg["imu_stride"]) {
        config_.imu_stride = std::max(1, cfg["imu_stride"].as<int>());
    }
    if (cfg["outer_iterations"]) {
        config_.outer_iterations = std::max(1, cfg["outer_iterations"].as<int>());
    }
    if (cfg["solver_max_iterations"]) {
        config_.solver_max_iterations = std::max(1, cfg["solver_max_iterations"].as<int>());
    }
    if (cfg["use_gnss_factors"]) {
        config_.use_gnss_factors = cfg["use_gnss_factors"].as<bool>();
    }
    if (cfg["use_imu_factors"]) {
        config_.use_imu_factors = cfg["use_imu_factors"].as<bool>();
    }
    if (cfg["initpos"] && cfg["initvel"] && cfg["initatt"]) {
        const auto initpos = cfg["initpos"].as<std::vector<double>>();
        const auto initvel = cfg["initvel"].as<std::vector<double>>();
        const auto initatt = cfg["initatt"].as<std::vector<double>>();
        if (initpos.size() == 3 && initvel.size() == 3 && initatt.size() == 3) {
            config_.use_explicit_init_state = true;
            config_.init_pos_blh = Vector3d(initpos[0] * kDegToRad, initpos[1] * kDegToRad, initpos[2]);
            config_.init_vel_ned = Vector3d(initvel[0], initvel[1], initvel[2]);
            config_.init_att_rpy_rad = Vector3d(initatt[0] * kDegToRad, initatt[1] * kDegToRad, initatt[2] * kDegToRad);
        }
    }
    if (cfg["initgyrbias"]) {
        const auto initbg = cfg["initgyrbias"].as<std::vector<double>>();
        if (initbg.size() == 3) {
            config_.init_bg_rps = Vector3d(initbg[0], initbg[1], initbg[2]) * (kDegToRad / 3600.0);
        }
    }
    if (cfg["initaccbias"]) {
        const auto initba = cfg["initaccbias"].as<std::vector<double>>();
        if (initba.size() == 3) {
            config_.init_ba_mps2 = Vector3d(initba[0], initba[1], initba[2]) * 1.0e-5;
        }
    }
    if (cfg["body_frame"]) {
        const YAML::Node body = cfg["body_frame"];
        if (body["q_body_imu_xyzw"]) {
            const auto v = body["q_body_imu_xyzw"].as<std::vector<double>>();
            if (v.size() == 4) {
                config_.body_frame.q_body_imu = Eigen::Quaterniond(v[3], v[0], v[1], v[2]).normalized();
            }
        }
        if (body["q_body_imu_prior_sigma_rad"]) {
            config_.body_frame.q_body_imu_prior_sigma_rad = body["q_body_imu_prior_sigma_rad"].as<double>();
        }
        if (body["enable_nhc"]) {
            config_.body_frame.enable_nhc = body["enable_nhc"].as<bool>();
        }
        if (body["estimate_q_body_imu"]) {
            config_.body_frame.estimate_q_body_imu = body["estimate_q_body_imu"].as<bool>();
        }
        if (body["nhc_enable_vx"]) {
            config_.body_frame.nhc_enable_vx = body["nhc_enable_vx"].as<bool>();
        }
        if (body["nhc_enable_vy"]) {
            config_.body_frame.nhc_enable_vy = body["nhc_enable_vy"].as<bool>();
        }
        if (body["nhc_enable_vz"]) {
            config_.body_frame.nhc_enable_vz = body["nhc_enable_vz"].as<bool>();
        }
        if (body["nhc_target_vx_mps"]) {
            config_.body_frame.nhc_target_vx_mps = body["nhc_target_vx_mps"].as<double>();
        }
        if (body["nhc_target_vy_mps"]) {
            config_.body_frame.nhc_target_vy_mps = body["nhc_target_vy_mps"].as<double>();
        }
        if (body["nhc_target_vz_mps"]) {
            config_.body_frame.nhc_target_vz_mps = body["nhc_target_vz_mps"].as<double>();
        }
        if (body["nhc_sigma_vx_mps"]) {
            config_.body_frame.nhc_sigma_vx_mps = body["nhc_sigma_vx_mps"].as<double>();
        }
        if (body["nhc_sigma_vy_mps"]) {
            config_.body_frame.nhc_sigma_vy_mps = body["nhc_sigma_vy_mps"].as<double>();
        }
        if (body["nhc_sigma_vz_mps"]) {
            config_.body_frame.nhc_sigma_vz_mps = body["nhc_sigma_vz_mps"].as<double>();
        }
    }

    const YAML::Node imu = cfg["imu_main"];
    std::filesystem::path imu_path = imu["file"].as<std::string>();
    if (imu_path.is_relative()) {
        imu_path = config_dir / imu_path;
    }
    config_.imu_main.file = imu_path.lexically_normal().string();
    config_.imu_main.columns = imu["columns"] ? imu["columns"].as<int>() : 7;
    config_.imu_main.rate_hz = imu["rate_hz"] ? imu["rate_hz"].as<double>() : 0.0;
    if (imu["values_are_increments"]) {
        config_.imu_main.values_are_increments = imu["values_are_increments"].as<bool>();
    }
    if (imu["antlever"]) {
        const auto v = imu["antlever"].as<std::vector<double>>();
        if (v.size() == 3) {
            config_.imu_main.antlever = Vector3d(v[0], v[1], v[2]);
        }
    }

    lever_arm_ = config_.imu_main.antlever;
    initial_q_body_imu_ = config_.body_frame.q_body_imu;
    q_body_imu_ = initial_q_body_imu_;
    return true;
}

bool System::Run() {
    std::fprintf(stderr, "[probe] Run: LoadMeasurements\n");
    if (!LoadMeasurements()) {
        std::fprintf(stderr, "[probe] Run: LoadMeasurements failed\n");
        return false;
    }
    std::fprintf(stderr, "[probe] Run: TrimMeasurementsToTimeWindow initial\n");
    TrimMeasurementsToTimeWindow();
    if (gnss_.empty() || imu_.empty()) {
        std::fprintf(stderr, "[probe] Run: empty after initial trim gnss=%zu imu=%zu\n", gnss_.size(), imu_.size());
        LOG(ERROR) << "No measurements remain after initial time-window trimming";
        return false;
    }

    if (config_.use_explicit_init_state) {
        origin_blh_ = config_.init_pos_blh;
        initial_alignment_.window_start_time = config_.start_time;
        initial_alignment_.window_end_time = config_.start_time;
        initial_alignment_.reference_time = config_.start_time;
        initial_alignment_.vel0_ned = config_.init_vel_ned;
        initial_alignment_.q_nb = EulerNedToQuaternion(config_.init_att_rpy_rad);
        initial_alignment_.bg0 = config_.init_bg_rps;
        initial_alignment_.ba0 = config_.init_ba_mps2;
    } else {
        origin_blh_ = gnss_.front().blh;
        std::fprintf(stderr, "[probe] Run: EstimateInitialAlignment\n");
        initial_alignment_ = EstimateInitialAlignment(imu_, origin_blh_, config_.align_time_s);
        initial_q_nb_ = initial_alignment_.q_nb;
        config_.start_time = std::max(config_.start_time, initial_alignment_.reference_time);
        std::fprintf(stderr, "[probe] Run: TrimMeasurementsToTimeWindow after alignment\n");
        TrimMeasurementsToTimeWindow();
        if (gnss_.empty() || imu_.empty()) {
            std::fprintf(stderr, "[probe] Run: empty after alignment trim gnss=%zu imu=%zu\n", gnss_.size(), imu_.size());
            LOG(ERROR) << "No measurements remain after applying alignment reference time";
            return false;
        }
    }
    initial_q_nb_ = initial_alignment_.q_nb;

    if (IsPureInertialReplay()) {
        std::fprintf(stderr, "[probe] Run: pure inertial UpdateNominalTrajectoryFromCurrentBiases\n");
        LOG(INFO) << "Pure inertial replay mode: skipping spline and optimization";
        UpdateNominalTrajectoryFromCurrentBiases();
        if (nominal_nav_.empty()) {
            std::fprintf(stderr, "[probe] Run: nominal_nav empty in pure inertial\n");
            LOG(ERROR) << "Nominal mechanization propagation failed in pure inertial replay";
            return false;
        }
        std::fprintf(stderr, "[probe] Run: pure inertial SaveOutputs\n");
        return SaveOutputs();
    }

    std::fprintf(stderr, "[probe] Run: InitializeControlPoints\n");
    if (!InitializeControlPoints()) {
        std::fprintf(stderr, "[probe] Run: InitializeControlPoints failed\n");
        return false;
    }

    Describe();
    LOG(INFO) << "GNSS count: " << gnss_.size();
    LOG(INFO) << "IMU count: " << imu_.size();
    LOG(INFO) << "Control point count: " << control_points_.size();
    LOG(INFO) << "Origin BLH(rad,rad,m): " << origin_blh_.transpose();
    LOG(INFO) << "Gravity at origin: " << Earth::Gravity(origin_blh_);
    LOG(INFO) << "Earth rotation in nav frame: " << Earth::Iewn(origin_blh_.x()).transpose();
    LOG(INFO) << "Static alignment window: [" << initial_alignment_.window_start_time
              << ", " << initial_alignment_.window_end_time << "]";
    LOG(INFO) << "Static alignment reference time: " << initial_alignment_.reference_time;

    for (int outer_iter = 0; outer_iter < config_.outer_iterations; ++outer_iter) {
        LOG(INFO) << "Outer iteration " << (outer_iter + 1) << "/" << config_.outer_iterations;
        if (!BuildAndSolveProblem()) {
            return false;
        }
        if (outer_iter + 1 < config_.outer_iterations && !ResetControlPointsFromNominalTrajectory(false)) {
            LOG(ERROR) << "Failed to reset control points from updated nominal trajectory";
            return false;
        }
    }
    return SaveOutputs();
}

bool System::IsPureInertialReplay() const {
    return !config_.use_gnss_factors && !config_.use_imu_factors;
}

void System::Describe() const {
    LOG(INFO) << "CT_FGO_SIM minimal problem";
    LOG(INFO) << "GNSS file: " << config_.gnss_file;
    LOG(INFO) << "IMU file: " << config_.imu_main.file;
    LOG(INFO) << "Spline dt: " << config_.spline_dt_s;
    LOG(INFO) << "Time window: [" << config_.start_time << ", " << config_.end_time << "]";
    LOG(INFO) << "GNSS sigma(h/v): " << config_.gnss_sigma_horizontal_m << ", " << config_.gnss_sigma_vertical_m;
    LOG(INFO) << "IMU sigma(a/g): " << config_.imu_sigma_accel_mps2 << ", " << config_.imu_sigma_gyro_rps;
    LOG(INFO) << "IMU stride: " << config_.imu_stride;
    LOG(INFO) << "Outer iterations: " << config_.outer_iterations;
    LOG(INFO) << "Use GNSS factors: " << (config_.use_gnss_factors ? "true" : "false");
    LOG(INFO) << "Use IMU factors: " << (config_.use_imu_factors ? "true" : "false");
    LOG(INFO) << "Enable body NHC: " << (config_.body_frame.enable_nhc ? "true" : "false");
    LOG(INFO) << "Estimate q_body_imu: "
              << ((config_.body_frame.enable_nhc && config_.body_frame.estimate_q_body_imu) ? "true" : "false");
    LOG(INFO) << "NHC axes enabled (vx, vy, vz): "
              << config_.body_frame.nhc_enable_vx << ", "
              << config_.body_frame.nhc_enable_vy << ", "
              << config_.body_frame.nhc_enable_vz;
}

bool System::LoadMeasurements() {
    gnss_ = io::LoadGnssFile(config_.gnss_file);
    imu_ = io::LoadImuFile(config_.imu_main.file, config_.imu_main.values_are_increments);
    if (gnss_.empty()) {
        LOG(ERROR) << "No GNSS measurements loaded from " << config_.gnss_file;
        return false;
    }
    if (imu_.empty()) {
        LOG(ERROR) << "No IMU measurements loaded from " << config_.imu_main.file;
        return false;
    }
    if (config_.start_time == 0.0 && config_.end_time == 0.0) {
        config_.start_time = std::max(gnss_.front().time, imu_.front().time);
        config_.end_time = std::min(gnss_.back().time, imu_.back().time);
    }
    return true;
}

void System::TrimMeasurementsToTimeWindow() {
    auto in_window = [this](double t) { return t >= config_.start_time && t <= config_.end_time; };
    gnss_.erase(
        std::remove_if(gnss_.begin(), gnss_.end(), [&](const GnssMeasurement& m) { return !in_window(m.time); }),
        gnss_.end());
    imu_.erase(
        std::remove_if(imu_.begin(), imu_.end(), [&](const ImuMeasurement& m) { return !in_window(m.time); }),
        imu_.end());
}

bool System::InitializeControlPoints() {
    if (gnss_.empty()) {
        LOG(ERROR) << "Cannot initialize control points without GNSS";
        return false;
    }
    if (imu_.empty()) {
        LOG(ERROR) << "Cannot initialize control points without IMU";
        return false;
    }

    UpdateNominalTrajectoryFromCurrentBiases();
    if (nominal_nav_.empty()) {
        LOG(ERROR) << "Nominal mechanization propagation failed";
        return false;
    }

    return ResetControlPointsFromNominalTrajectory(true);
}

bool System::ResetControlPointsFromNominalTrajectory(bool reset_biases) {
    if (nominal_nav_.empty()) {
        LOG(ERROR) << "Cannot reset control points from an empty nominal trajectory";
        return false;
    }

    spline::SplineInitializer::Path path;
    path.reserve(nominal_nav_.size());
    for (const auto& nav : nominal_nav_) {
        const Vector3d local_ned = Earth::GlobalToLocal(origin_blh_, nav.blh);
        path.emplace_back(nav.time, Sophus::SE3d(nav.q_nb, local_ned));
    }

    spline::ControlPointArray new_control_points =
        spline::SplineInitializer::InitializeFromPath(path, config_.spline_dt_s);
    if (new_control_points.empty()) {
        LOG(ERROR) << "Spline initialization from nominal trajectory produced no control points";
        return false;
    }

    AlignedVec3Array new_delta_theta(new_control_points.size(), Vector3d::Zero());
    AlignedVec3Array new_delta_vel(new_control_points.size(), Vector3d::Zero());
    AlignedVec3Array new_delta_pos(new_control_points.size(), Vector3d::Zero());
    AlignedVec3Array new_delta_bg(new_control_points.size(), Vector3d::Zero());
    AlignedVec3Array new_delta_ba(new_control_points.size(), Vector3d::Zero());
    if (!reset_biases) {
        if (delta_theta_nodes_.size() == new_control_points.size() &&
            delta_vel_nodes_.size() == new_control_points.size() &&
            delta_pos_nodes_.size() == new_control_points.size() &&
            delta_bg_nodes_.size() == new_control_points.size() &&
            delta_ba_nodes_.size() == new_control_points.size()) {
            new_delta_theta = delta_theta_nodes_;
            new_delta_vel = delta_vel_nodes_;
            new_delta_pos = delta_pos_nodes_;
            new_delta_bg = delta_bg_nodes_;
            new_delta_ba = delta_ba_nodes_;
        } else {
            LOG(WARNING) << "Delta-state node count changed from " << delta_theta_nodes_.size()
                         << " to " << new_control_points.size()
                         << "; resetting warm start";
        }
    }

    control_points_ = std::move(new_control_points);
    delta_theta_nodes_ = std::move(new_delta_theta);
    delta_vel_nodes_ = std::move(new_delta_vel);
    delta_pos_nodes_ = std::move(new_delta_pos);
    delta_bg_nodes_ = std::move(new_delta_bg);
    delta_ba_nodes_ = std::move(new_delta_ba);
    return !control_points_.empty();
}

bool System::BuildAndSolveProblem() {
    if (control_points_.size() < 2) {
        LOG(ERROR) << "Need at least 2 control points to build the problem";
        return false;
    }

    ceres::Problem problem;
    for (auto& delta_theta : delta_theta_nodes_) {
        problem.AddParameterBlock(delta_theta.data(), 3);
    }
    for (auto& delta_vel : delta_vel_nodes_) {
        problem.AddParameterBlock(delta_vel.data(), 3);
    }
    for (auto& delta_pos : delta_pos_nodes_) {
        problem.AddParameterBlock(delta_pos.data(), 3);
    }
    for (auto& delta_bg : delta_bg_nodes_) {
        problem.AddParameterBlock(delta_bg.data(), 3);
    }
    for (auto& delta_ba : delta_ba_nodes_) {
        problem.AddParameterBlock(delta_ba.data(), 3);
    }
    problem.AddParameterBlock(&time_offset_s_, 1);
    problem.AddParameterBlock(q_body_imu_.coeffs().data(), 4, new ceres::EigenQuaternionManifold);
    problem.SetParameterBlockConstant(delta_theta_nodes_.front().data());
    problem.SetParameterBlockConstant(delta_vel_nodes_.front().data());
    problem.SetParameterBlockConstant(delta_pos_nodes_.front().data());
    problem.SetParameterBlockConstant(delta_bg_nodes_.front().data());
    problem.SetParameterBlockConstant(delta_ba_nodes_.front().data());
    problem.SetParameterBlockConstant(&time_offset_s_);
    problem.SetParameterBlockConstant(q_body_imu_.coeffs().data());

    problem.AddResidualBlock(
        factors::QuaternionPriorFactor::Create(
            config_.body_frame.q_body_imu,
            config_.body_frame.q_body_imu_prior_sigma_rad),
        nullptr,
        q_body_imu_.coeffs().data());

    const Eigen::Matrix3d gnss_sqrt_info =
        BuildGnssSqrtInfo(config_.gnss_sigma_horizontal_m, config_.gnss_sigma_vertical_m);
    int gnss_factor_count = 0;
    if (config_.use_gnss_factors) {
        for (const auto& gnss : gnss_) {
            const int start = FindNodeIntervalStart(control_points_, gnss.time);
            if (start < 0 || start + 1 >= static_cast<int>(control_points_.size())) {
                continue;
            }
            const auto nominal_state = EvaluateNominalState(nominal_nav_, gnss.time);
            if (!nominal_state) {
                continue;
            }
            const double dt = control_points_[start + 1].Timestamp() - control_points_[start].Timestamp();
            if (dt <= 1.0e-9) {
                continue;
            }
            const double u = std::clamp((gnss.time - control_points_[start].Timestamp()) / dt, 0.0, 1.0);
            const Vector3d nominal_pos_ned = Earth::GlobalToLocal(origin_blh_, nominal_state->blh);
            const Vector3d local_ned = Earth::GlobalToLocal(origin_blh_, gnss.blh);
            ceres::CostFunction* cost = factors::ErrorStateGnssFactor::Create(
                u, nominal_pos_ned, local_ned, gnss_sqrt_info);
            problem.AddResidualBlock(
                cost,
                nullptr,
                delta_pos_nodes_[start].data(),
                delta_pos_nodes_[start + 1].data());
            ++gnss_factor_count;
        }
    }

    int inertial_factor_count = 0;
    int process_factor_count = 0;
    if (config_.use_imu_factors) {
        for (int imu_index = 0; imu_index < static_cast<int>(imu_.size()); imu_index += config_.imu_stride) {
            const auto& imu_meas = imu_[imu_index];
            if (imu_meas.dt <= 1.0e-9) {
                continue;
            }
            const int start = FindNodeIntervalStart(control_points_, imu_meas.time);
            if (start < 0 || start + 1 >= static_cast<int>(control_points_.size())) {
                continue;
            }
            const auto nominal_gyro_center = EvaluateNominalGyroCenterAtTime(imu_meas.time);
            const auto nominal_state = EvaluateNominalState(nominal_nav_, imu_meas.time);
            if (!nominal_state || !nominal_gyro_center) {
                continue;
            }
            const double dt = control_points_[start + 1].Timestamp() - control_points_[start].Timestamp();
            if (dt <= 1.0e-9) {
                continue;
            }
            const double u = std::clamp((imu_meas.time - control_points_[start].Timestamp()) / dt, 0.0, 1.0);
            const Vector3d gyro_meas_rps = imu_meas.dtheta / imu_meas.dt;
            ceres::CostFunction* cost = factors::ErrorStateAttitudeFactor::Create(
                u,
                dt,
                gyro_meas_rps,
                *nominal_gyro_center,
                nominal_state->bg,
                config_.imu_sigma_gyro_rps);
            problem.AddResidualBlock(
                cost,
                new ceres::HuberLoss(1.0),
                delta_theta_nodes_[start].data(),
                delta_theta_nodes_[start + 1].data(),
                delta_bg_nodes_[start].data(),
                delta_bg_nodes_[start + 1].data(),
                &time_offset_s_);
            ++inertial_factor_count;
        }

        for (int i = 0; i + 1 < static_cast<int>(control_points_.size()); ++i) {
            const double dt = control_points_[i + 1].Timestamp() - control_points_[i].Timestamp();
            if (dt <= 1.0e-9) {
                continue;
            }
            const auto nominal_accel = EvaluateNominalAccelAtTime(control_points_[i].Timestamp());
            const auto nominal_state = EvaluateNominalState(nominal_nav_, control_points_[i].Timestamp());
            if (!nominal_accel || !nominal_state) {
                continue;
            }
            const Vector3d omega_ie_n = Earth::Iewn(nominal_state->blh.x());
            const Vector3d omega_en_n = Earth::Wnen(nominal_state->blh, nominal_state->vel_ned);
            const Vector3d gravity_n(0.0, 0.0, Earth::Gravity(nominal_state->blh));
            const Vector3d specific_force_n =
                *nominal_accel - gravity_n + (2.0 * omega_ie_n + omega_en_n).cross(nominal_state->vel_ned);
            const Vector3d specific_force_b =
                nominal_state->q_nb.toRotationMatrix().transpose() * specific_force_n;
            problem.AddResidualBlock(
                factors::ErrorStateProcessFactor::Create(
                    dt,
                    nominal_state->blh,
                    nominal_state->vel_ned,
                    nominal_state->q_nb,
                    specific_force_b,
                    config_.imu_sigma_gyro_rps,
                    config_.imu_sigma_accel_mps2,
                    config_.gyro_bias_rw_sigma,
                    config_.accel_bias_rw_sigma,
                    config_.bias_tau_s),
                nullptr,
                delta_theta_nodes_[i].data(),
                delta_vel_nodes_[i].data(),
                delta_pos_nodes_[i].data(),
                delta_bg_nodes_[i].data(),
                delta_ba_nodes_[i].data(),
                delta_theta_nodes_[i + 1].data(),
                delta_vel_nodes_[i + 1].data(),
                delta_pos_nodes_[i + 1].data(),
                delta_bg_nodes_[i + 1].data(),
                delta_ba_nodes_[i + 1].data());
            ++process_factor_count;
        }
    }

    ceres::Solver::Options options;
    options.max_num_iterations = config_.solver_max_iterations;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.num_threads = std::max(1u, std::thread::hardware_concurrency());
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    LOG(INFO) << "GNSS factors: " << gnss_factor_count;
    LOG(INFO) << "Attitude factors: " << inertial_factor_count;
    LOG(INFO) << "Process factors: " << process_factor_count;
    LOG(INFO) << summary.BriefReport();
    UpdateNominalTrajectoryFromCurrentBiases();
    return summary.termination_type != ceres::FAILURE;
}

std::optional<Vector3d> System::EvaluateNodeValueAtTime(
    double time,
    const AlignedVec3Array& nodes) const {
    if (control_points_.empty() || nodes.empty() || control_points_.size() != nodes.size()) {
        return std::nullopt;
    }
    return InterpolateNodeValue(time, control_points_, nodes);
}

std::optional<Vector3d> System::EvaluateNodeDerivativeAtTime(
    double time,
    const AlignedVec3Array& nodes) const {
    if (control_points_.size() < 2 || nodes.size() != control_points_.size()) {
        return std::nullopt;
    }
    const int start = FindNodeIntervalStart(control_points_, time);
    if (start < 0 || start + 1 >= static_cast<int>(control_points_.size())) {
        return std::nullopt;
    }
    const double dt = control_points_[start + 1].Timestamp() - control_points_[start].Timestamp();
    if (dt <= 1.0e-9) {
        return std::nullopt;
    }
    return (nodes[start + 1] - nodes[start]) / dt;
}

std::optional<Vector3d> System::EvaluateNominalGyroCenterAtTime(double time) const {
    if (nominal_nav_.size() < 2) {
        return std::nullopt;
    }

    const auto upper = std::lower_bound(
        nominal_nav_.begin(),
        nominal_nav_.end(),
        time,
        [](const NominalNavState& state, double t) { return state.time < t; });
    size_t i = 0;
    size_t j = 1;
    if (upper == nominal_nav_.begin()) {
        i = 0;
        j = 1;
    } else if (upper == nominal_nav_.end()) {
        i = nominal_nav_.size() - 2;
        j = nominal_nav_.size() - 1;
    } else {
        j = static_cast<size_t>(std::distance(nominal_nav_.begin(), upper));
        i = j - 1;
    }

    const double dt = nominal_nav_[j].time - nominal_nav_[i].time;
    if (dt <= 1.0e-9) {
        return std::nullopt;
    }

    const Eigen::Quaterniond q_i = nominal_nav_[i].q_nb;
    const Eigen::Quaterniond q_j = nominal_nav_[j].q_nb;
    const Eigen::Quaterniond q_delta = (q_i.conjugate() * q_j).normalized();
    const Eigen::AngleAxisd aa(q_delta);
    Vector3d rotvec_body = Vector3d::Zero();
    if (std::abs(aa.angle()) > 1.0e-12) {
        rotvec_body = aa.axis() * aa.angle();
    }
    const Vector3d nominal_w_body = rotvec_body / dt;

    const auto nominal_state = EvaluateNominalState(nominal_nav_, time);
    if (!nominal_state) {
        return std::nullopt;
    }

    const Vector3d omega_ie_n = Earth::Iewn(nominal_state->blh.x());
    const Vector3d omega_en_n = Earth::Wnen(nominal_state->blh, nominal_state->vel_ned);
    const Vector3d omega_in_b = nominal_state->q_nb.toRotationMatrix().transpose() * (omega_ie_n + omega_en_n);
    return nominal_w_body + omega_in_b;
}

std::optional<Vector3d> System::EvaluateNominalAccelAtTime(double time) const {
    if (nominal_nav_.size() < 2) {
        return std::nullopt;
    }

    const auto upper = std::lower_bound(
        nominal_nav_.begin(),
        nominal_nav_.end(),
        time,
        [](const NominalNavState& state, double t) { return state.time < t; });
    size_t i = 0;
    size_t j = 1;
    if (upper == nominal_nav_.begin()) {
        i = 0;
        j = 1;
    } else if (upper == nominal_nav_.end()) {
        i = nominal_nav_.size() - 2;
        j = nominal_nav_.size() - 1;
    } else {
        j = static_cast<size_t>(std::distance(nominal_nav_.begin(), upper));
        i = j - 1;
    }

    const double dt = nominal_nav_[j].time - nominal_nav_[i].time;
    if (dt <= 1.0e-9) {
        return std::nullopt;
    }
    return (nominal_nav_[j].vel_ned - nominal_nav_[i].vel_ned) / dt;
}

std::optional<ComposedState> System::EvaluateComposedState(double time) const {
    const auto nominal_state = EvaluateNominalState(nominal_nav_, time);
    if (!nominal_state) {
        return std::nullopt;
    }

    if (control_points_.empty()) {
        ComposedState composed;
        composed.time = time;
        composed.nominal = *nominal_state;
        const Vector3d nominal_local_ned = Earth::GlobalToLocal(origin_blh_, nominal_state->blh);
        composed.full_pose = Sophus::SE3d(Sophus::SO3d(nominal_state->q_nb), nominal_local_ned);
        composed.full_vel_ned = nominal_state->vel_ned;
        composed.full_vel_body = nominal_state->q_nb.toRotationMatrix().transpose() * nominal_state->vel_ned;
        composed.full_bg = nominal_state->bg;
        composed.full_ba = nominal_state->ba;
        if (const auto nominal_gyro = EvaluateNominalGyroCenterAtTime(time)) {
            composed.full_omega_body = *nominal_gyro + nominal_state->bg;
        }
        if (const auto nominal_accel = EvaluateNominalAccelAtTime(time)) {
            composed.full_accel_ned = *nominal_accel;
        }
        return composed;
    }

    const auto delta_theta = EvaluateNodeValueAtTime(time, delta_theta_nodes_);
    const auto delta_vel = EvaluateNodeValueAtTime(time, delta_vel_nodes_);
    const auto delta_pos = EvaluateNodeValueAtTime(time, delta_pos_nodes_);
    const auto delta_bg = EvaluateNodeValueAtTime(time, delta_bg_nodes_);
    const auto delta_ba = EvaluateNodeValueAtTime(time, delta_ba_nodes_);
    const auto delta_theta_dot = EvaluateNodeDerivativeAtTime(time, delta_theta_nodes_);
    const auto nominal_accel = EvaluateNominalAccelAtTime(time);
    const auto nominal_gyro = EvaluateNominalGyroCenterAtTime(time);
    if (!delta_theta || !delta_vel || !delta_pos || !delta_bg || !delta_ba ||
        !delta_theta_dot || !nominal_accel || !nominal_gyro) {
        return std::nullopt;
    }

    ComposedState composed;
    composed.time = time;
    composed.nominal = *nominal_state;
    const Vector3d nominal_local_ned = Earth::GlobalToLocal(origin_blh_, nominal_state->blh);
    const Sophus::SO3d nominal_rot(nominal_state->q_nb);
    const Sophus::SO3d full_rot = nominal_rot * Sophus::SO3d::exp(*delta_theta);
    composed.delta_theta = *delta_theta;
    composed.delta_vel_ned = *delta_vel;
    composed.delta_pos_ned = *delta_pos;
    composed.delta_bg = *delta_bg;
    composed.delta_ba = *delta_ba;
    composed.full_pose = Sophus::SE3d(full_rot, nominal_local_ned + *delta_pos);
    composed.full_vel_ned = nominal_state->vel_ned + *delta_vel;
    composed.full_vel_body = full_rot.inverse() * composed.full_vel_ned;
    composed.full_omega_body = *nominal_gyro + *delta_theta_dot + nominal_state->bg + *delta_bg;
    composed.full_accel_ned = *nominal_accel;
    composed.full_alpha_body = Vector3d::Zero();
    composed.full_bg = nominal_state->bg + *delta_bg;
    composed.full_ba = nominal_state->ba + *delta_ba;
    return composed;
}

void System::UpdateNominalTrajectoryFromCurrentBiases() {
    std::vector<double> bias_times;
    bias_times.reserve(control_points_.size());
    AlignedVec3Array full_bg_nodes;
    AlignedVec3Array full_ba_nodes;
    full_bg_nodes.reserve(control_points_.size());
    full_ba_nodes.reserve(control_points_.size());
    for (const auto& control_point : control_points_) {
        bias_times.push_back(control_point.Timestamp());
    }

    double max_delta_bg_norm = 0.0;
    double max_delta_ba_norm = 0.0;
    if (control_points_.size() == delta_bg_nodes_.size() &&
        control_points_.size() == delta_ba_nodes_.size()) {
        for (size_t i = 0; i < control_points_.size(); ++i) {
            full_bg_nodes.push_back(initial_alignment_.bg0 + delta_bg_nodes_[i]);
            full_ba_nodes.push_back(initial_alignment_.ba0 + delta_ba_nodes_[i]);
            max_delta_bg_norm = std::max(max_delta_bg_norm, delta_bg_nodes_[i].norm());
            max_delta_ba_norm = std::max(max_delta_ba_norm, delta_ba_nodes_[i].norm());
        }
    } else {
        for (size_t i = 0; i < control_points_.size(); ++i) {
            full_bg_nodes.push_back(initial_alignment_.bg0);
            full_ba_nodes.push_back(initial_alignment_.ba0);
        }
    }

    nominal_nav_ = PropagateNominalTrajectory(
        imu_,
        origin_blh_,
        initial_alignment_,
        bias_times,
        full_bg_nodes,
        full_ba_nodes);

    if (control_points_.size() == delta_bg_nodes_.size() &&
        control_points_.size() == delta_ba_nodes_.size()) {
        LOG(INFO) << "Closed-loop bias feedback injected into nominal mechanization, max |delta_bg|="
                  << max_delta_bg_norm << " rad/s, max |delta_ba|=" << max_delta_ba_norm << " m/s^2";
        for (auto& delta_bg : delta_bg_nodes_) {
            delta_bg.setZero();
        }
        for (auto& delta_ba : delta_ba_nodes_) {
            delta_ba.setZero();
        }
    }
}

bool System::SaveOutputs() const {
    std::filesystem::create_directories(config_.output_path);

    const std::filesystem::path trajectory_path = config_.output_path / "trajectory_enu.txt";
    std::ofstream trajectory_ofs(trajectory_path);
    trajectory_ofs << "# time_s east_m north_m up_m qx qy qz qw\n";
    for (const auto& gnss : gnss_) {
        const auto composed = EvaluateComposedState(gnss.time);
        if (!composed) {
            continue;
        }
        const Eigen::Quaterniond q = QnbNedToQebEnu(Eigen::Quaterniond(composed->full_pose.so3().matrix()));
        const Vector3d t = NedToEnu(composed->full_pose.translation());
        trajectory_ofs << gnss.time << ' ' << t.x() << ' ' << t.y() << ' ' << t.z() << ' '
                       << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w() << '\n';
    }

    const std::filesystem::path bias_path = config_.output_path / "bias_nodes.txt";
    std::ofstream bias_ofs(bias_path);
    bias_ofs << "# time_s d_bgx d_bgy d_bgz d_bax d_bay d_baz\n";
    for (size_t i = 0; i < control_points_.size(); ++i) {
        bias_ofs << control_points_[i].Timestamp() << ' '
                 << delta_bg_nodes_[i].x() << ' ' << delta_bg_nodes_[i].y() << ' ' << delta_bg_nodes_[i].z() << ' '
                 << delta_ba_nodes_[i].x() << ' ' << delta_ba_nodes_[i].y() << ' ' << delta_ba_nodes_[i].z() << '\n';
    }

    const std::filesystem::path summary_path = config_.output_path / "run_summary.txt";
    std::ofstream summary_ofs(summary_path);
    summary_ofs << std::setprecision(17);
    summary_ofs << "gnss_file: " << config_.gnss_file << '\n';
    summary_ofs << "imu_file: " << config_.imu_main.file << '\n';
    summary_ofs << "use_gnss_factors: " << config_.use_gnss_factors << '\n';
    summary_ofs << "use_imu_factors: " << config_.use_imu_factors << '\n';
    summary_ofs << "gnss_count: " << gnss_.size() << '\n';
    summary_ofs << "imu_count: " << imu_.size() << '\n';
    summary_ofs << "control_point_count: " << control_points_.size() << '\n';
    summary_ofs << "outer_iterations: " << config_.outer_iterations << '\n';
    summary_ofs << "time_offset_s: " << time_offset_s_ << '\n';
    summary_ofs << "lever_arm_m: "
                << lever_arm_.x() << ' '
                << lever_arm_.y() << ' '
                << lever_arm_.z() << '\n';
    summary_ofs << "q_body_imu_xyzw: "
                << q_body_imu_.x() << ' '
                << q_body_imu_.y() << ' '
                << q_body_imu_.z() << ' '
                << q_body_imu_.w() << '\n';
    const Eigen::Quaterniond q_body_imu_delta = initial_q_body_imu_.conjugate() * q_body_imu_;
    const double q_body_imu_delta_angle_rad =
        2.0 * std::atan2(q_body_imu_delta.vec().norm(), std::abs(q_body_imu_delta.w()));
    summary_ofs << "initial_q_body_imu_xyzw: "
                << initial_q_body_imu_.x() << ' '
                << initial_q_body_imu_.y() << ' '
                << initial_q_body_imu_.z() << ' '
                << initial_q_body_imu_.w() << '\n';
    summary_ofs << "q_body_imu_delta_angle_rad: " << q_body_imu_delta_angle_rad << '\n';
    summary_ofs << "nhc_enable_vx: " << config_.body_frame.nhc_enable_vx << '\n';
    summary_ofs << "nhc_enable_vy: " << config_.body_frame.nhc_enable_vy << '\n';
    summary_ofs << "nhc_enable_vz: " << config_.body_frame.nhc_enable_vz << '\n';
    summary_ofs << "estimate_q_body_imu: " << config_.body_frame.estimate_q_body_imu << '\n';
    summary_ofs << "origin_blh_rad: "
                << origin_blh_.x() << ' '
                << origin_blh_.y() << ' '
                << origin_blh_.z() << '\n';
    summary_ofs << "initial_q_nb_xyzw: "
                << initial_q_nb_.x() << ' '
                << initial_q_nb_.y() << ' '
                << initial_q_nb_.z() << ' '
                << initial_q_nb_.w() << '\n';
    summary_ofs << "alignment_window_start_time_s: " << initial_alignment_.window_start_time << '\n';
    summary_ofs << "alignment_window_end_time_s: " << initial_alignment_.window_end_time << '\n';
    summary_ofs << "alignment_reference_time_s: " << initial_alignment_.reference_time << '\n';
    summary_ofs << "initial_bg0_rps: " << initial_alignment_.bg0.transpose() << '\n';
    summary_ofs << "initial_ba0_mps2: " << initial_alignment_.ba0.transpose() << '\n';

    const std::filesystem::path nominal_path = config_.output_path / "nominal_nav.txt";
    std::ofstream nominal_ofs(nominal_path);
    nominal_ofs << "# time_s lat_rad lon_rad h_m ve_mps vn_mps vu_mps qx qy qz qw bgx bgy bgz bax bay baz\n";
    for (const auto& nav : nominal_nav_) {
        const Vector3d vel_enu = NedToEnu(nav.vel_ned);
        const Eigen::Quaterniond q_enu = QnbNedToQebEnu(nav.q_nb);
        nominal_ofs << std::setprecision(17)
                    << nav.time << ' '
                    << nav.blh.x() << ' '
                    << nav.blh.y() << ' '
                    << nav.blh.z() << ' '
                    << vel_enu.x() << ' '
                    << vel_enu.y() << ' '
                    << vel_enu.z() << ' '
                    << q_enu.x() << ' '
                    << q_enu.y() << ' '
                    << q_enu.z() << ' '
                    << q_enu.w() << ' '
                    << nav.bg.x() << ' '
                    << nav.bg.y() << ' '
                    << nav.bg.z() << ' '
                    << nav.ba.x() << ' '
                    << nav.ba.y() << ' '
                    << nav.ba.z() << '\n';
    }

    const std::filesystem::path delta_path = config_.output_path / "delta_estimates.txt";
    std::ofstream delta_ofs(delta_path);
    delta_ofs << "# time_s dtheta_x_rad dtheta_y_rad dtheta_z_rad "
                 "dvx_mps dvy_mps dvz_mps dpx_m dpy_m dpz_m dbg_x_rps dbg_y_rps dbg_z_rps dba_x dba_y dba_z\n";
    for (int imu_index = 0; imu_index < static_cast<int>(imu_.size()); imu_index += config_.imu_stride) {
        const auto composed = EvaluateComposedState(imu_[imu_index].time);
        if (!composed) {
            continue;
        }
        const Vector3d delta_vel_enu = NedToEnu(composed->delta_vel_ned);
        const Vector3d delta_pos_enu = NedToEnu(composed->delta_pos_ned);
        delta_ofs << std::setprecision(17)
                  << composed->time << ' '
                  << composed->delta_theta.x() << ' '
                  << composed->delta_theta.y() << ' '
                  << composed->delta_theta.z() << ' '
                  << delta_vel_enu.x() << ' '
                  << delta_vel_enu.y() << ' '
                  << delta_vel_enu.z() << ' '
                  << delta_pos_enu.x() << ' '
                  << delta_pos_enu.y() << ' '
                  << delta_pos_enu.z() << ' '
                  << composed->delta_bg.x() << ' '
                  << composed->delta_bg.y() << ' '
                  << composed->delta_bg.z() << ' '
                  << composed->delta_ba.x() << ' '
                  << composed->delta_ba.y() << ' '
                  << composed->delta_ba.z() << '\n';
    }

    LOG(INFO) << "Wrote outputs to " << config_.output_path.string();
    return trajectory_ofs.good() && bias_ofs.good() && summary_ofs.good() && nominal_ofs.good() && delta_ofs.good();
}

}  // namespace ct_fgo_sim
