#include "ct_fgo_sim/core/system.h"

#include "ct_fgo_sim/factors/bias_random_walk_factor.h"
#include "ct_fgo_sim/factors/continuous_gnss_factor.h"
#include "ct_fgo_sim/factors/continuous_inertial_factor.h"
#include "ct_fgo_sim/io/text_measurement_io.h"

#include <ceres/ceres.h>
#include <glog/logging.h>
#include <sophus/ceres_manifold.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <thread>

namespace ct_fgo_sim {

namespace {

int FindSplineWindowStart(const std::vector<spline::ControlPoint>& control_points, double spline_dt_s, double t) {
    if (control_points.size() < 4 || spline_dt_s <= 0.0) {
        return -1;
    }

    const double t_first = control_points.front().Timestamp();
    const int raw_index = static_cast<int>(std::floor((t - t_first) / spline_dt_s));
    return std::clamp(raw_index, 0, static_cast<int>(control_points.size()) - 4);
}

Eigen::Matrix3d BuildGnssSqrtInfo(double sigma_horizontal_m, double sigma_vertical_m) {
    Eigen::Matrix3d sqrt_info = Eigen::Matrix3d::Zero();
    sqrt_info(0, 0) = 1.0 / std::max(1.0e-6, sigma_horizontal_m);
    sqrt_info(1, 1) = 1.0 / std::max(1.0e-6, sigma_horizontal_m);
    sqrt_info(2, 2) = 1.0 / std::max(1.0e-6, sigma_vertical_m);
    return sqrt_info;
}

Eigen::Matrix3d BuildTriadFrame(const Vector3d& primary, const Vector3d& secondary) {
    const Vector3d t1 = primary.normalized();
    Vector3d t2 = t1.cross(secondary);
    if (t2.norm() < 1.0e-12) {
        t2 = t1.unitOrthogonal();
    } else {
        t2.normalize();
    }
    const Vector3d t3 = t1.cross(t2);
    Eigen::Matrix3d frame;
    frame.col(0) = t1;
    frame.col(1) = t2;
    frame.col(2) = t3;
    return frame;
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
    if (cfg["solver_max_iterations"]) {
        config_.solver_max_iterations = std::max(1, cfg["solver_max_iterations"].as<int>());
    }
    if (cfg["use_imu_factors"]) {
        config_.use_imu_factors = cfg["use_imu_factors"].as<bool>();
    }

    const YAML::Node imu = cfg["imu_main"];
    std::filesystem::path imu_path = imu["file"].as<std::string>();
    if (imu_path.is_relative()) {
        imu_path = config_dir / imu_path;
    }
    config_.imu_main.file = imu_path.lexically_normal().string();
    config_.imu_main.columns = imu["columns"] ? imu["columns"].as<int>() : 7;
    config_.imu_main.rate_hz = imu["rate_hz"] ? imu["rate_hz"].as<double>() : 0.0;
    if (imu["antlever"]) {
        const auto v = imu["antlever"].as<std::vector<double>>();
        if (v.size() == 3) {
            config_.imu_main.antlever = Vector3d(v[0], v[1], v[2]);
        }
    }

    lever_arm_ = config_.imu_main.antlever;
    return true;
}

bool System::Run() {
    if (!LoadMeasurements()) {
        return false;
    }
    TrimMeasurementsToTimeWindow();
    if (!InitializeControlPoints()) {
        return false;
    }

    Describe();
    LOG(INFO) << "GNSS count: " << gnss_.size();
    LOG(INFO) << "IMU count: " << imu_.size();
    LOG(INFO) << "Control point count: " << control_points_.size();
    LOG(INFO) << "Origin BLH(rad,rad,m): " << origin_blh_.transpose();
    LOG(INFO) << "Gravity at origin: " << Earth::Gravity(origin_blh_);
    LOG(INFO) << "Earth rotation in nav frame: " << Earth::Iewn(origin_blh_.x()).transpose();

    if (!BuildAndSolveProblem()) {
        return false;
    }
    return SaveOutputs();
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
    LOG(INFO) << "Use IMU factors: " << (config_.use_imu_factors ? "true" : "false");
}

bool System::LoadMeasurements() {
    gnss_ = io::LoadGnssFile(config_.gnss_file);
    imu_ = io::LoadImuFile(config_.imu_main.file);
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

    origin_blh_ = gnss_.front().blh;
    initial_q_nb_ = EstimateInitialAttitude();
    std::vector<std::pair<double, Sophus::SE3d>> path;
    path.reserve(gnss_.size());
    for (const auto& gnss : gnss_) {
        const Vector3d local_enu = Earth::GlobalToLocal(origin_blh_, gnss.blh);
        path.emplace_back(gnss.time, Sophus::SE3d(initial_q_nb_, local_enu));
    }
    control_points_ = spline::SplineInitializer::InitializeFromPath(path, config_.spline_dt_s);
    gyro_biases_.assign(control_points_.size(), Vector3d::Zero());
    accel_biases_.assign(control_points_.size(), Vector3d::Zero());
    return !control_points_.empty();
}

Eigen::Quaterniond System::EstimateInitialAttitude() const {
    if (imu_.empty()) {
        return Eigen::Quaterniond::Identity();
    }

    const double t_end = std::min(config_.end_time, imu_.front().time + config_.align_time_s);
    Vector3d gyro_mean = Vector3d::Zero();
    Vector3d accel_mean = Vector3d::Zero();
    int count = 0;
    for (const auto& imu_meas : imu_) {
        if (imu_meas.time > t_end) {
            break;
        }
        gyro_mean += imu_meas.dtheta;
        accel_mean += imu_meas.dvel;
        ++count;
    }
    if (count < 10 || accel_mean.norm() < 1.0e-6 || gyro_mean.norm() < 1.0e-9) {
        return Eigen::Quaterniond::Identity();
    }

    gyro_mean /= static_cast<double>(count);
    accel_mean /= static_cast<double>(count);

    const Vector3d up_b = accel_mean.normalized();
    const Vector3d up_n = Vector3d::UnitZ();
    const Vector3d wie_n = Earth::Iewn(origin_blh_.x());
    const Eigen::Matrix3d triad_b = BuildTriadFrame(up_b, gyro_mean.normalized());
    const Eigen::Matrix3d triad_n = BuildTriadFrame(up_n, wie_n.normalized());
    const Eigen::Matrix3d r_nb = triad_n * triad_b.transpose();
    return Eigen::Quaterniond(r_nb);
}

bool System::BuildAndSolveProblem() {
    if (control_points_.size() < 4) {
        LOG(ERROR) << "Need at least 4 control points to build the problem";
        return false;
    }

    ceres::Problem problem;
    for (auto& control_point : control_points_) {
        problem.AddParameterBlock(control_point.PoseData(), 7, new Sophus::Manifold<Sophus::SE3>());
    }
    for (auto& bg : gyro_biases_) {
        problem.AddParameterBlock(bg.data(), 3);
    }
    for (auto& ba : accel_biases_) {
        problem.AddParameterBlock(ba.data(), 3);
    }
    problem.AddParameterBlock(lever_arm_.data(), 3);
    problem.AddParameterBlock(&time_offset_s_, 1);
    problem.SetParameterBlockConstant(control_points_.front().PoseData());
    problem.SetParameterBlockConstant(gyro_biases_.front().data());
    problem.SetParameterBlockConstant(accel_biases_.front().data());
    problem.SetParameterBlockConstant(lever_arm_.data());
    problem.SetParameterBlockConstant(&time_offset_s_);

    const Eigen::Matrix3d gnss_sqrt_info =
        BuildGnssSqrtInfo(config_.gnss_sigma_horizontal_m, config_.gnss_sigma_vertical_m);
    int gnss_factor_count = 0;
    for (const auto& gnss : gnss_) {
        const int start = FindSplineWindowStart(control_points_, config_.spline_dt_s, gnss.time);
        if (start < 0) {
            continue;
        }
        const Vector3d local_enu = Earth::GlobalToLocal(origin_blh_, gnss.blh);
        ceres::CostFunction* cost = factors::ContinuousGnssFactor::Create(
            gnss.time,
            config_.spline_dt_s,
            control_points_[start].Timestamp(),
            local_enu,
            gnss_sqrt_info);
        problem.AddResidualBlock(
            cost,
            nullptr,
            control_points_[start + 0].PoseData(),
            control_points_[start + 1].PoseData(),
            control_points_[start + 2].PoseData(),
            control_points_[start + 3].PoseData(),
            lever_arm_.data());
        ++gnss_factor_count;
    }

    int inertial_factor_count = 0;
    int bias_factor_count = 0;
    if (config_.use_imu_factors) {
        for (int imu_index = 0; imu_index < static_cast<int>(imu_.size()); imu_index += config_.imu_stride) {
            const auto& imu_meas = imu_[imu_index];
            const int start = FindSplineWindowStart(control_points_, config_.spline_dt_s, imu_meas.time);
            if (start < 0 || start + 2 >= static_cast<int>(gyro_biases_.size())) {
                continue;
            }
            ceres::CostFunction* cost = factors::ContinuousInertialFactor::Create(
                imu_meas.time,
                imu_meas.dvel,
                imu_meas.dtheta,
                origin_blh_,
                config_.spline_dt_s,
                control_points_[start].Timestamp(),
                config_.imu_sigma_accel_mps2,
                config_.imu_sigma_gyro_rps);
            problem.AddResidualBlock(
                cost,
                new ceres::HuberLoss(1.0),
                control_points_[start + 0].PoseData(),
                control_points_[start + 1].PoseData(),
                control_points_[start + 2].PoseData(),
                control_points_[start + 3].PoseData(),
                gyro_biases_[start + 1].data(),
                gyro_biases_[start + 2].data(),
                accel_biases_[start + 1].data(),
                accel_biases_[start + 2].data(),
                lever_arm_.data(),
                &time_offset_s_);
            ++inertial_factor_count;
        }

        for (int i = 0; i + 1 < static_cast<int>(gyro_biases_.size()); ++i) {
            const double dt = control_points_[i + 1].Timestamp() - control_points_[i].Timestamp();
            problem.AddResidualBlock(
                factors::BiasRandomWalkFactor::Create(dt, config_.gyro_bias_rw_sigma, config_.bias_tau_s),
                nullptr,
                gyro_biases_[i].data(),
                gyro_biases_[i + 1].data());
            problem.AddResidualBlock(
                factors::BiasRandomWalkFactor::Create(dt, config_.accel_bias_rw_sigma, config_.bias_tau_s),
                nullptr,
                accel_biases_[i].data(),
                accel_biases_[i + 1].data());
            bias_factor_count += 2;
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
    LOG(INFO) << "Inertial factors: " << inertial_factor_count;
    LOG(INFO) << "Bias RW factors: " << bias_factor_count;
    LOG(INFO) << summary.BriefReport();
    return summary.termination_type != ceres::FAILURE;
}

bool System::SaveOutputs() const {
    std::filesystem::create_directories(config_.output_path);

    const std::filesystem::path trajectory_path = config_.output_path / "trajectory_enu.txt";
    std::ofstream trajectory_ofs(trajectory_path);
    trajectory_ofs << "# time_s east_m north_m up_m qx qy qz qw\n";
    for (const auto& gnss : gnss_) {
        const int start = FindSplineWindowStart(control_points_, config_.spline_dt_s, gnss.time);
        if (start < 0) {
            continue;
        }
        const double u = (gnss.time - control_points_[start].Timestamp()) / config_.spline_dt_s;
        const auto result = spline::BSplineEvaluator::Evaluate(
            u,
            config_.spline_dt_s,
            control_points_[start + 0].Pose(),
            control_points_[start + 1].Pose(),
            control_points_[start + 2].Pose(),
            control_points_[start + 3].Pose());
        const Eigen::Quaterniond q(result.pose.so3().matrix());
        const Vector3d t = result.pose.translation();
        trajectory_ofs << gnss.time << ' ' << t.x() << ' ' << t.y() << ' ' << t.z() << ' '
                       << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w() << '\n';
    }

    const std::filesystem::path bias_path = config_.output_path / "bias_nodes.txt";
    std::ofstream bias_ofs(bias_path);
    bias_ofs << "# time_s bgx bgy bgz bax bay baz\n";
    for (size_t i = 0; i < control_points_.size(); ++i) {
        bias_ofs << control_points_[i].Timestamp() << ' '
                 << gyro_biases_[i].x() << ' ' << gyro_biases_[i].y() << ' ' << gyro_biases_[i].z() << ' '
                 << accel_biases_[i].x() << ' ' << accel_biases_[i].y() << ' ' << accel_biases_[i].z() << '\n';
    }

    const std::filesystem::path summary_path = config_.output_path / "run_summary.txt";
    std::ofstream summary_ofs(summary_path);
    summary_ofs << std::setprecision(17);
    summary_ofs << "gnss_file: " << config_.gnss_file << '\n';
    summary_ofs << "imu_file: " << config_.imu_main.file << '\n';
    summary_ofs << "gnss_count: " << gnss_.size() << '\n';
    summary_ofs << "imu_count: " << imu_.size() << '\n';
    summary_ofs << "control_point_count: " << control_points_.size() << '\n';
    summary_ofs << "time_offset_s: " << time_offset_s_ << '\n';
    summary_ofs << "lever_arm_m: "
                << lever_arm_.x() << ' '
                << lever_arm_.y() << ' '
                << lever_arm_.z() << '\n';
    summary_ofs << "origin_blh_rad: "
                << origin_blh_.x() << ' '
                << origin_blh_.y() << ' '
                << origin_blh_.z() << '\n';
    summary_ofs << "initial_q_nb_xyzw: "
                << initial_q_nb_.x() << ' '
                << initial_q_nb_.y() << ' '
                << initial_q_nb_.z() << ' '
                << initial_q_nb_.w() << '\n';

    LOG(INFO) << "Wrote outputs to " << config_.output_path.string();
    return trajectory_ofs.good() && bias_ofs.good() && summary_ofs.good();
}

}  // namespace ct_fgo_sim
