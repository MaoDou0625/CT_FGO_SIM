#include "ct_fgo_sim/core/system.h"

#include "ct_fgo_sim/core/app_yaml_io.h"
#include "ct_fgo_sim/core/spline_helpers.h"

#include <glog/logging.h>
#include <sophus/so3.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <limits>

#include <ceres/types.h>

namespace ct_fgo_sim {

namespace {

constexpr double kDegToRad = M_PI / 180.0;

double WrapAngleRad(double angle_rad) {
    while (angle_rad > M_PI) {
        angle_rad -= 2.0 * M_PI;
    }
    while (angle_rad < -M_PI) {
        angle_rad += 2.0 * M_PI;
    }
    return angle_rad;
}

double YawFromQuaternionNed(const Eigen::Quaterniond& q_nb) {
    const Eigen::Matrix3d rot = q_nb.toRotationMatrix();
    return std::atan2(rot(1, 0), rot(0, 0));
}

struct YawFeedbackSample {
    double time = 0.0;
    double speed_mps = 0.0;
    double yaw_error_rad = 0.0;
};

double CircularMeanRad(const std::vector<YawFeedbackSample>& samples, size_t begin, size_t end) {
    double weighted_sin = 0.0;
    double weighted_cos = 0.0;
    for (size_t i = begin; i < end; ++i) {
        const double weight = std::clamp(samples[i].speed_mps, 0.5, 5.0);
        weighted_sin += weight * std::sin(samples[i].yaw_error_rad);
        weighted_cos += weight * std::cos(samples[i].yaw_error_rad);
    }
    return std::atan2(weighted_sin, weighted_cos);
}

std::vector<double> AlignAnglesAroundSeed(
    const std::vector<YawFeedbackSample>& samples,
    size_t begin,
    size_t end,
    double seed_rad) {
    std::vector<double> aligned;
    aligned.reserve(end - begin);
    for (size_t i = begin; i < end; ++i) {
        aligned.push_back(seed_rad + WrapAngleRad(samples[i].yaw_error_rad - seed_rad));
    }
    return aligned;
}

double MedianOfVector(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    const size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + mid, values.end());
    double median = values[mid];
    if (values.size() % 2 == 0) {
        std::nth_element(values.begin(), values.begin() + mid - 1, values.end());
        median = 0.5 * (median + values[mid - 1]);
    }
    return median;
}

double WeightedMedian(std::vector<std::pair<double, double>> value_weight_pairs) {
    if (value_weight_pairs.empty()) {
        return 0.0;
    }
    std::sort(
        value_weight_pairs.begin(),
        value_weight_pairs.end(),
        [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
    double total_weight = 0.0;
    for (const auto& value_weight : value_weight_pairs) {
        total_weight += std::max(0.0, value_weight.second);
    }
    if (total_weight <= 0.0) {
        return value_weight_pairs[value_weight_pairs.size() / 2].first;
    }
    double cumulative_weight = 0.0;
    for (const auto& value_weight : value_weight_pairs) {
        cumulative_weight += std::max(0.0, value_weight.second);
        if (cumulative_weight >= 0.5 * total_weight) {
            return value_weight.first;
        }
    }
    return value_weight_pairs.back().first;
}

double MedianAbsoluteDeviation(const std::vector<double>& values, double median) {
    std::vector<double> absolute_deviations;
    absolute_deviations.reserve(values.size());
    for (const double value : values) {
        absolute_deviations.push_back(std::abs(value - median));
    }
    return MedianOfVector(std::move(absolute_deviations));
}

double LinearSlope(const std::vector<YawFeedbackSample>& samples, size_t begin, size_t end, const std::vector<double>& values) {
    if (end <= begin + 1 || values.size() != end - begin) {
        return 0.0;
    }
    double mean_time = 0.0;
    double mean_value = 0.0;
    for (size_t i = begin; i < end; ++i) {
        mean_time += samples[i].time;
        mean_value += values[i - begin];
    }
    const double inv_count = 1.0 / static_cast<double>(end - begin);
    mean_time *= inv_count;
    mean_value *= inv_count;

    double covariance = 0.0;
    double variance = 0.0;
    for (size_t i = begin; i < end; ++i) {
        const double dt = samples[i].time - mean_time;
        covariance += dt * (values[i - begin] - mean_value);
        variance += dt * dt;
    }
    if (variance <= 1.0e-12) {
        return 0.0;
    }
    return covariance / variance;
}

spline::ControlPointArray BuildKnotGridFromNominal(
    const NominalNavStates& nominal_nav,
    const Vector3d& origin_blh,
    double knot_dt_s) {
    spline::ControlPointArray knots;
    if (nominal_nav.empty() || knot_dt_s <= 0.0) {
        return knots;
    }

    const double start_time = nominal_nav.front().time;
    const double end_time = nominal_nav.back().time;
    const double span = end_time - start_time;

    // Nominal nav times coincide with IMU propagation steps. Snapping interior knot targets to
    // the nearest nominal sample makes knot_interval.end_time align with an IMU interval end
    // (interval_propagation.cpp ~1e-6 s check), so ErrorStateIntervalFactor is not dropped.
    const auto snap_to_nearest_nominal_time = [&](double t) -> double {
        const auto upper = std::lower_bound(
            nominal_nav.begin(),
            nominal_nav.end(),
            t,
            [](const NominalNavState& state, double time) { return state.time < time; });
        if (upper == nominal_nav.begin()) {
            return nominal_nav.front().time;
        }
        if (upper == nominal_nav.end()) {
            return nominal_nav.back().time;
        }
        const double t1 = upper->time;
        const double t0 = (upper - 1)->time;
        return (std::abs(t1 - t) < std::abs(t - t0)) ? t1 : t0;
    };

    std::vector<double> knot_times;
    if (span <= 1.0e-12) {
        knot_times.push_back(start_time);
    } else {
        const int n_intervals =
            std::max(3, static_cast<int>(std::ceil(span / knot_dt_s)));
        const double dt_uniform = span / static_cast<double>(n_intervals);
        knot_times.reserve(static_cast<size_t>(n_intervals) + 1U);
        for (int i = 0; i <= n_intervals; ++i) {
            const double ideal = start_time + dt_uniform * static_cast<double>(i);
            double t_k = ideal;
            if (i == 0) {
                t_k = start_time;
            } else if (i == n_intervals) {
                t_k = end_time;
            } else {
                t_k = snap_to_nearest_nominal_time(ideal);
            }
            if (!knot_times.empty() && t_k <= knot_times.back() + 1.0e-9) {
                const auto it = std::upper_bound(
                    nominal_nav.begin(),
                    nominal_nav.end(),
                    knot_times.back(),
                    [](double time, const NominalNavState& st) { return time < st.time; });
                if (it == nominal_nav.end()) {
                    break;
                }
                t_k = it->time;
            }
            if (!knot_times.empty() && t_k <= knot_times.back() + 1.0e-9) {
                continue;
            }
            knot_times.push_back(t_k);
        }
        if (knot_times.empty() || std::abs(knot_times.back() - end_time) > 1.0e-9) {
            if (knot_times.empty() || end_time > knot_times.back() + 1.0e-9) {
                knot_times.push_back(end_time);
            } else {
                knot_times.back() = end_time;
            }
        }
    }

    knots.reserve(knot_times.size());
    for (double t : knot_times) {
        const auto nominal_state = EvaluateNominalState(nominal_nav, t);
        if (!nominal_state) {
            continue;
        }
        const Vector3d local_ned = Earth::GlobalToLocal(origin_blh, nominal_state->blh);
        knots.emplace_back(t, Sophus::SE3d(nominal_state->q_nb, local_ned));
    }
    return knots;
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
    ImuExtrinsicDefaults extra;
    if (!LoadAppConfigYaml(config_path, config_, extra)) {
        return false;
    }
    lever_arm_ = extra.lever_arm;
    initial_q_body_imu_ = extra.q_body_imu;
    q_body_imu_ = initial_q_body_imu_;
    return true;
}

bool System::Run() {
    yaw_bias_rad_ = 0.0;
    yaw_bias_feedback_total_rad_ = 0.0;
    post_opt_reprop_trigger_count_ = 0;
    post_opt_reprop_incremental_count_ = 0;
    post_opt_reprop_full_rebuild_fallback_count_ = 0;
    post_opt_reprop_total_covered_s_ = 0.0;
    post_opt_reprop_last_covered_s_ = 0.0;
    post_opt_reprop_last_tail_error_s_ = 0.0;
    post_opt_reprop_max_tail_error_s_ = 0.0;
    sliding_marginalization_trigger_count_ = 0;
    sliding_marginalization_removed_knots_total_ = 0;
    sliding_window_total_build_solve_s_ = 0.0;
    sliding_window_total_marg_s_ = 0.0;
    sliding_window_total_reprop_s_ = 0.0;
    sliding_window_current_max_iterations_ = config_.solver_max_iterations_window;
    if (!LoadMeasurements()) {
        return false;
    }
    TrimMeasurementsToTimeWindow();
    if (gnss_.empty() || imu_.empty()) {
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
        initial_alignment_ = EstimateInitialAlignment(imu_, origin_blh_, config_.align_time_s);
        initial_q_nb_ = initial_alignment_.q_nb;
        config_.start_time = std::max(config_.start_time, initial_alignment_.reference_time);
        TrimMeasurementsToTimeWindow();
        if (gnss_.empty() || imu_.empty()) {
            LOG(ERROR) << "No measurements remain after applying alignment reference time";
            return false;
        }
    }
    initial_q_nb_ = initial_alignment_.q_nb;

    if (IsPureInertialReplay()) {
        LOG(INFO) << "Pure inertial replay mode: skipping spline and optimization";
        UpdateNominalTrajectoryFromCurrentBiases();
        if (nominal_nav_.empty()) {
            LOG(ERROR) << "Nominal mechanization propagation failed in pure inertial replay";
            return false;
        }
        return SaveOutputs();
    }

    if (config_.sliding_window_enabled && config_.sliding_window_causal) {
        if (gnss_.empty() || imu_.empty()) {
            LOG(ERROR) << "Cannot run causal sliding without GNSS and IMU";
            return false;
        }
    } else if (!InitializeControlPoints()) {
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
        if (config_.sliding_window_enabled) {
            if (config_.sliding_window_causal && outer_iter > 0) {
                LOG(INFO) << "Causal sliding: skipping additional outer solves (use outer_iterations: 1)";
            } else if (!BuildAndSolveProblemSliding()) {
                return false;
            }
        } else if (!BuildAndSolveProblem()) {
            return false;
        }
        if (outer_iter + 1 < config_.outer_iterations) {
            ApplyInitialYawFeedbackFromGnss();
        }
        if (!InjectCurrentErrorStateIntoNominalTrajectory()) {
            LOG(ERROR) << "Failed to inject current error-state estimate into nominal trajectory";
            return false;
        }
        if (!config_.sliding_window_enabled) {
            if (!RepropagateNominalToLatestImuAfterOptimization(0)) {
                LOG(ERROR) << "Failed to repropagate nominal trajectory after optimization";
                return false;
            }
        }
        if (outer_iter + 1 < config_.outer_iterations) {
            if (config_.sliding_window_causal) {
                LOG(INFO) << "Causal sliding: skipping control-point reset between outer iterations";
            } else if (!ResetControlPointsFromNominalTrajectory(false)) {
                LOG(ERROR) << "Failed to reset control points from updated nominal trajectory";
                return false;
            }
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
    LOG(INFO) << "GNSS vertical Cauchy scale (m): " << config_.gnss_vertical_cauchy_scale_m;
    LOG(INFO) << "IMU sigma(a/g): " << config_.imu_sigma_accel_mps2 << ", " << config_.imu_sigma_gyro_rps;
    LOG(INFO) << "IMU stride: " << config_.imu_stride;
    LOG(INFO) << "Outer iterations: " << config_.outer_iterations;
    LOG(INFO) << "Enable initial yaw feedback: " << (config_.enable_initial_yaw_feedback ? "true" : "false");
    LOG(INFO) << "Enable yaw bias optimization: " << (config_.yaw_bias_enable ? "true" : "false");
    if (config_.yaw_bias_enable) {
        LOG(INFO) << "  yaw_bias prior_sigma_deg=" << (config_.yaw_bias_prior_sigma_rad * 180.0 / M_PI)
                  << " heading_sigma_deg=" << (config_.yaw_bias_heading_sigma_rad * 180.0 / M_PI)
                  << " heading_min_speed_mps=" << config_.yaw_bias_heading_min_speed_mps
                  << " step_limit_deg=" << (config_.yaw_bias_window_step_limit_rad * 180.0 / M_PI);
    }
    LOG(INFO) << "Use GNSS factors: " << (config_.use_gnss_factors ? "true" : "false");
    LOG(INFO) << "Use IMU factors: " << (config_.use_imu_factors ? "true" : "false");
    LOG(INFO) << "Sliding window: " << (config_.sliding_window_enabled ? "true" : "false");
    if (config_.sliding_window_enabled) {
        LOG(INFO) << "  causal=" << (config_.sliding_window_causal ? "true" : "false")
                  << " knots=" << config_.sliding_window_knots << " step_knots=" << config_.sliding_window_step_knots
                  << " marg=" << (config_.sliding_window_marginalization ? "true" : "false")
                  << " win_solver_iter=" << config_.solver_max_iterations_window
                  << " adaptive_win_iter=" << (config_.sliding_window_adaptive_solver_iterations ? "true" : "false")
                  << " win_func_tol=" << config_.sliding_window_function_tolerance
                  << " win_grad_tol=" << config_.sliding_window_gradient_tolerance;
    }
    LOG(INFO) << "Output query dt: " << config_.output_query_dt_s;
    LOG(INFO) << "Enable body NHC: " << (config_.body_frame.enable_nhc ? "true" : "false");
    LOG(INFO) << "Estimate q_body_imu: "
              << ((config_.body_frame.enable_nhc && config_.body_frame.estimate_q_body_imu) ? "true" : "false");
    LOG(INFO) << "NHC axes enabled (vx, vy, vz): "
              << config_.body_frame.nhc_enable_vx << ", "
              << config_.body_frame.nhc_enable_vy << ", "
              << config_.body_frame.nhc_enable_vz;
}

bool System::LoadMeasurements() {
    return LoadMeasurementBundle(config_, gnss_, imu_, nhc_, true);
}

void System::TrimMeasurementsToTimeWindow() {
    TrimNavMeasurementsToConfigWindow(config_, gnss_, imu_, nhc_);
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

    nominal_nav_ = PropagateNominalTrajectory(
        imu_,
        origin_blh_,
        initial_alignment_,
        {},
        {},
        {},
        {},
        {});
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

    spline::ControlPointArray new_control_points =
        BuildKnotGridFromNominal(nominal_nav_, origin_blh_, config_.spline_dt_s);
    if (new_control_points.empty()) {
        LOG(ERROR) << "Knot grid initialization from nominal trajectory produced no control points";
        return false;
    }

    AlignedVec3Array new_delta_theta(new_control_points.size(), Vector3d::Zero());
    AlignedVec3Array new_delta_vel(new_control_points.size(), Vector3d::Zero());
    AlignedVec3Array new_delta_pos(new_control_points.size(), Vector3d::Zero());
    AlignedVec3Array new_delta_bg(new_control_points.size(), Vector3d::Zero());
    AlignedVec3Array new_delta_ba(new_control_points.size(), Vector3d::Zero());
    AlignedVec3Array new_delta_sg(new_control_points.size(), Vector3d::Zero());
    AlignedVec3Array new_delta_sa(new_control_points.size(), Vector3d::Zero());
    if (!reset_biases) {
        if (delta_theta_nodes_.size() == new_control_points.size() &&
            delta_vel_nodes_.size() == new_control_points.size() &&
            delta_pos_nodes_.size() == new_control_points.size() &&
            delta_bg_nodes_.size() == new_control_points.size() &&
            delta_ba_nodes_.size() == new_control_points.size() &&
            delta_sg_nodes_.size() == new_control_points.size() &&
            delta_sa_nodes_.size() == new_control_points.size()) {
            new_delta_theta = delta_theta_nodes_;
            new_delta_vel = delta_vel_nodes_;
            new_delta_pos = delta_pos_nodes_;
            new_delta_bg = delta_bg_nodes_;
            new_delta_ba = delta_ba_nodes_;
            new_delta_sg = delta_sg_nodes_;
            new_delta_sa = delta_sa_nodes_;
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
    delta_sg_nodes_ = std::move(new_delta_sg);
    delta_sa_nodes_ = std::move(new_delta_sa);
    try {
        const auto t_cache0 = std::chrono::steady_clock::now();
        BuildIntervalPropagationCache(
            imu_,
            nominal_nav_,
            control_points_,
            config_.imu_sigma_gyro_rps,
            config_.imu_sigma_accel_mps2,
            config_.gyro_bias_rw_sigma,
            config_.accel_bias_rw_sigma,
            config_.gyro_scale_rw_sigma,
            config_.accel_scale_rw_sigma,
            config_.bias_tau_s,
            interval_cache_);
        if (config_.sliding_window_log_timing) {
            const auto t_cache1 = std::chrono::steady_clock::now();
            LOG(INFO) << "BuildIntervalPropagationCache wall (s): "
                      << std::chrono::duration<double>(t_cache1 - t_cache0).count();
        }
    } catch (const std::exception& ex) {
        LOG(ERROR) << "BuildIntervalPropagationCache failed: " << ex.what();
        return false;
    } catch (...) {
        LOG(ERROR) << "BuildIntervalPropagationCache failed with unknown exception";
        return false;
    }
    return !control_points_.empty();
}

FactorGraphSession System::MakeFactorGraphSession() {
    FactorGraphSession session{};
    session.config = &config_;
    session.origin_blh = &origin_blh_;
    session.gnss = &gnss_;
    session.imu = &imu_;
    session.nhc = &nhc_;
    session.control_points = &control_points_;
    session.delta_theta_nodes = &delta_theta_nodes_;
    session.delta_vel_nodes = &delta_vel_nodes_;
    session.delta_pos_nodes = &delta_pos_nodes_;
    session.delta_bg_nodes = &delta_bg_nodes_;
    session.delta_ba_nodes = &delta_ba_nodes_;
    session.delta_sg_nodes = &delta_sg_nodes_;
    session.delta_sa_nodes = &delta_sa_nodes_;
    session.lever_arm = &lever_arm_;
    session.time_offset_s = &time_offset_s_;
    session.yaw_bias_rad = &yaw_bias_rad_;
    session.q_body_imu = &q_body_imu_;
    session.nominal_nav = &nominal_nav_;
    session.interval_cache = &interval_cache_;
    session.window_knot_lo = -1;
    session.window_knot_hi = -1;
    session.marginalization_frontier = nullptr;
    return session;
}

bool System::BuildAndSolveProblem() {
    FactorGraphSession session = MakeFactorGraphSession();
    return BuildAndSolveFactorGraph(session);
}

bool System::BuildAndSolveProblemSliding() {
    if (config_.sliding_window_marginalization &&
        config_.body_frame.enable_nhc &&
        config_.body_frame.estimate_q_body_imu) {
        LOG(ERROR) << "Unsupported combination: sliding_window.marginalization=true with "
                   << "body_frame.enable_nhc=true and body_frame.estimate_q_body_imu=true. "
                   << "Current 15D frontier cannot preserve historical q_body_imu coupling.";
        return false;
    }
    if (config_.sliding_window_causal) {
        return BuildAndSolveProblemSlidingCausal();
    }
    return BuildAndSolveProblemSlidingReplayFullSpan();
}

bool System::RunSlidingWindowPass(
    int k_lo,
    int W,
    double* acc_build_solve_seconds,
    double* acc_marg_seconds,
    double* acc_reprop_seconds,
    bool has_future_knot) {
    const int k_hi = k_lo + W - 1;
    FactorGraphSession session = MakeFactorGraphSession();
    session.window_knot_lo = k_lo;
    session.window_knot_hi = k_hi;
    const MarginalizationFrontier* marg_ptr = nullptr;
    if (marginalization_frontier_.valid && marginalization_frontier_.anchor_knot_index == k_lo) {
        marg_ptr = &marginalization_frontier_;
    }
    session.marginalization_frontier = marg_ptr;
    if (config_.yaw_bias_enable && config_.yaw_bias_window_step_limit_rad > 0.0) {
        session.has_yaw_bias_step_limit = true;
        session.yaw_bias_center_rad = yaw_bias_rad_;
        session.yaw_bias_step_limit_rad = config_.yaw_bias_window_step_limit_rad;
    }

    WindowSolverStats win_stats{};
    session.window_solver_stats_out =
        config_.sliding_window_adaptive_solver_iterations ? &win_stats : nullptr;
    session.sliding_solver_max_iterations_override =
        config_.sliding_window_adaptive_solver_iterations ? sliding_window_current_max_iterations_ : -1;

    const auto t0 = std::chrono::steady_clock::now();
    if (!BuildAndSolveFactorGraph(session)) {
        return false;
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double dt_build_solve = std::chrono::duration<double>(t1 - t0).count();
    sliding_window_total_build_solve_s_ += dt_build_solve;
    if (acc_build_solve_seconds) {
        *acc_build_solve_seconds += dt_build_solve;
    }

    if (config_.sliding_window_adaptive_solver_iterations) {
        const auto term = static_cast<ceres::TerminationType>(win_stats.termination_type);
        if (term != ceres::FAILURE && win_stats.num_successful_steps <= 2 && win_stats.initial_cost > 0.0 &&
            win_stats.final_cost < win_stats.initial_cost) {
            sliding_window_current_max_iterations_ =
                std::max(config_.sliding_window_adaptive_solver_min_iterations, sliding_window_current_max_iterations_ - 2);
        } else {
            sliding_window_current_max_iterations_ = config_.solver_max_iterations_window;
        }
    }

    double dt_marg = 0.0;
    if (config_.sliding_window_marginalization && has_future_knot) {
        const auto tm0 = std::chrono::steady_clock::now();
        MarginalizationFrontier next{};
        if (!MarginalizeOldestKnotTwoKnotWindow(k_lo, session, marg_ptr, next)) {
            LOG(ERROR) << "Marginalization failed at k_lo=" << k_lo
                       << "; aborting sliding pass to avoid weakly constrained windows";
            marginalization_frontier_.reset();
            return false;
        } else {
            marginalization_frontier_ = next;
            sliding_marginalization_trigger_count_ += 1;
            sliding_marginalization_removed_knots_total_ += 1;
            if (marginalization_frontier_.anchor_knot_index != k_lo + 1) {
                LOG(ERROR) << "Unexpected marginalization anchor index: got "
                           << marginalization_frontier_.anchor_knot_index << " expected " << (k_lo + 1);
                return false;
            }
        }
        const auto tm1 = std::chrono::steady_clock::now();
        dt_marg = std::chrono::duration<double>(tm1 - tm0).count();
        sliding_window_total_marg_s_ += dt_marg;
        if (acc_marg_seconds) {
            *acc_marg_seconds += dt_marg;
        }
    } else if (has_future_knot) {
        marginalization_frontier_.reset();
    }

    const auto tr0 = std::chrono::steady_clock::now();
    if (!RepropagateNominalToLatestImuAfterOptimization(k_lo)) {
        LOG(ERROR) << "Sliding window step repropagation failed at k_lo=" << k_lo;
        return false;
    }
    const auto tr1 = std::chrono::steady_clock::now();
    const double dt_reprop = std::chrono::duration<double>(tr1 - tr0).count();
    sliding_window_total_reprop_s_ += dt_reprop;
    if (acc_reprop_seconds) {
        *acc_reprop_seconds += dt_reprop;
    }

    if (config_.sliding_window_log_timing) {
        LOG(INFO) << "Sliding window k_lo=" << k_lo << " wall (s): build+solve=" << dt_build_solve
                  << " marginalization=" << dt_marg << " reprop=" << dt_reprop;
    }
    return true;
}

bool System::BuildAndSolveProblemSlidingReplayFullSpan() {
    marginalization_frontier_.reset();
    sliding_window_total_build_solve_s_ = 0.0;
    sliding_window_total_marg_s_ = 0.0;
    sliding_window_total_reprop_s_ = 0.0;
    sliding_window_current_max_iterations_ = config_.solver_max_iterations_window;
    const int n_knots = static_cast<int>(control_points_.size());
    if (n_knots < 2) {
        LOG(ERROR) << "Sliding window requires at least two control points";
        return false;
    }
    int W = std::max(3, config_.sliding_window_knots);
    int step = std::max(1, config_.sliding_window_step_knots);
    if (W > 1 && step >= W) {
        LOG(WARNING) << "sliding_window_step_knots reset from " << step
                     << " to " << (W - 1)
                     << " (to avoid uncovered gaps between adjacent windows)";
        step = W - 1;
    }
    if (config_.sliding_window_marginalization && step != 1) {
        LOG(WARNING) << "sliding_window_step_knots reset from " << step << " to 1 (required for marginalization)";
        step = 1;
    }
    if (W > n_knots) {
        LOG(INFO) << "Sliding window knots " << W << " > trajectory knots " << n_knots
                  << "; solving one full-span window";
        W = n_knots;
    }

    const auto t_wall0 = std::chrono::steady_clock::now();
    int last_k_lo_executed = -1;

    for (int k_lo = 0; k_lo + W <= n_knots; k_lo += step) {
        const bool has_future_knot = (k_lo + W) < n_knots;
        if (!RunSlidingWindowPass(k_lo, W, nullptr, nullptr, nullptr, has_future_knot)) {
            return false;
        }
        last_k_lo_executed = k_lo;
    }
    const int k_tail = n_knots - W;
    if (k_tail > last_k_lo_executed) {
        LOG(INFO) << "Sliding window tail solve at k_lo=" << k_tail << " (covers knots to end)";
        if (!RunSlidingWindowPass(k_tail, W, nullptr, nullptr, nullptr, false)) {
            return false;
        }
    }

    if (config_.sliding_window_log_timing) {
        const auto t_wall1 = std::chrono::steady_clock::now();
        LOG(INFO) << "Sliding window total wall (s): " << std::chrono::duration<double>(t_wall1 - t_wall0).count()
                  << " build+solve_sum_s=" << sliding_window_total_build_solve_s_
                  << " marginalization_sum_s=" << sliding_window_total_marg_s_
                  << " reprop_sum_s=" << sliding_window_total_reprop_s_;
    }
    return true;
}

bool System::BuildAndSolveProblemSlidingCausal() {
    if (config_.outer_iterations > 1) {
        LOG(WARNING) << "sliding_window_causal: only the first outer iteration runs the sliding estimator; "
                     << "set outer_iterations: 1 for strict online semantics.";
    }

    marginalization_frontier_.reset();
    sliding_window_total_build_solve_s_ = 0.0;
    sliding_window_total_marg_s_ = 0.0;
    sliding_window_total_reprop_s_ = 0.0;
    sliding_window_current_max_iterations_ = config_.solver_max_iterations_window;

    const NominalNavStates nominal_schedule =
        PropagateNominalTrajectory(imu_, origin_blh_, initial_alignment_, {}, {}, {}, {}, {});
    const spline::ControlPointArray knot_targets =
        BuildKnotGridFromNominal(nominal_schedule, origin_blh_, config_.spline_dt_s);
    if (knot_targets.size() < 2) {
        LOG(ERROR) << "Causal sliding requires at least two knot targets";
        return false;
    }

    nominal_nav_.clear();
    control_points_.clear();
    interval_cache_ = IntervalPropagationCache{};
    delta_theta_nodes_.clear();
    delta_vel_nodes_.clear();
    delta_pos_nodes_.clear();
    delta_bg_nodes_.clear();
    delta_ba_nodes_.clear();
    delta_sg_nodes_.clear();
    delta_sa_nodes_.clear();

    const int n_knots_final = static_cast<int>(knot_targets.size());
    int W = std::max(3, config_.sliding_window_knots);
    int step = std::max(1, config_.sliding_window_step_knots);
    if (W > 1 && step >= W) {
        LOG(WARNING) << "sliding_window_step_knots reset from " << step
                     << " to " << (W - 1)
                     << " (to avoid uncovered gaps between adjacent windows)";
        step = W - 1;
    }
    if (config_.sliding_window_marginalization && step != 1) {
        LOG(WARNING) << "sliding_window_step_knots reset from " << step << " to 1 (required for marginalization)";
        step = 1;
    }
    if (W > n_knots_final) {
        LOG(INFO) << "Sliding window knots " << W << " > knot schedule " << n_knots_final
                  << "; solving one full-span window";
        W = n_knots_final;
    }

    const auto t_wall0 = std::chrono::steady_clock::now();
    int last_k_lo_executed = -1;
    size_t imu_hi = 0;

    for (int k = 0; k < n_knots_final; ++k) {
        const double t_k = knot_targets[static_cast<size_t>(k)].Timestamp();

        while (nominal_nav_.empty() || nominal_nav_.back().time < t_k - 1.0e-6) {
            // Keep causal suffix mechanization representation consistent with the active graph:
            // before global injection/relinearization, pose/vel/att/bias corrections remain in
            // error-state nodes, so propagation uses nominal biases here.
            const std::vector<double> bias_times;
            const AlignedVec3Array full_bg;
            const AlignedVec3Array full_ba;
            const AlignedVec3Array full_sg;
            const AlignedVec3Array full_sa;
            if (imu_hi + 1 >= imu_.size()) {
                ExtendNominalNavToImuIndex(
                    nominal_nav_,
                    imu_,
                    origin_blh_,
                    initial_alignment_,
                    bias_times,
                    full_bg,
                    full_ba,
                    full_sg,
                    full_sa,
                    imu_.size() - 1);
                if (nominal_nav_.empty() || nominal_nav_.back().time < t_k - 1.0e-6) {
                    LOG(ERROR) << "Causal sliding: IMU stream ends before knot time " << t_k;
                    return false;
                }
                break;
            }
            ExtendNominalNavToImuIndex(
                nominal_nav_,
                imu_,
                origin_blh_,
                initial_alignment_,
                bias_times,
                full_bg,
                full_ba,
                full_sg,
                full_sa,
                imu_hi);
            if (nominal_nav_.back().time < t_k - 1.0e-6) {
                ++imu_hi;
            }
        }

        const auto nominal_state = EvaluateNominalState(nominal_nav_, t_k);
        if (!nominal_state) {
            LOG(ERROR) << "Causal sliding: failed to evaluate nominal at knot time " << t_k;
            return false;
        }
        const Vector3d local_ned = Earth::GlobalToLocal(origin_blh_, nominal_state->blh);
        control_points_.emplace_back(t_k, Sophus::SE3d(nominal_state->q_nb, local_ned));
        delta_theta_nodes_.push_back(Vector3d::Zero());
        delta_vel_nodes_.push_back(Vector3d::Zero());
        delta_pos_nodes_.push_back(Vector3d::Zero());
        delta_bg_nodes_.push_back(Vector3d::Zero());
        delta_ba_nodes_.push_back(Vector3d::Zero());
        delta_sg_nodes_.push_back(Vector3d::Zero());
        delta_sa_nodes_.push_back(Vector3d::Zero());

        try {
            const auto t_cache0 = std::chrono::steady_clock::now();
            AppendIntervalPropagationCache(
                imu_,
                nominal_nav_,
                control_points_,
                config_.imu_sigma_gyro_rps,
                config_.imu_sigma_accel_mps2,
                config_.gyro_bias_rw_sigma,
                config_.accel_bias_rw_sigma,
                config_.gyro_scale_rw_sigma,
                config_.accel_scale_rw_sigma,
                config_.bias_tau_s,
                interval_cache_);
            if (config_.sliding_window_log_timing) {
                const auto t_cache1 = std::chrono::steady_clock::now();
                LOG(INFO) << "Causal sliding: BuildIntervalPropagationCache wall (s): "
                          << std::chrono::duration<double>(t_cache1 - t_cache0).count()
                          << " knots=" << control_points_.size();
            }
        } catch (const std::exception& ex) {
            LOG(ERROR) << "BuildIntervalPropagationCache failed (causal): " << ex.what();
            return false;
        } catch (...) {
            LOG(ERROR) << "BuildIntervalPropagationCache failed (causal) with unknown exception";
            return false;
        }

        const int K = static_cast<int>(control_points_.size());
        if (K >= W) {
            const int k_lo = K - W;
            if (last_k_lo_executed < 0 || (k_lo - last_k_lo_executed) >= step) {
                if (config_.sliding_window_log_timing) {
                    LOG(INFO) << "Causal sliding: solve window k_lo=" << k_lo << " k_hi=" << (k_lo + W - 1);
                }
                const bool has_future_knot = (k + 1) < n_knots_final;
                if (!RunSlidingWindowPass(k_lo, W, nullptr, nullptr, nullptr, has_future_knot)) {
                    return false;
                }
                last_k_lo_executed = k_lo;
            }
        }
    }

    const int k_tail = n_knots_final - W;
    if (k_tail > last_k_lo_executed && k_tail >= 0) {
        LOG(INFO) << "Causal sliding: tail solve at k_lo=" << k_tail;
        if (!RunSlidingWindowPass(k_tail, W, nullptr, nullptr, nullptr, false)) {
            return false;
        }
    }

    if (config_.sliding_window_log_timing) {
        const auto t_wall1 = std::chrono::steady_clock::now();
        LOG(INFO) << "Causal sliding total wall (s): " << std::chrono::duration<double>(t_wall1 - t_wall0).count()
                  << " build+solve_sum_s=" << sliding_window_total_build_solve_s_
                  << " marginalization_sum_s=" << sliding_window_total_marg_s_
                  << " reprop_sum_s=" << sliding_window_total_reprop_s_;
    }

    return true;
}

bool System::ApplyInitialYawFeedbackFromGnss() {
    if (!config_.enable_initial_yaw_feedback || initial_yaw_feedback_applied_ || gnss_.size() < 2) {
        return false;
    }

    const double window_end_time = initial_alignment_.reference_time + config_.initial_yaw_feedback_window_s;
    std::vector<YawFeedbackSample> samples;
    samples.reserve(gnss_.size());

    for (size_t i = 1; i < gnss_.size(); ++i) {
        const double dt = gnss_[i].time - gnss_[i - 1].time;
        if (dt <= 1.0e-3) {
            continue;
        }

        const double mid_time = 0.5 * (gnss_[i].time + gnss_[i - 1].time);
        if (mid_time < initial_alignment_.reference_time || mid_time > window_end_time) {
            continue;
        }

        const Vector3d p_prev_ned = Earth::GlobalToLocal(origin_blh_, gnss_[i - 1].blh);
        const Vector3d p_cur_ned = Earth::GlobalToLocal(origin_blh_, gnss_[i].blh);
        const Vector3d v_ned = (p_cur_ned - p_prev_ned) / dt;
        const double horizontal_speed = v_ned.head<2>().norm();
        if (horizontal_speed < config_.initial_yaw_feedback_min_speed_mps) {
            continue;
        }

        const auto composed = EvaluateComposedState(mid_time);
        if (!composed) {
            continue;
        }

        const double rtk_yaw = std::atan2(v_ned.y(), v_ned.x());
        const Eigen::Quaterniond q_nb(Eigen::Matrix3d(composed->full_pose.so3().matrix()));
        const double ct_yaw = YawFromQuaternionNed(q_nb);
        const double yaw_error = WrapAngleRad(rtk_yaw - ct_yaw);
        samples.push_back(YawFeedbackSample{mid_time, horizontal_speed, yaw_error});
    }

    if (static_cast<int>(samples.size()) < config_.initial_yaw_feedback_min_pairs) {
        LOG(INFO) << "Skipping initial yaw feedback: only " << samples.size()
                  << " RTK heading pairs in the start window";
        return false;
    }

    const size_t window_size = std::min(
        samples.size(),
        static_cast<size_t>(std::max(config_.initial_yaw_feedback_min_pairs, 8)));
    const double kMaxSampleGapS = 0.75;
    const double kStartupSkipS = 4.0;
    size_t best_begin = 0;
    size_t best_end = window_size;
    double best_score = std::numeric_limits<double>::infinity();
    double best_center = 0.0;
    double best_mad = std::numeric_limits<double>::infinity();

    size_t first_candidate_begin = 0;
    while (first_candidate_begin < samples.size() &&
           samples[first_candidate_begin].time < samples.front().time + kStartupSkipS) {
        ++first_candidate_begin;
    }
    if (first_candidate_begin + window_size > samples.size()) {
        first_candidate_begin = 0;
    }

    for (size_t begin = first_candidate_begin; begin + window_size <= samples.size(); ++begin) {
        const size_t end = begin + window_size;
        bool has_large_gap = false;
        for (size_t i = begin + 1; i < end; ++i) {
            if (samples[i].time - samples[i - 1].time > kMaxSampleGapS) {
                has_large_gap = true;
                break;
            }
        }
        if (has_large_gap) {
            continue;
        }

        const double seed = CircularMeanRad(samples, begin, end);
        const std::vector<double> aligned = AlignAnglesAroundSeed(samples, begin, end, seed);
        const double center = MedianOfVector(aligned);
        const double mad = MedianAbsoluteDeviation(aligned, center);
        const double slope = LinearSlope(samples, begin, end, aligned);
        double mean_speed = 0.0;
        for (size_t i = begin; i < end; ++i) {
            mean_speed += samples[i].speed_mps;
        }
        mean_speed /= static_cast<double>(end - begin);

        const double score =
            4.0 * mad +
            1.5 * std::abs(slope) -
            0.05 * mean_speed +
            1.0e-4 * (samples[begin].time - initial_alignment_.reference_time);
        if (score < best_score) {
            best_score = score;
            best_begin = begin;
            best_end = end;
            best_center = center;
            best_mad = mad;
        }
    }

    if (!std::isfinite(best_score)) {
        LOG(INFO) << "Skipping initial yaw feedback: no contiguous RTK heading segment found";
        return false;
    }

    const double robust_gate = std::max(10.0 * kDegToRad, 3.0 * std::max(best_mad, 1.0 * kDegToRad));
    while (best_begin > 0) {
        const size_t candidate = best_begin - 1;
        if (samples[best_begin].time - samples[candidate].time > kMaxSampleGapS) {
            break;
        }
        const double aligned_error = best_center + WrapAngleRad(samples[candidate].yaw_error_rad - best_center);
        if (std::abs(aligned_error - best_center) > robust_gate) {
            break;
        }
        best_begin = candidate;
    }
    while (best_end < samples.size()) {
        if (samples[best_end].time - samples[best_end - 1].time > kMaxSampleGapS) {
            break;
        }
        const double aligned_error = best_center + WrapAngleRad(samples[best_end].yaw_error_rad - best_center);
        if (std::abs(aligned_error - best_center) > robust_gate) {
            break;
        }
        ++best_end;
    }

    const double refined_seed = CircularMeanRad(samples, best_begin, best_end);
    const std::vector<double> refined_aligned = AlignAnglesAroundSeed(samples, best_begin, best_end, refined_seed);
    const double refined_center = MedianOfVector(refined_aligned);
    const double refined_mad = MedianAbsoluteDeviation(refined_aligned, refined_center);
    const double refined_gate = std::max(8.0 * kDegToRad, 2.5 * std::max(refined_mad, 1.0 * kDegToRad));
    std::vector<std::pair<double, double>> inlier_value_weights;
    inlier_value_weights.reserve(best_end - best_begin);
    for (size_t i = best_begin; i < best_end; ++i) {
        const double aligned_error = refined_center + WrapAngleRad(samples[i].yaw_error_rad - refined_center);
        if (std::abs(aligned_error - refined_center) > refined_gate) {
            continue;
        }
        inlier_value_weights.emplace_back(
            aligned_error,
            std::clamp(samples[i].speed_mps, config_.initial_yaw_feedback_min_speed_mps, 5.0));
    }

    if (static_cast<int>(inlier_value_weights.size()) < config_.initial_yaw_feedback_min_pairs) {
        LOG(INFO) << "Skipping initial yaw feedback: robust inlier count only "
                  << inlier_value_weights.size();
        return false;
    }

    const size_t robust_inlier_count = inlier_value_weights.size();
    double yaw_correction = WeightedMedian(std::move(inlier_value_weights));
    yaw_correction = WrapAngleRad(yaw_correction);
    yaw_correction = std::clamp(
        yaw_correction,
        -config_.initial_yaw_feedback_max_abs_rad,
        config_.initial_yaw_feedback_max_abs_rad);

    if (std::abs(yaw_correction) < 1.0e-4) {
        LOG(INFO) << "Initial yaw feedback below threshold, skipping injection";
        return false;
    }

    const Eigen::Quaterniond q_yaw_correction(Eigen::AngleAxisd(yaw_correction, Vector3d::UnitZ()));
    initial_alignment_.q_nb = (q_yaw_correction * initial_alignment_.q_nb).normalized();
    initial_q_nb_ = initial_alignment_.q_nb;
    initial_yaw_feedback_applied_ = true;
    initial_yaw_feedback_total_rad_ += yaw_correction;

    LOG(INFO) << "Injected initial yaw feedback from RTK heading, correction = "
              << yaw_correction << " rad (" << yaw_correction / kDegToRad << " deg)"
              << ", raw_pair_count = " << samples.size()
              << ", selected_pair_count = " << (best_end - best_begin)
              << ", robust_inlier_count = " << robust_inlier_count;

    UpdateNominalTrajectoryFromCurrentBiases();
    return true;
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
    return ct_fgo_sim::EvaluateNominalGyroCenterAtTime(interval_cache_, time);
}

std::optional<Vector3d> System::EvaluateNominalAccelAtTime(double time) const {
    return ct_fgo_sim::EvaluateNominalAccelAtTime(interval_cache_, time);
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
        composed.full_sg = nominal_state->sg;
        composed.full_sa = nominal_state->sa;
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
    const auto delta_sg = EvaluateNodeValueAtTime(time, delta_sg_nodes_);
    const auto delta_sa = EvaluateNodeValueAtTime(time, delta_sa_nodes_);
    const auto delta_theta_dot = EvaluateNodeDerivativeAtTime(time, delta_theta_nodes_);
    const auto nominal_accel = EvaluateNominalAccelAtTime(time);
    const auto nominal_gyro = EvaluateNominalGyroCenterAtTime(time);
    if (!delta_theta || !delta_vel || !delta_pos || !delta_bg || !delta_ba || !delta_sg || !delta_sa ||
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
    composed.delta_sg = *delta_sg;
    composed.delta_sa = *delta_sa;
    composed.full_pose = Sophus::SE3d(full_rot, nominal_local_ned + *delta_pos);
    composed.full_vel_ned = nominal_state->vel_ned + *delta_vel;
    composed.full_vel_body = full_rot.inverse() * composed.full_vel_ned;
    composed.full_omega_body = *nominal_gyro + *delta_theta_dot + nominal_state->bg + *delta_bg;
    composed.full_accel_ned = *nominal_accel;
    composed.full_alpha_body = Vector3d::Zero();
    composed.full_bg = nominal_state->bg + *delta_bg;
    composed.full_ba = nominal_state->ba + *delta_ba;
    composed.full_sg = nominal_state->sg + *delta_sg;
    composed.full_sa = nominal_state->sa + *delta_sa;
    return composed;
}

bool System::InjectCurrentErrorStateIntoNominalTrajectory() {
    if (nominal_nav_.empty()) {
        LOG(ERROR) << "Cannot inject error state into an empty nominal trajectory";
        return false;
    }
    if (control_points_.empty()) {
        return true;
    }
    if (control_points_.size() != delta_theta_nodes_.size() ||
        control_points_.size() != delta_vel_nodes_.size() ||
        control_points_.size() != delta_pos_nodes_.size() ||
        control_points_.size() != delta_bg_nodes_.size() ||
        control_points_.size() != delta_ba_nodes_.size() ||
        control_points_.size() != delta_sg_nodes_.size() ||
        control_points_.size() != delta_sa_nodes_.size()) {
        LOG(ERROR) << "Node arrays are inconsistent with control-point count during error-state injection";
        return false;
    }

    double max_delta_theta_norm = 0.0;
    double max_delta_vel_norm = 0.0;
    double max_delta_pos_norm = 0.0;
    double max_delta_bg_norm = 0.0;
    double max_delta_ba_norm = 0.0;

    double yaw_feedback_apply_rad = 0.0;
    if (config_.yaw_bias_enable) {
        yaw_feedback_apply_rad = yaw_bias_rad_;
        const double max_abs = std::max(0.0, config_.yaw_bias_max_abs_rad);
        if (max_abs > 0.0) {
            const double lb = -max_abs - yaw_bias_feedback_total_rad_;
            const double ub = max_abs - yaw_bias_feedback_total_rad_;
            yaw_feedback_apply_rad = std::clamp(yaw_feedback_apply_rad, lb, ub);
        }
    }
    const Eigen::Quaterniond q_yaw_bias(Eigen::AngleAxisd(yaw_feedback_apply_rad, Vector3d::UnitZ()));
    for (auto& nominal_state : nominal_nav_) {
        const auto delta_theta = EvaluateNodeValueAtTime(nominal_state.time, delta_theta_nodes_);
        const auto delta_vel = EvaluateNodeValueAtTime(nominal_state.time, delta_vel_nodes_);
        const auto delta_pos = EvaluateNodeValueAtTime(nominal_state.time, delta_pos_nodes_);
        const auto delta_bg = EvaluateNodeValueAtTime(nominal_state.time, delta_bg_nodes_);
        const auto delta_ba = EvaluateNodeValueAtTime(nominal_state.time, delta_ba_nodes_);
        const auto delta_sg = EvaluateNodeValueAtTime(nominal_state.time, delta_sg_nodes_);
        const auto delta_sa = EvaluateNodeValueAtTime(nominal_state.time, delta_sa_nodes_);
        if (!delta_theta || !delta_vel || !delta_pos || !delta_bg || !delta_ba || !delta_sg || !delta_sa) {
            continue;
        }

        const Sophus::SO3d nominal_rot(nominal_state.q_nb);
        nominal_state.q_nb =
            (q_yaw_bias * (nominal_rot * Sophus::SO3d::exp(*delta_theta)).unit_quaternion()).normalized();
        nominal_state.vel_ned += *delta_vel;
        const Vector3d nominal_local_ned = Earth::GlobalToLocal(origin_blh_, nominal_state.blh);
        nominal_state.blh = Earth::LocalToGlobal(origin_blh_, nominal_local_ned + *delta_pos);
        nominal_state.bg += *delta_bg;
        nominal_state.ba += *delta_ba;
        nominal_state.sg += *delta_sg;
        nominal_state.sa += *delta_sa;

        max_delta_theta_norm = std::max(max_delta_theta_norm, delta_theta->norm());
        max_delta_vel_norm = std::max(max_delta_vel_norm, delta_vel->norm());
        max_delta_pos_norm = std::max(max_delta_pos_norm, delta_pos->norm());
        max_delta_bg_norm = std::max(max_delta_bg_norm, delta_bg->norm());
        max_delta_ba_norm = std::max(max_delta_ba_norm, delta_ba->norm());
    }

    if (!nominal_nav_.empty()) {
        initial_alignment_.q_nb = nominal_nav_.front().q_nb;
        initial_alignment_.vel0_ned = nominal_nav_.front().vel_ned;
        initial_alignment_.bg0 = nominal_nav_.front().bg;
        initial_alignment_.ba0 = nominal_nav_.front().ba;
        initial_q_nb_ = initial_alignment_.q_nb;
    }
    if (config_.yaw_bias_enable) {
        yaw_bias_feedback_total_rad_ += yaw_feedback_apply_rad;
        yaw_bias_rad_ = 0.0;
    }

    for (auto& delta_theta : delta_theta_nodes_) {
        delta_theta.setZero();
    }
    for (auto& delta_vel : delta_vel_nodes_) {
        delta_vel.setZero();
    }
    for (auto& delta_pos : delta_pos_nodes_) {
        delta_pos.setZero();
    }
    for (auto& delta_bg : delta_bg_nodes_) {
        delta_bg.setZero();
    }
    for (auto& delta_ba : delta_ba_nodes_) {
        delta_ba.setZero();
    }
    for (auto& delta_sg : delta_sg_nodes_) {
        delta_sg.setZero();
    }
    for (auto& delta_sa : delta_sa_nodes_) {
        delta_sa.setZero();
    }

    try {
        BuildIntervalPropagationCache(
            imu_,
            nominal_nav_,
            control_points_,
            config_.imu_sigma_gyro_rps,
            config_.imu_sigma_accel_mps2,
            config_.gyro_bias_rw_sigma,
            config_.accel_bias_rw_sigma,
            config_.gyro_scale_rw_sigma,
            config_.accel_scale_rw_sigma,
            config_.bias_tau_s,
            interval_cache_);
    } catch (const std::exception& ex) {
        LOG(ERROR) << "BuildIntervalPropagationCache failed after error-state injection: " << ex.what();
        return false;
    } catch (...) {
        LOG(ERROR) << "BuildIntervalPropagationCache failed with unknown exception after error-state injection";
        return false;
    }

    LOG(INFO) << "Injected error-state nodes into nominal trajectory, max |dtheta|="
              << max_delta_theta_norm << " rad, max |dv|=" << max_delta_vel_norm
              << " m/s, max |dp|=" << max_delta_pos_norm << " m, max |dbg|="
              << max_delta_bg_norm << " rad/s, max |dba|=" << max_delta_ba_norm << " m/s^2";
    if (config_.yaw_bias_enable) {
        LOG(INFO) << "Yaw-bias feedback total applied (rad): " << yaw_bias_feedback_total_rad_;
    }
    return true;
}

bool System::RepropagateNominalToLatestImuAfterOptimization(int reprop_knot_lo) {
    if (imu_.empty() || nominal_nav_.empty() || control_points_.empty()) {
        return true;
    }

    const double latest_imu_time = imu_.back().time;
    const double nominal_tail_before = nominal_nav_.back().time;
    NominalNavStates nominal_backup = nominal_nav_;
    IntervalPropagationCache cache_backup = interval_cache_;

    std::vector<double> bias_times;
    bias_times.reserve(control_points_.size());
    AlignedVec3Array full_bg_nodes;
    AlignedVec3Array full_ba_nodes;
    AlignedVec3Array full_sg_nodes;
    AlignedVec3Array full_sa_nodes;
    full_bg_nodes.reserve(control_points_.size());
    full_ba_nodes.reserve(control_points_.size());
    full_sg_nodes.reserve(control_points_.size());
    full_sa_nodes.reserve(control_points_.size());
    for (const auto& control_point : control_points_) {
        const double t_k = control_point.Timestamp();
        bias_times.push_back(t_k);
        const auto composed_k = EvaluateComposedState(t_k);
        if (composed_k) {
            full_bg_nodes.push_back(composed_k->full_bg);
            full_ba_nodes.push_back(composed_k->full_ba);
            full_sg_nodes.push_back(composed_k->full_sg);
            full_sa_nodes.push_back(composed_k->full_sa);
        } else {
            const auto nominal_k = EvaluateNominalState(nominal_nav_, t_k);
            if (nominal_k) {
                full_bg_nodes.push_back(nominal_k->bg);
                full_ba_nodes.push_back(nominal_k->ba);
                full_sg_nodes.push_back(nominal_k->sg);
                full_sa_nodes.push_back(nominal_k->sa);
            } else {
                full_bg_nodes.push_back(initial_alignment_.bg0);
                full_ba_nodes.push_back(initial_alignment_.ba0);
                full_sg_nodes.push_back(config_.init_sg);
                full_sa_nodes.push_back(config_.init_sa);
            }
        }
    }

    const int anchor_knot =
        std::clamp(reprop_knot_lo, 0, static_cast<int>(control_points_.size()) - 1);
    const double anchor_knot_time = control_points_[static_cast<size_t>(anchor_knot)].Timestamp();
    const auto imu_it = std::upper_bound(
        imu_.begin(), imu_.end(), anchor_knot_time, [](double t, const ImuMeasurement& m) { return t < m.time; });
    const size_t anchor_imu_index =
        imu_it == imu_.begin() ? 0 : static_cast<size_t>(std::distance(imu_.begin(), imu_it) - 1);
    const double anchor_imu_time = imu_[anchor_imu_index].time;

    try {
        if (anchor_imu_index + 1 < nominal_nav_.size()) {
            nominal_nav_.resize(anchor_imu_index + 1);
        }
        const auto composed_anchor = EvaluateComposedState(anchor_imu_time);
        if (composed_anchor) {
            NominalNavState anchor_state{};
            anchor_state.time = anchor_imu_time;
            anchor_state.blh = Earth::LocalToGlobal(origin_blh_, composed_anchor->full_pose.translation());
            anchor_state.vel_ned = composed_anchor->full_vel_ned;
            anchor_state.q_nb = composed_anchor->full_pose.unit_quaternion();
            anchor_state.bg = composed_anchor->full_bg;
            anchor_state.ba = composed_anchor->full_ba;
            anchor_state.sg = composed_anchor->full_sg;
            anchor_state.sa = composed_anchor->full_sa;
            if (nominal_nav_.empty()) {
                nominal_nav_.push_back(anchor_state);
            } else {
                nominal_nav_.back() = anchor_state;
            }
        }

        ExtendNominalNavToImuIndex(
            nominal_nav_,
            imu_,
            origin_blh_,
            initial_alignment_,
            bias_times,
            full_bg_nodes,
            full_ba_nodes,
            full_sg_nodes,
            full_sa_nodes,
            imu_.size() - 1);

        while (!interval_cache_.imu_intervals.empty() &&
               interval_cache_.imu_intervals.back().imu_index > anchor_imu_index) {
            interval_cache_.imu_intervals.pop_back();
        }
        if (static_cast<size_t>(anchor_knot) < interval_cache_.knot_intervals.size()) {
            interval_cache_.knot_intervals.resize(static_cast<size_t>(anchor_knot));
        }

        AppendIntervalPropagationCache(
            imu_,
            nominal_nav_,
            control_points_,
            config_.imu_sigma_gyro_rps,
            config_.imu_sigma_accel_mps2,
            config_.gyro_bias_rw_sigma,
            config_.accel_bias_rw_sigma,
            config_.gyro_scale_rw_sigma,
            config_.accel_scale_rw_sigma,
            config_.bias_tau_s,
            interval_cache_);
        post_opt_reprop_incremental_count_ += 1;
    } catch (const std::exception& ex) {
        LOG(WARNING) << "Post-optimization incremental repropagation failed, fallback to full rebuild: " << ex.what();
        nominal_nav_ = std::move(nominal_backup);
        interval_cache_ = std::move(cache_backup);
        try {
            nominal_nav_ = PropagateNominalTrajectory(
                imu_,
                origin_blh_,
                initial_alignment_,
                bias_times,
                full_bg_nodes,
                full_ba_nodes,
                full_sg_nodes,
                full_sa_nodes);
            BuildIntervalPropagationCache(
                imu_,
                nominal_nav_,
                control_points_,
                config_.imu_sigma_gyro_rps,
                config_.imu_sigma_accel_mps2,
                config_.gyro_bias_rw_sigma,
                config_.accel_bias_rw_sigma,
                config_.gyro_scale_rw_sigma,
                config_.accel_scale_rw_sigma,
                config_.bias_tau_s,
                interval_cache_);
            post_opt_reprop_full_rebuild_fallback_count_ += 1;
        } catch (...) {
            nominal_nav_ = std::move(nominal_backup);
            interval_cache_ = std::move(cache_backup);
            return true;
        }
    } catch (...) {
        LOG(WARNING) << "Post-optimization incremental repropagation failed with unknown exception, restoring previous nominal/cache";
        nominal_nav_ = std::move(nominal_backup);
        interval_cache_ = std::move(cache_backup);
        return true;
    }

    for (auto& delta_bg : delta_bg_nodes_) {
        delta_bg.setZero();
    }
    for (auto& delta_ba : delta_ba_nodes_) {
        delta_ba.setZero();
    }
    for (auto& delta_sg : delta_sg_nodes_) {
        delta_sg.setZero();
    }
    for (auto& delta_sa : delta_sa_nodes_) {
        delta_sa.setZero();
    }

    const double nominal_tail_after = nominal_nav_.empty() ? nominal_tail_before : nominal_nav_.back().time;
    post_opt_reprop_trigger_count_ += 1;
    post_opt_reprop_last_covered_s_ = std::max(0.0, nominal_tail_after - nominal_tail_before);
    post_opt_reprop_total_covered_s_ += post_opt_reprop_last_covered_s_;
    post_opt_reprop_last_tail_error_s_ = std::max(0.0, latest_imu_time - nominal_tail_after);
    post_opt_reprop_max_tail_error_s_ =
        std::max(post_opt_reprop_max_tail_error_s_, post_opt_reprop_last_tail_error_s_);

    LOG(INFO) << "Post-opt repropagation triggered (anchor knot " << anchor_knot
              << "), covered " << post_opt_reprop_last_covered_s_ << " s, latest IMU tail gap "
              << post_opt_reprop_last_tail_error_s_ << " s";
    return true;
}

void System::UpdateNominalTrajectoryFromCurrentBiases() {
    std::vector<double> bias_times;
    bias_times.reserve(control_points_.size());
    AlignedVec3Array full_bg_nodes;
    AlignedVec3Array full_ba_nodes;
    AlignedVec3Array full_sg_nodes;
    AlignedVec3Array full_sa_nodes;
    full_bg_nodes.reserve(control_points_.size());
    full_ba_nodes.reserve(control_points_.size());
    full_sg_nodes.reserve(control_points_.size());
    full_sa_nodes.reserve(control_points_.size());
    for (const auto& control_point : control_points_) {
        bias_times.push_back(control_point.Timestamp());
    }

    double max_delta_bg_norm = 0.0;
    double max_delta_ba_norm = 0.0;
    const Vector3d base_sg = nominal_nav_.empty() ? config_.init_sg : nominal_nav_.front().sg;
    const Vector3d base_sa = nominal_nav_.empty() ? config_.init_sa : nominal_nav_.front().sa;
    if (!control_points_.empty() &&
        control_points_.size() == delta_bg_nodes_.size() &&
        control_points_.size() == delta_ba_nodes_.size() &&
        control_points_.size() == delta_sg_nodes_.size() &&
        control_points_.size() == delta_sa_nodes_.size()) {
        for (size_t i = 0; i < control_points_.size(); ++i) {
            full_bg_nodes.push_back(initial_alignment_.bg0 + delta_bg_nodes_[i]);
            full_ba_nodes.push_back(initial_alignment_.ba0 + delta_ba_nodes_[i]);
            full_sg_nodes.push_back(base_sg + delta_sg_nodes_[i]);
            full_sa_nodes.push_back(base_sa + delta_sa_nodes_[i]);
            max_delta_bg_norm = std::max(max_delta_bg_norm, delta_bg_nodes_[i].norm());
            max_delta_ba_norm = std::max(max_delta_ba_norm, delta_ba_nodes_[i].norm());
        }
    } else {
        for (size_t i = 0; i < control_points_.size(); ++i) {
            full_bg_nodes.push_back(initial_alignment_.bg0);
            full_ba_nodes.push_back(initial_alignment_.ba0);
            full_sg_nodes.push_back(base_sg);
            full_sa_nodes.push_back(base_sa);
        }
    }

    nominal_nav_ = PropagateNominalTrajectory(
        imu_,
        origin_blh_,
        initial_alignment_,
        bias_times,
        full_bg_nodes,
        full_ba_nodes,
        full_sg_nodes,
        full_sa_nodes);

    if (!control_points_.empty() &&
        control_points_.size() == delta_bg_nodes_.size() &&
        control_points_.size() == delta_ba_nodes_.size() &&
        control_points_.size() == delta_sg_nodes_.size() &&
        control_points_.size() == delta_sa_nodes_.size()) {
        LOG(INFO) << "Closed-loop bias feedback injected into nominal mechanization, max |delta_bg|="
                  << max_delta_bg_norm << " rad/s, max |delta_ba|=" << max_delta_ba_norm << " m/s^2";
        for (auto& delta_bg : delta_bg_nodes_) {
            delta_bg.setZero();
        }
        for (auto& delta_ba : delta_ba_nodes_) {
            delta_ba.setZero();
        }
        for (auto& delta_sg : delta_sg_nodes_) {
            delta_sg.setZero();
        }
        for (auto& delta_sa : delta_sa_nodes_) {
            delta_sa.setZero();
        }
    }

    if (control_points_.size() >= 2) {
        try {
            BuildIntervalPropagationCache(
                imu_,
                nominal_nav_,
                control_points_,
                config_.imu_sigma_gyro_rps,
                config_.imu_sigma_accel_mps2,
                config_.gyro_bias_rw_sigma,
                config_.accel_bias_rw_sigma,
                config_.gyro_scale_rw_sigma,
                config_.accel_scale_rw_sigma,
                config_.bias_tau_s,
                interval_cache_);
        } catch (const std::exception& ex) {
            LOG(ERROR) << "BuildIntervalPropagationCache failed: " << ex.what();
            interval_cache_ = IntervalPropagationCache();
        } catch (...) {
            LOG(ERROR) << "BuildIntervalPropagationCache failed with unknown exception";
            interval_cache_ = IntervalPropagationCache();
        }
    } else {
        interval_cache_ = IntervalPropagationCache();
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

    const std::filesystem::path dense_trajectory_path = config_.output_path / "dense_trajectory_enu.txt";
    std::ofstream dense_trajectory_ofs(dense_trajectory_path);
    dense_trajectory_ofs << "# time_s east_m north_m up_m qx qy qz qw\n";
    const double dense_dt = config_.output_query_dt_s > 0.0
        ? config_.output_query_dt_s
        : (config_.imu_main.rate_hz > 0.0 ? 1.0 / config_.imu_main.rate_hz : 0.0);
    if (dense_dt > 0.0 && config_.end_time > config_.start_time) {
        for (double query_time = config_.start_time; query_time <= config_.end_time + 1.0e-9; query_time += dense_dt) {
            const auto composed = EvaluateComposedState(query_time);
            if (!composed) {
                continue;
            }
            const Eigen::Quaterniond q = QnbNedToQebEnu(Eigen::Quaterniond(composed->full_pose.so3().matrix()));
            const Vector3d t = NedToEnu(composed->full_pose.translation());
            dense_trajectory_ofs << query_time << ' ' << t.x() << ' ' << t.y() << ' ' << t.z() << ' '
                                 << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w() << '\n';
        }
    }

    const std::filesystem::path bias_path = config_.output_path / "bias_nodes.txt";
    std::ofstream bias_ofs(bias_path);
    bias_ofs << "# time_s d_bgx d_bgy d_bgz d_bax d_bay d_baz d_sgx d_sgy d_sgz d_sax d_say d_saz\n";
    for (size_t i = 0; i < control_points_.size(); ++i) {
        bias_ofs << control_points_[i].Timestamp() << ' '
                 << delta_bg_nodes_[i].x() << ' ' << delta_bg_nodes_[i].y() << ' ' << delta_bg_nodes_[i].z() << ' '
                 << delta_ba_nodes_[i].x() << ' ' << delta_ba_nodes_[i].y() << ' ' << delta_ba_nodes_[i].z() << ' '
                 << delta_sg_nodes_[i].x() << ' ' << delta_sg_nodes_[i].y() << ' ' << delta_sg_nodes_[i].z() << ' '
                 << delta_sa_nodes_[i].x() << ' ' << delta_sa_nodes_[i].y() << ' ' << delta_sa_nodes_[i].z() << '\n';
    }

    const std::filesystem::path summary_path = config_.output_path / "run_summary.txt";
    std::ofstream summary_ofs(summary_path);
    summary_ofs << std::setprecision(17);
    std::vector<std::pair<double, double>> propagation_yaw_value_weight;
    const double initial_yaw_ref_rad = YawFromQuaternionNed(initial_q_nb_);
    const double propagation_heading_min_speed_mps = std::max(0.1, config_.initial_yaw_feedback_min_speed_mps);
    if (gnss_.size() >= 2) {
        propagation_yaw_value_weight.reserve(gnss_.size() - 1);
        for (size_t i = 1; i < gnss_.size(); ++i) {
            const auto& prev = gnss_[i - 1];
            const auto& curr = gnss_[i];
            const double dt = curr.time - prev.time;
            if (dt <= 1.0e-3) {
                continue;
            }
            const Vector3d p_prev = Earth::GlobalToLocal(origin_blh_, prev.blh);
            const Vector3d p_curr = Earth::GlobalToLocal(origin_blh_, curr.blh);
            const Vector3d vel_ned = (p_curr - p_prev) / dt;
            const double speed = vel_ned.head<2>().norm();
            if (speed < propagation_heading_min_speed_mps) {
                continue;
            }
            const double t_mid = 0.5 * (prev.time + curr.time);
            const auto composed = EvaluateComposedState(t_mid);
            if (!composed) {
                continue;
            }
            const Eigen::Quaterniond q_nb(Eigen::Quaterniond(composed->full_pose.so3().matrix()));
            const double yaw_now = YawFromQuaternionNed(q_nb);
            const double yaw_error = WrapAngleRad(yaw_now - initial_yaw_ref_rad);
            propagation_yaw_value_weight.emplace_back(yaw_error, std::clamp(speed, 0.5, 5.0));
        }
    }
    const double propagation_heading_error_rad_est = WeightedMedian(propagation_yaw_value_weight);
    const size_t propagation_heading_error_sample_count = propagation_yaw_value_weight.size();
    summary_ofs << "gnss_file: " << config_.gnss_file << '\n';
    summary_ofs << "imu_file: " << config_.imu_main.file << '\n';
    summary_ofs << "use_gnss_factors: " << config_.use_gnss_factors << '\n';
    summary_ofs << "use_imu_factors: " << config_.use_imu_factors << '\n';
    summary_ofs << "output_query_dt_s: " << config_.output_query_dt_s << '\n';
    summary_ofs << "gnss_count: " << gnss_.size() << '\n';
    summary_ofs << "imu_count: " << imu_.size() << '\n';
    summary_ofs << "control_point_count: " << control_points_.size() << '\n';
    summary_ofs << "outer_iterations: " << config_.outer_iterations << '\n';
    summary_ofs << "enable_initial_yaw_feedback: " << config_.enable_initial_yaw_feedback << '\n';
    summary_ofs << "initial_yaw_feedback_applied: " << initial_yaw_feedback_applied_ << '\n';
    summary_ofs << "initial_yaw_feedback_total_rad: " << initial_yaw_feedback_total_rad_ << '\n';
    summary_ofs << "propagation_heading_error_rad_est: " << propagation_heading_error_rad_est << '\n';
    summary_ofs << "propagation_heading_error_deg_est: " << (propagation_heading_error_rad_est * 180.0 / M_PI) << '\n';
    summary_ofs << "propagation_heading_error_sample_count: " << propagation_heading_error_sample_count << '\n';
    summary_ofs << "propagation_heading_error_min_speed_mps: " << propagation_heading_min_speed_mps << '\n';
    summary_ofs << "yaw_bias_enable: " << config_.yaw_bias_enable << '\n';
    summary_ofs << "yaw_bias_feedback_total_rad: " << yaw_bias_feedback_total_rad_ << '\n';
    summary_ofs << "yaw_bias_current_rad: " << yaw_bias_rad_ << '\n';
    const double reprop_avg_covered_s =
        post_opt_reprop_trigger_count_ == 0
            ? 0.0
            : post_opt_reprop_total_covered_s_ / static_cast<double>(post_opt_reprop_trigger_count_);
    summary_ofs << "post_opt_reprop_trigger_count: " << post_opt_reprop_trigger_count_ << '\n';
    summary_ofs << "post_opt_reprop_incremental_count: " << post_opt_reprop_incremental_count_ << '\n';
    summary_ofs << "post_opt_reprop_full_rebuild_fallback_count: " << post_opt_reprop_full_rebuild_fallback_count_
                << '\n';
    summary_ofs << "post_opt_reprop_total_covered_s: " << post_opt_reprop_total_covered_s_ << '\n';
    summary_ofs << "post_opt_reprop_avg_covered_s: " << reprop_avg_covered_s << '\n';
    summary_ofs << "post_opt_reprop_last_covered_s: " << post_opt_reprop_last_covered_s_ << '\n';
    summary_ofs << "post_opt_reprop_last_tail_error_s: " << post_opt_reprop_last_tail_error_s_ << '\n';
    summary_ofs << "post_opt_reprop_max_tail_error_s: " << post_opt_reprop_max_tail_error_s_ << '\n';
    summary_ofs << "sliding_marginalization_trigger_count: " << sliding_marginalization_trigger_count_ << '\n';
    summary_ofs << "sliding_marginalization_removed_knots_total: " << sliding_marginalization_removed_knots_total_
                << '\n';
    summary_ofs << "sliding_window_total_build_solve_s: " << sliding_window_total_build_solve_s_ << '\n';
    summary_ofs << "sliding_window_total_marginalization_s: " << sliding_window_total_marg_s_ << '\n';
    summary_ofs << "sliding_window_total_reprop_s: " << sliding_window_total_reprop_s_ << '\n';
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
    nominal_ofs << "# time_s lat_rad lon_rad h_m ve_mps vn_mps vu_mps qx qy qz qw "
                   "bgx bgy bgz bax bay baz sgx sgy sgz sax say saz\n";
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
                    << nav.ba.z() << ' '
                    << nav.sg.x() << ' '
                    << nav.sg.y() << ' '
                    << nav.sg.z() << ' '
                    << nav.sa.x() << ' '
                    << nav.sa.y() << ' '
                    << nav.sa.z() << '\n';
    }

    const std::filesystem::path delta_path = config_.output_path / "delta_estimates.txt";
    std::ofstream delta_ofs(delta_path);
    delta_ofs << "# time_s dtheta_x_rad dtheta_y_rad dtheta_z_rad "
                 "dvx_mps dvy_mps dvz_mps dpx_m dpy_m dpz_m dbg_x_rps dbg_y_rps dbg_z_rps "
                 "dba_x dba_y dba_z dsg_x dsg_y dsg_z dsa_x dsa_y dsa_z\n";
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
                  << composed->delta_ba.z() << ' '
                  << composed->delta_sg.x() << ' '
                  << composed->delta_sg.y() << ' '
                  << composed->delta_sg.z() << ' '
                  << composed->delta_sa.x() << ' '
                  << composed->delta_sa.y() << ' '
                  << composed->delta_sa.z() << '\n';
    }

    LOG(INFO) << "Wrote outputs to " << config_.output_path.string();
    return trajectory_ofs.good() && bias_ofs.good() && summary_ofs.good() && nominal_ofs.good() && delta_ofs.good();
}

}  // namespace ct_fgo_sim
