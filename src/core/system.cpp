#include "ct_fgo_sim/core/system.h"

#include "ct_fgo_sim/core/app_yaml_io.h"
#include "ct_fgo_sim/core/factor_graph_session.h"
#include "ct_fgo_sim/core/spline_helpers.h"

#include <glog/logging.h>
#include <sophus/so3.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>

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
    LOG(INFO) << "Static alignment window: [" << initial_alignment_.window_start_time
              << ", " << initial_alignment_.window_end_time << "]";
    LOG(INFO) << "Static alignment reference time: " << initial_alignment_.reference_time;

    for (int outer_iter = 0; outer_iter < config_.outer_iterations; ++outer_iter) {
        LOG(INFO) << "Outer iteration " << (outer_iter + 1) << "/" << config_.outer_iterations;
        if (!BuildAndSolveProblem()) {
            return false;
        }
        if (outer_iter + 1 < config_.outer_iterations) {
            ApplyInitialYawFeedbackFromGnss();
        }
        if (!InjectCurrentErrorStateIntoNominalTrajectory()) {
            LOG(ERROR) << "Failed to inject current error-state estimate into nominal trajectory";
            return false;
        }
        if (outer_iter + 1 < config_.outer_iterations) {
            if (!ResetControlPointsFromNominalTrajectory(false)) {
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
    LOG(INFO) << "Use GNSS factors: " << (config_.use_gnss_factors ? "true" : "false");
    LOG(INFO) << "Use IMU factors: " << (config_.use_imu_factors ? "true" : "false");
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
    try {
        BuildIntervalPropagationCache(
            imu_,
            nominal_nav_,
            control_points_,
            config_.imu_sigma_gyro_rps,
            config_.imu_sigma_accel_mps2,
            config_.gyro_bias_rw_sigma,
            config_.accel_bias_rw_sigma,
            config_.bias_tau_s,
            interval_cache_);
    } catch (const std::exception& ex) {
        LOG(ERROR) << "BuildIntervalPropagationCache failed: " << ex.what();
        return false;
    } catch (...) {
        LOG(ERROR) << "BuildIntervalPropagationCache failed with unknown exception";
        return false;
    }
    return !control_points_.empty();
}

bool System::BuildAndSolveProblem() {
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
    session.lever_arm = &lever_arm_;
    session.time_offset_s = &time_offset_s_;
    session.q_body_imu = &q_body_imu_;
    session.nominal_nav = &nominal_nav_;
    session.interval_cache = &interval_cache_;
    return BuildAndSolveFactorGraph(session);
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
        control_points_.size() != delta_ba_nodes_.size()) {
        LOG(ERROR) << "Node arrays are inconsistent with control-point count during error-state injection";
        return false;
    }

    double max_delta_theta_norm = 0.0;
    double max_delta_vel_norm = 0.0;
    double max_delta_pos_norm = 0.0;
    double max_delta_bg_norm = 0.0;
    double max_delta_ba_norm = 0.0;

    for (auto& nominal_state : nominal_nav_) {
        const auto delta_theta = EvaluateNodeValueAtTime(nominal_state.time, delta_theta_nodes_);
        const auto delta_vel = EvaluateNodeValueAtTime(nominal_state.time, delta_vel_nodes_);
        const auto delta_pos = EvaluateNodeValueAtTime(nominal_state.time, delta_pos_nodes_);
        const auto delta_bg = EvaluateNodeValueAtTime(nominal_state.time, delta_bg_nodes_);
        const auto delta_ba = EvaluateNodeValueAtTime(nominal_state.time, delta_ba_nodes_);
        if (!delta_theta || !delta_vel || !delta_pos || !delta_bg || !delta_ba) {
            continue;
        }

        const Sophus::SO3d nominal_rot(nominal_state.q_nb);
        nominal_state.q_nb = (nominal_rot * Sophus::SO3d::exp(*delta_theta)).unit_quaternion();
        nominal_state.vel_ned += *delta_vel;
        const Vector3d nominal_local_ned = Earth::GlobalToLocal(origin_blh_, nominal_state.blh);
        nominal_state.blh = Earth::LocalToGlobal(origin_blh_, nominal_local_ned + *delta_pos);
        nominal_state.bg += *delta_bg;
        nominal_state.ba += *delta_ba;

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

    try {
        BuildIntervalPropagationCache(
            imu_,
            nominal_nav_,
            control_points_,
            config_.imu_sigma_gyro_rps,
            config_.imu_sigma_accel_mps2,
            config_.gyro_bias_rw_sigma,
            config_.accel_bias_rw_sigma,
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
    return true;
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
    if (!control_points_.empty() &&
        control_points_.size() == delta_bg_nodes_.size() &&
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

    if (!control_points_.empty() &&
        control_points_.size() == delta_bg_nodes_.size() &&
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
    summary_ofs << "output_query_dt_s: " << config_.output_query_dt_s << '\n';
    summary_ofs << "gnss_count: " << gnss_.size() << '\n';
    summary_ofs << "imu_count: " << imu_.size() << '\n';
    summary_ofs << "control_point_count: " << control_points_.size() << '\n';
    summary_ofs << "outer_iterations: " << config_.outer_iterations << '\n';
    summary_ofs << "enable_initial_yaw_feedback: " << config_.enable_initial_yaw_feedback << '\n';
    summary_ofs << "initial_yaw_feedback_applied: " << initial_yaw_feedback_applied_ << '\n';
    summary_ofs << "initial_yaw_feedback_total_rad: " << initial_yaw_feedback_total_rad_ << '\n';
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
