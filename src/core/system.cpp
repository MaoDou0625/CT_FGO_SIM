#include "ct_fgo_sim/core/system.h"

#include "ct_fgo_sim/factors/bias_random_walk_factor.h"
#include "ct_fgo_sim/factors/continuous_gnss_factor.h"
#include "ct_fgo_sim/factors/continuous_inertial_factor.h"
#include "ct_fgo_sim/factors/error_state_gnss_factor.h"
#include "ct_fgo_sim/factors/error_state_interval_factor.h"
#include "ct_fgo_sim/factors/error_state_nhc_factor.h"
#include "ct_fgo_sim/factors/quaternion_prior_factor.h"
#include "ct_fgo_sim/factors/road_profile_factor.h"
#include "ct_fgo_sim/factors/vertical_profile_factor.h"
#include "ct_fgo_sim/io/text_measurement_io.h"

#include <ceres/ceres.h>
#include <glog/logging.h>
#include <sophus/so3.hpp>
#include <sophus/ceres_manifold.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
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
    std::vector<double> knot_times;
    for (double t = start_time; t < end_time - 1.0e-9; t += knot_dt_s) {
        const auto upper = std::lower_bound(
            nominal_nav.begin(),
            nominal_nav.end(),
            t,
            [](const NominalNavState& state, double time) { return state.time < time; });
        double snapped_time = t;
        if (upper == nominal_nav.begin()) {
            snapped_time = nominal_nav.front().time;
        } else if (upper == nominal_nav.end()) {
            snapped_time = nominal_nav.back().time;
        } else {
            const double t1 = upper->time;
            const double t0 = (upper - 1)->time;
            snapped_time = (std::abs(t1 - t) < std::abs(t - t0)) ? t1 : t0;
        }
        if (knot_times.empty() || std::abs(knot_times.back() - snapped_time) > 1.0e-9) {
            knot_times.push_back(snapped_time);
        }
    }
    if (knot_times.empty() || std::abs(knot_times.back() - nominal_nav.back().time) > 1.0e-9) {
        knot_times.push_back(nominal_nav.back().time);
    }
    if (knot_times.size() == 1 && nominal_nav.size() >= 2) {
        knot_times.push_back(nominal_nav.back().time);
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

double InterpolateScalarNodeValue(
    double time,
    const spline::ControlPointArray& control_points,
    const std::vector<double>& nodes) {
    if (control_points.empty() || nodes.empty() || control_points.size() != nodes.size()) {
        return 0.0;
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

std::vector<double> BuildNominalDistanceAxis(
    const NominalNavStates& nominal_nav,
    const Vector3d& origin_blh) {
    std::vector<double> s_axis;
    if (nominal_nav.empty()) {
        return s_axis;
    }
    s_axis.resize(nominal_nav.size(), 0.0);
    if (nominal_nav.size() == 1) {
        return s_axis;
    }
    Vector3d prev_local_ned = Earth::GlobalToLocal(origin_blh, nominal_nav.front().blh);
    for (size_t i = 1; i < nominal_nav.size(); ++i) {
        const Vector3d local_ned = Earth::GlobalToLocal(origin_blh, nominal_nav[i].blh);
        const Eigen::Vector2d dn = (local_ned - prev_local_ned).head<2>();
        s_axis[i] = s_axis[i - 1] + dn.norm();
        prev_local_ned = local_ned;
    }
    return s_axis;
}

double InterpolateScalarSeries(
    double x,
    const std::vector<double>& x_axis,
    const std::vector<double>& y_axis) {
    if (x_axis.empty() || y_axis.empty() || x_axis.size() != y_axis.size()) {
        return 0.0;
    }
    if (x <= x_axis.front()) {
        return y_axis.front();
    }
    if (x >= x_axis.back()) {
        return y_axis.back();
    }
    const auto upper = std::lower_bound(x_axis.begin(), x_axis.end(), x);
    const size_t j = static_cast<size_t>(std::distance(x_axis.begin(), upper));
    const size_t i = j - 1;
    const double dx = x_axis[j] - x_axis[i];
    if (dx <= 1.0e-12) {
        return y_axis[i];
    }
    const double u = std::clamp((x - x_axis[i]) / dx, 0.0, 1.0);
    return y_axis[i] * (1.0 - u) + y_axis[j] * u;
}

bool FindScalarSeriesInterval(
    double x,
    const std::vector<double>& x_axis,
    int& start,
    double& u) {
    start = -1;
    u = 0.0;
    if (x_axis.size() < 2) {
        return false;
    }
    if (x <= x_axis.front()) {
        start = 0;
        u = 0.0;
        return true;
    }
    if (x >= x_axis.back()) {
        start = static_cast<int>(x_axis.size()) - 2;
        u = 1.0;
        return true;
    }
    const auto upper = std::lower_bound(x_axis.begin(), x_axis.end(), x);
    const int j = static_cast<int>(std::distance(x_axis.begin(), upper));
    const int i = j - 1;
    const double dx = x_axis[j] - x_axis[i];
    if (dx <= 1.0e-12) {
        return false;
    }
    start = i;
    u = std::clamp((x - x_axis[i]) / dx, 0.0, 1.0);
    return true;
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
    if (cfg["gnss_vertical_cauchy_scale_m"]) {
        config_.gnss_vertical_cauchy_scale_m = cfg["gnss_vertical_cauchy_scale_m"].as<double>();
    }
    if (cfg["enable_vertical_profile_field"]) {
        config_.enable_vertical_profile_field = cfg["enable_vertical_profile_field"].as<bool>();
    }
    if (cfg["vertical_gnss_sigma_m"]) {
        config_.vertical_gnss_sigma_m = cfg["vertical_gnss_sigma_m"].as<double>();
    }
    if (cfg["vertical_gnss_cauchy_scale_m"]) {
        config_.vertical_gnss_cauchy_scale_m = cfg["vertical_gnss_cauchy_scale_m"].as<double>();
    }
    if (cfg["vertical_smooth_sigma_m"]) {
        config_.vertical_smooth_sigma_m = cfg["vertical_smooth_sigma_m"].as<double>();
    }
    if (cfg["vertical_prior_sigma_m"]) {
        config_.vertical_prior_sigma_m = cfg["vertical_prior_sigma_m"].as<double>();
    }
    if (cfg["enable_road_profile_state"]) {
        config_.enable_road_profile_state = cfg["enable_road_profile_state"].as<bool>();
    }
    if (cfg["road_profile_ds_m"]) {
        config_.road_profile_ds_m = cfg["road_profile_ds_m"].as<double>();
    }
    if (cfg["road_profile_prior_sigma_m"]) {
        config_.road_profile_prior_sigma_m = cfg["road_profile_prior_sigma_m"].as<double>();
    }
    if (cfg["road_profile_curvature_sigma_m"]) {
        config_.road_profile_curvature_sigma_m = cfg["road_profile_curvature_sigma_m"].as<double>();
    }
    if (cfg["road_profile_anchor_sigma_m"]) {
        config_.road_profile_anchor_sigma_m = cfg["road_profile_anchor_sigma_m"].as<double>();
    }
    if (cfg["road_profile_anchor_spacing_m"]) {
        config_.road_profile_anchor_spacing_m = cfg["road_profile_anchor_spacing_m"].as<double>();
    }
    if (cfg["road_profile_enable_dual_layer"]) {
        config_.road_profile_enable_dual_layer = cfg["road_profile_enable_dual_layer"].as<bool>();
    }
    if (cfg["road_profile_base_ds_m"]) {
        config_.road_profile_base_ds_m = cfg["road_profile_base_ds_m"].as<double>();
    }
    if (cfg["road_profile_base_prior_sigma_m"]) {
        config_.road_profile_base_prior_sigma_m = cfg["road_profile_base_prior_sigma_m"].as<double>();
    }
    if (cfg["road_profile_base_curvature_sigma_m"]) {
        config_.road_profile_base_curvature_sigma_m = cfg["road_profile_base_curvature_sigma_m"].as<double>();
    }
    if (cfg["road_profile_base_anchor_sigma_m"]) {
        config_.road_profile_base_anchor_sigma_m = cfg["road_profile_base_anchor_sigma_m"].as<double>();
    }
    if (cfg["road_profile_base_anchor_spacing_m"]) {
        config_.road_profile_base_anchor_spacing_m = cfg["road_profile_base_anchor_spacing_m"].as<double>();
    }
    if (cfg["road_profile_residual_ds_m"]) {
        config_.road_profile_residual_ds_m = cfg["road_profile_residual_ds_m"].as<double>();
    }
    if (cfg["road_profile_residual_prior_sigma_m"]) {
        config_.road_profile_residual_prior_sigma_m = cfg["road_profile_residual_prior_sigma_m"].as<double>();
    }
    if (cfg["road_profile_residual_curvature_sigma_m"]) {
        config_.road_profile_residual_curvature_sigma_m = cfg["road_profile_residual_curvature_sigma_m"].as<double>();
    }
    if (cfg["road_profile_residual_zero_sigma_m"]) {
        config_.road_profile_residual_zero_sigma_m = cfg["road_profile_residual_zero_sigma_m"].as<double>();
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
    if (cfg["initial_yaw_feedback"]) {
        const YAML::Node yaw_feedback = cfg["initial_yaw_feedback"];
        if (yaw_feedback["enable"]) {
            config_.enable_initial_yaw_feedback = yaw_feedback["enable"].as<bool>();
        }
        if (yaw_feedback["window_s"]) {
            config_.initial_yaw_feedback_window_s = yaw_feedback["window_s"].as<double>();
        }
        if (yaw_feedback["min_speed_mps"]) {
            config_.initial_yaw_feedback_min_speed_mps = yaw_feedback["min_speed_mps"].as<double>();
        }
        if (yaw_feedback["min_pairs"]) {
            config_.initial_yaw_feedback_min_pairs = std::max(1, yaw_feedback["min_pairs"].as<int>());
        }
        if (yaw_feedback["max_abs_deg"]) {
            config_.initial_yaw_feedback_max_abs_rad = yaw_feedback["max_abs_deg"].as<double>() * kDegToRad;
        }
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
    if (cfg["output_query_dt_s"]) {
        config_.output_query_dt_s = cfg["output_query_dt_s"].as<double>();
    }
    if (cfg["use_direct_spline_state"]) {
        config_.use_direct_spline_state = cfg["use_direct_spline_state"].as<bool>();
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
        if (body["nhc_file"]) {
            std::filesystem::path nhc_path = body["nhc_file"].as<std::string>();
            if (nhc_path.is_relative()) {
                nhc_path = config_dir / nhc_path;
            }
            config_.body_frame.nhc_file = nhc_path.lexically_normal().string();
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
        if (config_.use_direct_spline_state) {
            continue;
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
    LOG(INFO) << "Enable vertical profile field: " << (config_.enable_vertical_profile_field ? "true" : "false");
    LOG(INFO) << "Vertical GNSS sigma (m): " << config_.vertical_gnss_sigma_m;
    LOG(INFO) << "Vertical GNSS Cauchy scale (m): " << config_.vertical_gnss_cauchy_scale_m;
    LOG(INFO) << "Vertical smooth sigma (m): " << config_.vertical_smooth_sigma_m;
    LOG(INFO) << "Vertical prior sigma (m): " << config_.vertical_prior_sigma_m;
    LOG(INFO) << "Enable road profile state: " << (config_.enable_road_profile_state ? "true" : "false");
    LOG(INFO) << "Road profile ds (m): " << config_.road_profile_ds_m;
    LOG(INFO) << "Road profile prior sigma (m): " << config_.road_profile_prior_sigma_m;
    LOG(INFO) << "Road profile curvature sigma (m): " << config_.road_profile_curvature_sigma_m;
    LOG(INFO) << "Road profile anchor sigma (m): " << config_.road_profile_anchor_sigma_m;
    LOG(INFO) << "Road profile anchor spacing (m): " << config_.road_profile_anchor_spacing_m;
    LOG(INFO) << "Road profile dual-layer: " << (config_.road_profile_enable_dual_layer ? "true" : "false");
    LOG(INFO) << "Road profile base ds (m): " << config_.road_profile_base_ds_m;
    LOG(INFO) << "Road profile base prior sigma (m): " << config_.road_profile_base_prior_sigma_m;
    LOG(INFO) << "Road profile base curvature sigma (m): " << config_.road_profile_base_curvature_sigma_m;
    LOG(INFO) << "Road profile base anchor sigma (m): " << config_.road_profile_base_anchor_sigma_m;
    LOG(INFO) << "Road profile base anchor spacing (m): " << config_.road_profile_base_anchor_spacing_m;
    LOG(INFO) << "Road profile residual ds (m): " << config_.road_profile_residual_ds_m;
    LOG(INFO) << "Road profile residual prior sigma (m): " << config_.road_profile_residual_prior_sigma_m;
    LOG(INFO) << "Road profile residual curvature sigma (m): " << config_.road_profile_residual_curvature_sigma_m;
    LOG(INFO) << "Road profile residual zero sigma (m): " << config_.road_profile_residual_zero_sigma_m;
    LOG(INFO) << "IMU sigma(a/g): " << config_.imu_sigma_accel_mps2 << ", " << config_.imu_sigma_gyro_rps;
    LOG(INFO) << "IMU stride: " << config_.imu_stride;
    LOG(INFO) << "Outer iterations: " << config_.outer_iterations;
    LOG(INFO) << "Enable initial yaw feedback: " << (config_.enable_initial_yaw_feedback ? "true" : "false");
    LOG(INFO) << "Use GNSS factors: " << (config_.use_gnss_factors ? "true" : "false");
    LOG(INFO) << "Use IMU factors: " << (config_.use_imu_factors ? "true" : "false");
    LOG(INFO) << "Output query dt: " << config_.output_query_dt_s;
    LOG(INFO) << "Use direct spline state: " << (config_.use_direct_spline_state ? "true" : "false");
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
    if (config_.body_frame.enable_nhc && !config_.body_frame.nhc_file.empty()) {
        nhc_ = io::LoadNhcFile(config_.body_frame.nhc_file);
    } else {
        nhc_.clear();
    }
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
    nhc_.erase(
        std::remove_if(nhc_.begin(), nhc_.end(), [&](const NhcMeasurement& m) { return !in_window(m.time); }),
        nhc_.end());
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
    std::vector<double> new_delta_z(new_control_points.size(), 0.0);
    if (!reset_biases) {
        if (delta_theta_nodes_.size() == new_control_points.size() &&
            delta_vel_nodes_.size() == new_control_points.size() &&
            delta_pos_nodes_.size() == new_control_points.size() &&
            delta_bg_nodes_.size() == new_control_points.size() &&
            delta_ba_nodes_.size() == new_control_points.size() &&
            delta_z_nodes_.size() == new_control_points.size()) {
            new_delta_theta = delta_theta_nodes_;
            new_delta_vel = delta_vel_nodes_;
            new_delta_pos = delta_pos_nodes_;
            new_delta_bg = delta_bg_nodes_;
            new_delta_ba = delta_ba_nodes_;
            new_delta_z = delta_z_nodes_;
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
    delta_z_nodes_ = std::move(new_delta_z);
    nominal_distance_s_ = BuildNominalDistanceAxis(nominal_nav_, origin_blh_);
    ResetRoadProfileNodesFromNominalTrajectory();
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
    if (control_points_.size() < 2) {
        LOG(ERROR) << "Need at least 2 control points to build the problem";
        return false;
    }

    if (config_.use_direct_spline_state) {
        if (control_points_.size() < 4) {
            LOG(ERROR) << "Direct spline-state mode needs at least 4 control points";
            return false;
        }

        ceres::Problem problem;
        for (auto& control_point : control_points_) {
            problem.AddParameterBlock(
                control_point.PoseData(),
                Sophus::SE3d::num_parameters,
                new Sophus::Manifold<Sophus::SE3>());
        }
        for (auto& delta_bg : delta_bg_nodes_) {
            problem.AddParameterBlock(delta_bg.data(), 3);
        }
        for (auto& delta_ba : delta_ba_nodes_) {
            problem.AddParameterBlock(delta_ba.data(), 3);
        }
        if (config_.enable_vertical_profile_field) {
            for (auto& delta_z : delta_z_nodes_) {
                problem.AddParameterBlock(&delta_z, 1);
            }
        }
        problem.AddParameterBlock(lever_arm_.data(), 3);
        problem.AddParameterBlock(&time_offset_s_, 1);
        problem.AddParameterBlock(q_body_imu_.coeffs().data(), 4, new ceres::EigenQuaternionManifold);

        problem.SetParameterBlockConstant(control_points_.front().PoseData());
        problem.SetParameterBlockConstant(delta_bg_nodes_.front().data());
        problem.SetParameterBlockConstant(delta_ba_nodes_.front().data());
        if (config_.enable_vertical_profile_field && !delta_z_nodes_.empty()) {
            problem.SetParameterBlockConstant(&delta_z_nodes_.front());
        }
        problem.SetParameterBlockConstant(lever_arm_.data());
        problem.SetParameterBlockConstant(&time_offset_s_);
        problem.SetParameterBlockConstant(q_body_imu_.coeffs().data());

        int gnss_factor_count = 0;
        if (config_.use_gnss_factors) {
            for (const auto& gnss : gnss_) {
                const int start = FindSplineWindowStart(control_points_, config_.spline_dt_s, gnss.time);
                if (start < 0 || start + 3 >= static_cast<int>(control_points_.size())) {
                    continue;
                }
                const double dt = control_points_[start + 1].Timestamp() - control_points_[start].Timestamp();
                if (dt <= 1.0e-9) {
                    continue;
                }
                const Vector3d meas_pos_ned = Earth::GlobalToLocal(origin_blh_, gnss.blh);
                Matrix3d sqrt_info = Matrix3d::Zero();
                sqrt_info(0, 0) = 1.0 / std::max(1.0e-6, config_.gnss_sigma_horizontal_m);
                sqrt_info(1, 1) = 1.0 / std::max(1.0e-6, config_.gnss_sigma_horizontal_m);
                sqrt_info(2, 2) = 1.0 / std::max(1.0e-6, config_.gnss_sigma_vertical_m);
                problem.AddResidualBlock(
                    factors::ContinuousGnssFactor::Create(
                        gnss.time,
                        dt,
                        control_points_[start].Timestamp(),
                        meas_pos_ned,
                        sqrt_info),
                    nullptr,
                    control_points_[start].PoseData(),
                    control_points_[start + 1].PoseData(),
                    control_points_[start + 2].PoseData(),
                    control_points_[start + 3].PoseData(),
                    lever_arm_.data());
                ++gnss_factor_count;
            }
        }

        int vertical_gnss_factor_count = 0;
        int vertical_smoothness_factor_count = 0;
        int vertical_prior_factor_count = 0;
        if (config_.enable_vertical_profile_field && config_.use_gnss_factors) {
            const double vertical_sigma = config_.vertical_gnss_sigma_m > 0.0
                ? config_.vertical_gnss_sigma_m
                : config_.gnss_sigma_vertical_m;
            const double cauchy_scale = config_.vertical_gnss_cauchy_scale_m > 0.0
                ? config_.vertical_gnss_cauchy_scale_m
                : config_.gnss_vertical_cauchy_scale_m;
            for (const auto& gnss : gnss_) {
                const int start = FindSplineWindowStart(control_points_, config_.spline_dt_s, gnss.time);
                if (start < 0 || start + 3 >= static_cast<int>(control_points_.size()) ||
                    start + 1 >= static_cast<int>(delta_z_nodes_.size())) {
                    continue;
                }
                const double dt = control_points_[start + 1].Timestamp() - control_points_[start].Timestamp();
                if (dt <= 1.0e-9) {
                    continue;
                }
                const Vector3d meas_pos_ned = Earth::GlobalToLocal(origin_blh_, gnss.blh);
                ceres::LossFunction* vertical_loss = nullptr;
                if (cauchy_scale > 0.0) {
                    vertical_loss = new ceres::CauchyLoss(cauchy_scale / std::max(1.0e-6, vertical_sigma));
                }
                problem.AddResidualBlock(
                    factors::ContinuousVerticalGnssFactor::Create(
                        gnss.time,
                        dt,
                        control_points_[start].Timestamp(),
                        meas_pos_ned.z(),
                        vertical_sigma),
                    vertical_loss,
                    control_points_[start].PoseData(),
                    control_points_[start + 1].PoseData(),
                    control_points_[start + 2].PoseData(),
                    control_points_[start + 3].PoseData(),
                    &delta_z_nodes_[start],
                    &delta_z_nodes_[start + 1],
                    lever_arm_.data());
                ++vertical_gnss_factor_count;
            }
            for (int i = 0; i + 1 < static_cast<int>(delta_z_nodes_.size()); ++i) {
                problem.AddResidualBlock(
                    factors::VerticalSmoothnessFactor::Create(config_.vertical_smooth_sigma_m),
                    nullptr,
                    &delta_z_nodes_[i],
                    &delta_z_nodes_[i + 1]);
                ++vertical_smoothness_factor_count;
            }
            for (size_t i = 1; i < delta_z_nodes_.size(); ++i) {
                problem.AddResidualBlock(
                    factors::VerticalZeroPriorFactor::Create(config_.vertical_prior_sigma_m),
                    nullptr,
                    &delta_z_nodes_[i]);
                ++vertical_prior_factor_count;
            }
        }

        int inertial_factor_count = 0;
        if (config_.use_imu_factors) {
            const size_t imu_stride = static_cast<size_t>(std::max(1, config_.imu_stride));
            for (size_t imu_index = 1; imu_index < imu_.size(); imu_index += imu_stride) {
                const auto& meas = imu_[imu_index];
                if (meas.dt <= 1.0e-9) {
                    continue;
                }
                const int start = FindSplineWindowStart(control_points_, config_.spline_dt_s, meas.time);
                if (start < 0 || start + 3 >= static_cast<int>(control_points_.size()) ||
                    start + 1 >= static_cast<int>(delta_bg_nodes_.size()) ||
                    start + 1 >= static_cast<int>(delta_ba_nodes_.size())) {
                    continue;
                }
                const double dt = control_points_[start + 1].Timestamp() - control_points_[start].Timestamp();
                if (dt <= 1.0e-9) {
                    continue;
                }
                const Vector3d gyro_meas = meas.dtheta / meas.dt;
                const Vector3d accel_meas = meas.dvel / meas.dt;
                problem.AddResidualBlock(
                    factors::ContinuousInertialFactor::Create(
                        meas.time,
                        accel_meas,
                        gyro_meas,
                        origin_blh_,
                        dt,
                        control_points_[start].Timestamp(),
                        config_.imu_sigma_accel_mps2,
                        config_.imu_sigma_gyro_rps),
                    nullptr,
                    control_points_[start].PoseData(),
                    control_points_[start + 1].PoseData(),
                    control_points_[start + 2].PoseData(),
                    control_points_[start + 3].PoseData(),
                    delta_bg_nodes_[start].data(),
                    delta_bg_nodes_[start + 1].data(),
                    delta_ba_nodes_[start].data(),
                    delta_ba_nodes_[start + 1].data(),
                    lever_arm_.data(),
                    &time_offset_s_);
                ++inertial_factor_count;
            }
        }

        int bias_rw_factor_count = 0;
        for (int i = 0; i + 1 < static_cast<int>(control_points_.size()); ++i) {
            const double dt = control_points_[i + 1].Timestamp() - control_points_[i].Timestamp();
            if (dt <= 1.0e-9) {
                continue;
            }
            problem.AddResidualBlock(
                factors::BiasRandomWalkFactor::Create(dt, config_.gyro_bias_rw_sigma, config_.bias_tau_s),
                nullptr,
                delta_bg_nodes_[i].data(),
                delta_bg_nodes_[i + 1].data());
            problem.AddResidualBlock(
                factors::BiasRandomWalkFactor::Create(dt, config_.accel_bias_rw_sigma, config_.bias_tau_s),
                nullptr,
                delta_ba_nodes_[i].data(),
                delta_ba_nodes_[i + 1].data());
            bias_rw_factor_count += 2;
        }

        int road_profile_gnss_factor_count = 0;
        int road_profile_base_smoothness_factor_count = 0;
        int road_profile_base_curvature_factor_count = 0;
        int road_profile_base_anchor_factor_count = 0;
        int road_profile_residual_smoothness_factor_count = 0;
        int road_profile_residual_curvature_factor_count = 0;
        int road_profile_residual_zero_factor_count = 0;
        if (config_.enable_road_profile_state && !nominal_nav_.empty() &&
            nominal_distance_s_.size() == nominal_nav_.size()) {
            std::vector<double> nav_times;
            nav_times.reserve(nominal_nav_.size());
            for (const auto& nav : nominal_nav_) {
                nav_times.push_back(nav.time);
            }

            const double profile_sigma = config_.vertical_gnss_sigma_m > 0.0
                ? config_.vertical_gnss_sigma_m
                : config_.gnss_sigma_vertical_m;
            const bool use_dual_layer =
                config_.road_profile_enable_dual_layer &&
                road_profile_base_h_nodes_.size() >= 2 &&
                road_profile_base_h_nodes_.size() == road_profile_base_s_nodes_.size() &&
                road_profile_residual_h_nodes_.size() >= 2 &&
                road_profile_residual_h_nodes_.size() == road_profile_residual_s_nodes_.size();

            if (use_dual_layer) {
                for (auto& road_h : road_profile_base_h_nodes_) {
                    problem.AddParameterBlock(&road_h, 1);
                }
                for (auto& road_h : road_profile_residual_h_nodes_) {
                    problem.AddParameterBlock(&road_h, 1);
                }
                problem.SetParameterBlockConstant(&road_profile_base_h_nodes_.front());

                for (const auto& gnss : gnss_) {
                    const double s_query = InterpolateScalarSeries(gnss.time, nav_times, nominal_distance_s_);
                    int base_start = -1;
                    int residual_start = -1;
                    double base_u = 0.0;
                    double residual_u = 0.0;
                    if (!FindScalarSeriesInterval(s_query, road_profile_base_s_nodes_, base_start, base_u) ||
                        !FindScalarSeriesInterval(s_query, road_profile_residual_s_nodes_, residual_start, residual_u)) {
                        continue;
                    }
                    const Vector3d meas_pos_ned = Earth::GlobalToLocal(origin_blh_, gnss.blh);
                    problem.AddResidualBlock(
                        factors::DualRoadProfileGnssFactor::Create(base_u, residual_u, meas_pos_ned.z(), profile_sigma),
                        nullptr,
                        &road_profile_base_h_nodes_[base_start],
                        &road_profile_base_h_nodes_[base_start + 1],
                        &road_profile_residual_h_nodes_[residual_start],
                        &road_profile_residual_h_nodes_[residual_start + 1]);
                    ++road_profile_gnss_factor_count;
                }
                for (int i = 0; i + 1 < static_cast<int>(road_profile_base_h_nodes_.size()); ++i) {
                    problem.AddResidualBlock(
                        factors::RoadProfileSmoothnessFactor::Create(config_.road_profile_base_prior_sigma_m),
                        nullptr,
                        &road_profile_base_h_nodes_[i],
                        &road_profile_base_h_nodes_[i + 1]);
                    ++road_profile_base_smoothness_factor_count;
                }
                for (int i = 1; i + 1 < static_cast<int>(road_profile_base_h_nodes_.size()); ++i) {
                    problem.AddResidualBlock(
                        factors::RoadProfileCurvatureFactor::Create(config_.road_profile_base_curvature_sigma_m),
                        nullptr,
                        &road_profile_base_h_nodes_[i - 1],
                        &road_profile_base_h_nodes_[i],
                        &road_profile_base_h_nodes_[i + 1]);
                    ++road_profile_base_curvature_factor_count;
                }
                const double base_anchor_spacing = std::max(
                    config_.road_profile_base_ds_m,
                    config_.road_profile_base_anchor_spacing_m);
                double next_base_anchor_s = base_anchor_spacing;
                for (size_t i = 1; i < road_profile_base_h_nodes_.size(); ++i) {
                    if (road_profile_base_s_nodes_[i] + 1.0e-9 < next_base_anchor_s &&
                        i + 1 < road_profile_base_h_nodes_.size()) {
                        continue;
                    }
                    const double ref_h = road_profile_base_h_nodes_[i];
                    problem.AddResidualBlock(
                        factors::RoadProfileAnchorFactor::Create(ref_h, config_.road_profile_base_anchor_sigma_m),
                        nullptr,
                        &road_profile_base_h_nodes_[i]);
                    ++road_profile_base_anchor_factor_count;
                    next_base_anchor_s += base_anchor_spacing;
                }
                for (int i = 0; i + 1 < static_cast<int>(road_profile_residual_h_nodes_.size()); ++i) {
                    problem.AddResidualBlock(
                        factors::RoadProfileSmoothnessFactor::Create(config_.road_profile_residual_prior_sigma_m),
                        nullptr,
                        &road_profile_residual_h_nodes_[i],
                        &road_profile_residual_h_nodes_[i + 1]);
                    ++road_profile_residual_smoothness_factor_count;
                }
                for (int i = 1; i + 1 < static_cast<int>(road_profile_residual_h_nodes_.size()); ++i) {
                    problem.AddResidualBlock(
                        factors::RoadProfileCurvatureFactor::Create(config_.road_profile_residual_curvature_sigma_m),
                        nullptr,
                        &road_profile_residual_h_nodes_[i - 1],
                        &road_profile_residual_h_nodes_[i],
                        &road_profile_residual_h_nodes_[i + 1]);
                    ++road_profile_residual_curvature_factor_count;
                }
                for (auto& residual_h : road_profile_residual_h_nodes_) {
                    problem.AddResidualBlock(
                        factors::RoadProfileZeroFactor::Create(config_.road_profile_residual_zero_sigma_m),
                        nullptr,
                        &residual_h);
                    ++road_profile_residual_zero_factor_count;
                }
            } else if (road_profile_h_nodes_.size() >= 2 &&
                       road_profile_h_nodes_.size() == road_profile_s_nodes_.size()) {
                for (auto& road_h : road_profile_h_nodes_) {
                    problem.AddParameterBlock(&road_h, 1);
                }
                problem.SetParameterBlockConstant(&road_profile_h_nodes_.front());

                for (const auto& gnss : gnss_) {
                    const double s_query = InterpolateScalarSeries(gnss.time, nav_times, nominal_distance_s_);
                    int start = -1;
                    double u = 0.0;
                    if (!FindScalarSeriesInterval(s_query, road_profile_s_nodes_, start, u)) {
                        continue;
                    }
                    const Vector3d meas_pos_ned = Earth::GlobalToLocal(origin_blh_, gnss.blh);
                    problem.AddResidualBlock(
                        factors::RoadProfileGnssFactor::Create(u, meas_pos_ned.z(), profile_sigma),
                        nullptr,
                        &road_profile_h_nodes_[start],
                        &road_profile_h_nodes_[start + 1]);
                    ++road_profile_gnss_factor_count;
                }
                for (int i = 0; i + 1 < static_cast<int>(road_profile_h_nodes_.size()); ++i) {
                    problem.AddResidualBlock(
                        factors::RoadProfileSmoothnessFactor::Create(config_.road_profile_prior_sigma_m),
                        nullptr,
                        &road_profile_h_nodes_[i],
                        &road_profile_h_nodes_[i + 1]);
                    ++road_profile_base_smoothness_factor_count;
                }
                for (int i = 1; i + 1 < static_cast<int>(road_profile_h_nodes_.size()); ++i) {
                    problem.AddResidualBlock(
                        factors::RoadProfileCurvatureFactor::Create(config_.road_profile_curvature_sigma_m),
                        nullptr,
                        &road_profile_h_nodes_[i - 1],
                        &road_profile_h_nodes_[i],
                        &road_profile_h_nodes_[i + 1]);
                    ++road_profile_base_curvature_factor_count;
                }
                const double anchor_spacing = std::max(config_.road_profile_ds_m, config_.road_profile_anchor_spacing_m);
                double next_anchor_s = anchor_spacing;
                for (size_t i = 1; i < road_profile_h_nodes_.size(); ++i) {
                    if (road_profile_s_nodes_[i] + 1.0e-9 < next_anchor_s && i + 1 < road_profile_h_nodes_.size()) {
                        continue;
                    }
                    const double ref_h = road_profile_h_nodes_[i];
                    problem.AddResidualBlock(
                        factors::RoadProfileAnchorFactor::Create(ref_h, config_.road_profile_anchor_sigma_m),
                        nullptr,
                        &road_profile_h_nodes_[i]);
                    ++road_profile_base_anchor_factor_count;
                    next_anchor_s += anchor_spacing;
                }
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

        LOG(INFO) << "Direct spline GNSS factors: " << gnss_factor_count;
        LOG(INFO) << "Vertical GNSS profile factors: " << vertical_gnss_factor_count;
        LOG(INFO) << "Vertical smoothness factors: " << vertical_smoothness_factor_count;
        LOG(INFO) << "Vertical prior factors: " << vertical_prior_factor_count;
        LOG(INFO) << "Road-profile GNSS factors: " << road_profile_gnss_factor_count;
        LOG(INFO) << "Road-profile base smoothness factors: " << road_profile_base_smoothness_factor_count;
        LOG(INFO) << "Road-profile base curvature factors: " << road_profile_base_curvature_factor_count;
        LOG(INFO) << "Road-profile base anchor factors: " << road_profile_base_anchor_factor_count;
        LOG(INFO) << "Road-profile residual smoothness factors: " << road_profile_residual_smoothness_factor_count;
        LOG(INFO) << "Road-profile residual curvature factors: " << road_profile_residual_curvature_factor_count;
        LOG(INFO) << "Road-profile residual zero factors: " << road_profile_residual_zero_factor_count;
        LOG(INFO) << "Direct spline inertial factors: " << inertial_factor_count;
        LOG(INFO) << "Bias random-walk factors: " << bias_rw_factor_count;
        LOG(INFO) << summary.BriefReport();
        return summary.termination_type != ceres::FAILURE;
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
    if (!(config_.body_frame.enable_nhc && config_.body_frame.estimate_q_body_imu)) {
        problem.SetParameterBlockConstant(q_body_imu_.coeffs().data());
    }

    problem.AddResidualBlock(
        factors::QuaternionPriorFactor::Create(
            config_.body_frame.q_body_imu,
            config_.body_frame.q_body_imu_prior_sigma_rad),
        nullptr,
        q_body_imu_.coeffs().data());

    int gnss_horizontal_factor_count = 0;
    int gnss_vertical_factor_count = 0;
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
            const Vector3d meas_pos_ned = Earth::GlobalToLocal(origin_blh_, gnss.blh);

            problem.AddResidualBlock(
                factors::ErrorStateGnssHorizontalLeverArmFactor::Create(
                    u,
                    nominal_pos_ned,
                    nominal_state->q_nb,
                    lever_arm_,
                    meas_pos_ned,
                    config_.gnss_sigma_horizontal_m),
                nullptr,
                delta_pos_nodes_[start].data(),
                delta_pos_nodes_[start + 1].data(),
                delta_theta_nodes_[start].data(),
                delta_theta_nodes_[start + 1].data());
            ++gnss_horizontal_factor_count;

            ceres::LossFunction* vertical_loss = nullptr;
            if (config_.gnss_vertical_cauchy_scale_m > 0.0) {
                const double whitened_scale =
                    config_.gnss_vertical_cauchy_scale_m / std::max(1.0e-6, config_.gnss_sigma_vertical_m);
                vertical_loss = new ceres::CauchyLoss(whitened_scale);
            }
            problem.AddResidualBlock(
                factors::ErrorStateGnssVerticalLeverArmFactor::Create(
                    u,
                    nominal_pos_ned,
                    nominal_state->q_nb,
                    lever_arm_,
                    meas_pos_ned,
                    config_.gnss_sigma_vertical_m),
                vertical_loss,
                delta_pos_nodes_[start].data(),
                delta_pos_nodes_[start + 1].data(),
                delta_theta_nodes_[start].data(),
                delta_theta_nodes_[start + 1].data());
            ++gnss_vertical_factor_count;
        }
    }

    int nhc_factor_count = 0;
    if (config_.body_frame.enable_nhc) {
        const bool any_axis_enabled =
            config_.body_frame.nhc_enable_vx || config_.body_frame.nhc_enable_vy || config_.body_frame.nhc_enable_vz;
        const Vector3d sigma_body_mps(
            config_.body_frame.nhc_enable_vx ? config_.body_frame.nhc_sigma_vx_mps : -1.0,
            config_.body_frame.nhc_enable_vy ? config_.body_frame.nhc_sigma_vy_mps : -1.0,
            config_.body_frame.nhc_enable_vz ? config_.body_frame.nhc_sigma_vz_mps : -1.0);
        if (any_axis_enabled) {
            const size_t nhc_stride = static_cast<size_t>(std::max(1, config_.imu_stride));
            for (size_t nhc_index = 0; nhc_index < nhc_.size(); nhc_index += nhc_stride) {
                const auto& nhc = nhc_[nhc_index];
                const int start = FindNodeIntervalStart(control_points_, nhc.time);
                if (start < 0 || start + 1 >= static_cast<int>(control_points_.size())) {
                    continue;
                }
                const auto nominal_state = EvaluateNominalState(nominal_nav_, nhc.time);
                if (!nominal_state) {
                    continue;
                }
                const double dt = control_points_[start + 1].Timestamp() - control_points_[start].Timestamp();
                if (dt <= 1.0e-9) {
                    continue;
                }
                const double u = std::clamp((nhc.time - control_points_[start].Timestamp()) / dt, 0.0, 1.0);
                Vector3d target_vel_body(
                    config_.body_frame.nhc_target_vx_mps,
                    config_.body_frame.nhc_target_vy_mps,
                    config_.body_frame.nhc_target_vz_mps);
                if (config_.body_frame.nhc_enable_vx) {
                    target_vel_body.x() = nhc.vel_body_mps.x();
                }
                if (config_.body_frame.nhc_enable_vy) {
                    target_vel_body.y() = nhc.vel_body_mps.y();
                }
                if (config_.body_frame.nhc_enable_vz) {
                    target_vel_body.z() = nhc.vel_body_mps.z();
                }

                problem.AddResidualBlock(
                    factors::ErrorStateBodyVelocityNhcFactor::Create(
                        u,
                        nominal_state->q_nb,
                        nominal_state->vel_ned,
                        target_vel_body,
                        sigma_body_mps),
                    nullptr,
                    delta_theta_nodes_[start].data(),
                    delta_theta_nodes_[start + 1].data(),
                    delta_vel_nodes_[start].data(),
                    delta_vel_nodes_[start + 1].data(),
                    q_body_imu_.coeffs().data());
                ++nhc_factor_count;
            }
        }
    }

    int process_factor_count = 0;
    if (config_.use_imu_factors) {
        for (int i = 0; i + 1 < static_cast<int>(control_points_.size()); ++i) {
            if (i >= static_cast<int>(interval_cache_.knot_intervals.size())) {
                continue;
            }
            const auto& knot_interval = interval_cache_.knot_intervals[static_cast<size_t>(i)];
            if (!knot_interval.valid) {
                continue;
            }
            problem.AddResidualBlock(
                factors::ErrorStateIntervalFactor::Create(
                    knot_interval.phi,
                    knot_interval.sqrt_info),
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

    LOG(INFO) << "GNSS factors (horizontal / vertical): "
              << gnss_horizontal_factor_count << " / " << gnss_vertical_factor_count;
    LOG(INFO) << "NHC factors: " << nhc_factor_count;
    LOG(INFO) << "Interval propagation factors: " << process_factor_count;
    LOG(INFO) << summary.BriefReport();
    return summary.termination_type != ceres::FAILURE;
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

    if (config_.use_direct_spline_state && control_points_.size() >= 4) {
        const int start = FindSplineWindowStart(control_points_, config_.spline_dt_s, time);
        if (start < 0 || start + 3 >= static_cast<int>(control_points_.size())) {
            return std::nullopt;
        }
        const double dt = control_points_[start + 1].Timestamp() - control_points_[start].Timestamp();
        if (dt <= 1.0e-9) {
            return std::nullopt;
        }
        const double u = std::clamp((time - control_points_[start].Timestamp()) / dt, 0.0, 1.0);
        const auto result = spline::BSplineEvaluator::Evaluate(
            u,
            dt,
            control_points_[start].Pose(),
            control_points_[start + 1].Pose(),
            control_points_[start + 2].Pose(),
            control_points_[start + 3].Pose());

        ComposedState composed;
        composed.time = time;
        composed.nominal = *nominal_state;
        composed.full_pose = result.pose;
        composed.full_vel_ned = result.v_world;
        composed.full_vel_body = result.v_body;
        composed.full_accel_ned = result.a_world;
        composed.full_omega_body = result.w_body;
        composed.full_alpha_body = result.alpha_body;
        composed.base_vertical_ned_m = result.pose.translation().z();

        if (config_.enable_vertical_profile_field && delta_z_nodes_.size() == control_points_.size()) {
            composed.vertical_profile_correction_m =
                InterpolateScalarNodeValue(time, control_points_, delta_z_nodes_);
            Vector3d t_with_vertical = composed.full_pose.translation();
            t_with_vertical.z() += composed.vertical_profile_correction_m;
            composed.full_pose = Sophus::SE3d(composed.full_pose.so3(), t_with_vertical);
        }
        const bool has_single_layer_profile =
            road_profile_h_nodes_.size() >= 2 && road_profile_h_nodes_.size() == road_profile_s_nodes_.size();
        const bool has_dual_layer_profile =
            config_.road_profile_enable_dual_layer &&
            road_profile_base_h_nodes_.size() >= 2 &&
            road_profile_base_h_nodes_.size() == road_profile_base_s_nodes_.size() &&
            road_profile_residual_h_nodes_.size() >= 2 &&
            road_profile_residual_h_nodes_.size() == road_profile_residual_s_nodes_.size();
        if (config_.enable_road_profile_state && (has_single_layer_profile || has_dual_layer_profile) &&
            nominal_distance_s_.size() == nominal_nav_.size() && !nominal_nav_.empty()) {
            std::vector<double> nav_times;
            nav_times.reserve(nominal_nav_.size());
            for (const auto& nav : nominal_nav_) {
                nav_times.push_back(nav.time);
            }
            const double s_query = InterpolateScalarSeries(time, nav_times, nominal_distance_s_);
            const double road_h_ned = EvaluateRoadProfileHeightAtDistance(s_query);
            Vector3d t_with_road = composed.full_pose.translation();
            t_with_road.z() = road_h_ned;
            composed.full_pose = Sophus::SE3d(composed.full_pose.so3(), t_with_road);
        }

        if (const auto delta_bg = EvaluateNodeValueAtTime(time, delta_bg_nodes_)) {
            composed.full_bg = initial_alignment_.bg0 + *delta_bg;
            composed.full_omega_body += composed.full_bg;
        } else {
            composed.full_bg = initial_alignment_.bg0;
            composed.full_omega_body += composed.full_bg;
        }
        if (const auto delta_ba = EvaluateNodeValueAtTime(time, delta_ba_nodes_)) {
            composed.full_ba = initial_alignment_.ba0 + *delta_ba;
        } else {
            composed.full_ba = initial_alignment_.ba0;
        }
        return composed;
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

void System::ResetRoadProfileNodesFromNominalTrajectory() {
    road_profile_s_nodes_.clear();
    road_profile_h_nodes_.clear();
    road_profile_base_s_nodes_.clear();
    road_profile_base_h_nodes_.clear();
    road_profile_residual_s_nodes_.clear();
    road_profile_residual_h_nodes_.clear();
    if (!config_.enable_road_profile_state || nominal_nav_.empty()) {
        return;
    }
    if (nominal_distance_s_.size() != nominal_nav_.size()) {
        nominal_distance_s_ = BuildNominalDistanceAxis(nominal_nav_, origin_blh_);
    }
    if (nominal_distance_s_.empty()) {
        return;
    }

    std::vector<double> nominal_h_ned;
    nominal_h_ned.reserve(nominal_nav_.size());
    for (const auto& nav : nominal_nav_) {
        nominal_h_ned.push_back(Earth::GlobalToLocal(origin_blh_, nav.blh).z());
    }

    const double s_end = nominal_distance_s_.back();
    if (config_.road_profile_enable_dual_layer) {
        const double base_ds = std::max(0.10, config_.road_profile_base_ds_m);
        const double residual_ds = std::max(0.05, config_.road_profile_residual_ds_m);
        for (double s = 0.0; s < s_end - 1.0e-9; s += base_ds) {
            road_profile_base_s_nodes_.push_back(s);
            road_profile_base_h_nodes_.push_back(
                InterpolateScalarSeries(s, nominal_distance_s_, nominal_h_ned));
        }
        road_profile_base_s_nodes_.push_back(s_end);
        road_profile_base_h_nodes_.push_back(nominal_h_ned.back());

        for (double s = 0.0; s < s_end - 1.0e-9; s += residual_ds) {
            road_profile_residual_s_nodes_.push_back(s);
            road_profile_residual_h_nodes_.push_back(0.0);
        }
        road_profile_residual_s_nodes_.push_back(s_end);
        road_profile_residual_h_nodes_.push_back(0.0);
        return;
    }

    const double ds = std::max(0.05, config_.road_profile_ds_m);
    for (double s = 0.0; s < s_end - 1.0e-9; s += ds) {
        road_profile_s_nodes_.push_back(s);
        road_profile_h_nodes_.push_back(InterpolateScalarSeries(s, nominal_distance_s_, nominal_h_ned));
    }
    road_profile_s_nodes_.push_back(s_end);
    road_profile_h_nodes_.push_back(nominal_h_ned.back());
}

double System::EvaluateRoadProfileHeightAtDistance(double s_query) const {
    const bool use_dual_layer =
        config_.road_profile_enable_dual_layer &&
        road_profile_base_h_nodes_.size() >= 2 &&
        road_profile_base_h_nodes_.size() == road_profile_base_s_nodes_.size() &&
        road_profile_residual_h_nodes_.size() >= 2 &&
        road_profile_residual_h_nodes_.size() == road_profile_residual_s_nodes_.size();
    if (use_dual_layer) {
        return InterpolateScalarSeries(s_query, road_profile_base_s_nodes_, road_profile_base_h_nodes_) +
               InterpolateScalarSeries(s_query, road_profile_residual_s_nodes_, road_profile_residual_h_nodes_);
    }
    if (!road_profile_s_nodes_.empty() && road_profile_s_nodes_.size() == road_profile_h_nodes_.size()) {
        return InterpolateScalarSeries(s_query, road_profile_s_nodes_, road_profile_h_nodes_);
    }
    return 0.0;
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

    if (config_.enable_vertical_profile_field && delta_z_nodes_.size() == control_points_.size()) {
        const std::filesystem::path vertical_profile_path = config_.output_path / "vertical_profile_nodes.txt";
        std::ofstream vertical_profile_ofs(vertical_profile_path);
        vertical_profile_ofs << "# time_s delta_z_m\n";
        for (size_t i = 0; i < control_points_.size(); ++i) {
            vertical_profile_ofs << control_points_[i].Timestamp() << ' ' << delta_z_nodes_[i] << '\n';
        }
    }

    if (config_.enable_road_profile_state) {
        if (config_.road_profile_enable_dual_layer &&
            !road_profile_base_s_nodes_.empty() &&
            road_profile_base_s_nodes_.size() == road_profile_base_h_nodes_.size() &&
            !road_profile_residual_s_nodes_.empty() &&
            road_profile_residual_s_nodes_.size() == road_profile_residual_h_nodes_.size()) {
            const std::filesystem::path road_profile_base_path = config_.output_path / "road_profile_base_nodes.txt";
            std::ofstream road_profile_base_ofs(road_profile_base_path);
            road_profile_base_ofs << "# s_m h_base_ned_m\n";
            for (size_t i = 0; i < road_profile_base_s_nodes_.size(); ++i) {
                road_profile_base_ofs << road_profile_base_s_nodes_[i] << ' ' << road_profile_base_h_nodes_[i] << '\n';
            }

            const std::filesystem::path road_profile_residual_path =
                config_.output_path / "road_profile_residual_nodes.txt";
            std::ofstream road_profile_residual_ofs(road_profile_residual_path);
            road_profile_residual_ofs << "# s_m h_residual_m\n";
            for (size_t i = 0; i < road_profile_residual_s_nodes_.size(); ++i) {
                road_profile_residual_ofs << road_profile_residual_s_nodes_[i] << ' '
                                          << road_profile_residual_h_nodes_[i] << '\n';
            }

            const std::filesystem::path road_profile_total_path = config_.output_path / "road_profile_nodes.txt";
            std::ofstream road_profile_total_ofs(road_profile_total_path);
            road_profile_total_ofs << "# s_m h_total_ned_m\n";
            for (size_t i = 0; i < road_profile_residual_s_nodes_.size(); ++i) {
                const double s = road_profile_residual_s_nodes_[i];
                road_profile_total_ofs << s << ' ' << EvaluateRoadProfileHeightAtDistance(s) << '\n';
            }
        } else if (!road_profile_s_nodes_.empty() && road_profile_s_nodes_.size() == road_profile_h_nodes_.size()) {
            const std::filesystem::path road_profile_path = config_.output_path / "road_profile_nodes.txt";
            std::ofstream road_profile_ofs(road_profile_path);
            road_profile_ofs << "# s_m h_ned_m\n";
            for (size_t i = 0; i < road_profile_s_nodes_.size(); ++i) {
                road_profile_ofs << road_profile_s_nodes_[i] << ' ' << road_profile_h_nodes_[i] << '\n';
            }
        }
    }

    const std::filesystem::path summary_path = config_.output_path / "run_summary.txt";
    std::ofstream summary_ofs(summary_path);
    summary_ofs << std::setprecision(17);
    summary_ofs << "gnss_file: " << config_.gnss_file << '\n';
    summary_ofs << "imu_file: " << config_.imu_main.file << '\n';
    summary_ofs << "use_gnss_factors: " << config_.use_gnss_factors << '\n';
    summary_ofs << "use_imu_factors: " << config_.use_imu_factors << '\n';
    summary_ofs << "enable_vertical_profile_field: " << config_.enable_vertical_profile_field << '\n';
    summary_ofs << "vertical_gnss_sigma_m: " << config_.vertical_gnss_sigma_m << '\n';
    summary_ofs << "vertical_gnss_cauchy_scale_m: " << config_.vertical_gnss_cauchy_scale_m << '\n';
    summary_ofs << "vertical_smooth_sigma_m: " << config_.vertical_smooth_sigma_m << '\n';
    summary_ofs << "vertical_prior_sigma_m: " << config_.vertical_prior_sigma_m << '\n';
    summary_ofs << "enable_road_profile_state: " << config_.enable_road_profile_state << '\n';
    summary_ofs << "road_profile_ds_m: " << config_.road_profile_ds_m << '\n';
    summary_ofs << "road_profile_prior_sigma_m: " << config_.road_profile_prior_sigma_m << '\n';
    summary_ofs << "road_profile_curvature_sigma_m: " << config_.road_profile_curvature_sigma_m << '\n';
    summary_ofs << "road_profile_anchor_sigma_m: " << config_.road_profile_anchor_sigma_m << '\n';
    summary_ofs << "road_profile_anchor_spacing_m: " << config_.road_profile_anchor_spacing_m << '\n';
    summary_ofs << "road_profile_enable_dual_layer: " << config_.road_profile_enable_dual_layer << '\n';
    summary_ofs << "road_profile_base_ds_m: " << config_.road_profile_base_ds_m << '\n';
    summary_ofs << "road_profile_base_prior_sigma_m: " << config_.road_profile_base_prior_sigma_m << '\n';
    summary_ofs << "road_profile_base_curvature_sigma_m: " << config_.road_profile_base_curvature_sigma_m << '\n';
    summary_ofs << "road_profile_base_anchor_sigma_m: " << config_.road_profile_base_anchor_sigma_m << '\n';
    summary_ofs << "road_profile_base_anchor_spacing_m: " << config_.road_profile_base_anchor_spacing_m << '\n';
    summary_ofs << "road_profile_residual_ds_m: " << config_.road_profile_residual_ds_m << '\n';
    summary_ofs << "road_profile_residual_prior_sigma_m: " << config_.road_profile_residual_prior_sigma_m << '\n';
    summary_ofs << "road_profile_residual_curvature_sigma_m: " << config_.road_profile_residual_curvature_sigma_m << '\n';
    summary_ofs << "road_profile_residual_zero_sigma_m: " << config_.road_profile_residual_zero_sigma_m << '\n';
    summary_ofs << "output_query_dt_s: " << config_.output_query_dt_s << '\n';
    summary_ofs << "gnss_count: " << gnss_.size() << '\n';
    summary_ofs << "imu_count: " << imu_.size() << '\n';
    summary_ofs << "control_point_count: " << control_points_.size() << '\n';
    summary_ofs << "road_profile_node_count: " << road_profile_s_nodes_.size() << '\n';
    summary_ofs << "road_profile_base_node_count: " << road_profile_base_s_nodes_.size() << '\n';
    summary_ofs << "road_profile_residual_node_count: " << road_profile_residual_s_nodes_.size() << '\n';
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
