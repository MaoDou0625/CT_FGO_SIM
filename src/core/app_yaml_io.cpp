#ifdef _MSC_VER
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#endif

#include "ct_fgo_sim/core/app_yaml_io.h"

#include "ct_fgo_sim/core/factor_graph_backend.h"
#include "ct_fgo_sim/core/system.h"
#include "ct_fgo_sim/io/text_measurement_io.h"

#include <glog/logging.h>
#include <yaml-cpp/yaml.h>

#include <cctype>
#include <cmath>
#include <algorithm>
#include <filesystem>

namespace ct_fgo_sim {

namespace {

constexpr double kDegToRad = M_PI / 180.0;

std::vector<TimeRange> ParseRtkOutageRanges(const YAML::Node& ranges_node) {
    std::vector<TimeRange> ranges;
    if (!ranges_node || !ranges_node.IsSequence()) {
        return ranges;
    }
    for (const auto& item : ranges_node) {
        if (!item.IsSequence() || item.size() != 2) {
            continue;
        }
        const double t0 = item[0].as<double>();
        const double t1 = item[1].as<double>();
        if (!std::isfinite(t0) || !std::isfinite(t1)) {
            continue;
        }
        ranges.push_back(TimeRange{std::min(t0, t1), std::max(t0, t1)});
    }
    std::sort(
        ranges.begin(),
        ranges.end(),
        [](const TimeRange& lhs, const TimeRange& rhs) { return lhs.start_time < rhs.start_time; });
    return ranges;
}

}  // namespace

bool LoadAppConfigYaml(
    const std::filesystem::path& config_path,
    AppConfig& config,
    ImuExtrinsicDefaults& imu_extrinsics) {
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
    config.gnss_file = gnss_path.lexically_normal().string();

    if (cfg["outputpath"]) {
        std::filesystem::path output_path = cfg["outputpath"].as<std::string>();
        if (output_path.is_relative()) {
            output_path = config_dir / output_path;
        }
        config.output_path = output_path.lexically_normal();
    } else {
        config.output_path = (config_dir / "../output").lexically_normal();
    }

    if (cfg["kf_interval_sec"]) {
        config.spline_dt_s = cfg["kf_interval_sec"].as<double>();
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
    if (cfg["gnss_sigma_horizontal_m"]) {
        config.gnss_sigma_horizontal_m = cfg["gnss_sigma_horizontal_m"].as<double>();
    }
    if (cfg["gnss_sigma_vertical_m"]) {
        config.gnss_sigma_vertical_m = cfg["gnss_sigma_vertical_m"].as<double>();
    }
    if (cfg["imu_sigma_accel_mps2"]) {
        config.imu_sigma_accel_mps2 = cfg["imu_sigma_accel_mps2"].as<double>();
    }
    if (cfg["imu_sigma_gyro_rps"]) {
        config.imu_sigma_gyro_rps = cfg["imu_sigma_gyro_rps"].as<double>();
    }
    if (cfg["gyro_bias_rw_sigma"]) {
        config.gyro_bias_rw_sigma = cfg["gyro_bias_rw_sigma"].as<double>();
    }
    if (cfg["accel_bias_rw_sigma"]) {
        config.accel_bias_rw_sigma = cfg["accel_bias_rw_sigma"].as<double>();
    }
    if (cfg["gyro_scale_rw_sigma"]) {
        config.gyro_scale_rw_sigma = cfg["gyro_scale_rw_sigma"].as<double>();
    }
    if (cfg["accel_scale_rw_sigma"]) {
        config.accel_scale_rw_sigma = cfg["accel_scale_rw_sigma"].as<double>();
    }
    if (cfg["bias_tau_s"]) {
        config.bias_tau_s = cfg["bias_tau_s"].as<double>();
    }
    if (cfg["initial_yaw_feedback"]) {
        const YAML::Node yaw_feedback = cfg["initial_yaw_feedback"];
        if (yaw_feedback["enable"]) {
            config.enable_initial_yaw_feedback = yaw_feedback["enable"].as<bool>();
        }
        if (yaw_feedback["window_s"]) {
            config.initial_yaw_feedback_window_s = yaw_feedback["window_s"].as<double>();
        }
        if (yaw_feedback["min_speed_mps"]) {
            config.initial_yaw_feedback_min_speed_mps = yaw_feedback["min_speed_mps"].as<double>();
        }
        if (yaw_feedback["min_pairs"]) {
            config.initial_yaw_feedback_min_pairs = std::max(1, yaw_feedback["min_pairs"].as<int>());
        }
        if (yaw_feedback["max_abs_deg"]) {
            config.initial_yaw_feedback_max_abs_rad = yaw_feedback["max_abs_deg"].as<double>() * kDegToRad;
        }
    }
    if (cfg["yaw_bias"]) {
        const YAML::Node yaw_bias = cfg["yaw_bias"];
        if (yaw_bias["enable"]) {
            config.yaw_bias_enable = yaw_bias["enable"].as<bool>();
        }
        if (yaw_bias["prior_sigma_deg"]) {
            config.yaw_bias_prior_sigma_rad = yaw_bias["prior_sigma_deg"].as<double>() * kDegToRad;
        }
        if (yaw_bias["window_step_limit_deg"]) {
            config.yaw_bias_window_step_limit_rad = yaw_bias["window_step_limit_deg"].as<double>() * kDegToRad;
        }
        if (yaw_bias["max_abs_deg"]) {
            config.yaw_bias_max_abs_rad = yaw_bias["max_abs_deg"].as<double>() * kDegToRad;
        }
    }
    if (cfg["imu_stride"]) {
        config.imu_stride = std::max(1, cfg["imu_stride"].as<int>());
    }
    if (cfg["outer_iterations"]) {
        config.outer_iterations = std::max(1, cfg["outer_iterations"].as<int>());
    }
    if (cfg["solver_max_iterations"]) {
        config.solver_max_iterations = std::max(1, cfg["solver_max_iterations"].as<int>());
    }
    if (cfg["use_gnss_factors"]) {
        config.use_gnss_factors = cfg["use_gnss_factors"].as<bool>();
    }
    if (cfg["use_imu_factors"]) {
        config.use_imu_factors = cfg["use_imu_factors"].as<bool>();
    }
    if (cfg["backend"]) {
        config.graph_backend = ParseGraphBackend(cfg["backend"].as<std::string>());
    }
    if (cfg["gtsam_allow_ceres_fallback"]) {
        config.gtsam_allow_ceres_fallback = cfg["gtsam_allow_ceres_fallback"].as<bool>();
    }
    if (cfg["gtsam_verbose_optimizer"]) {
        config.gtsam_verbose_optimizer = cfg["gtsam_verbose_optimizer"].as<bool>();
    }
    if (cfg["sliding_window"]) {
        const YAML::Node sw = cfg["sliding_window"];
        if (sw["enable"]) {
            config.sliding_window_enabled = sw["enable"].as<bool>();
        }
        if (sw["causal"]) {
            config.sliding_window_causal = sw["causal"].as<bool>();
        }
        if (sw["knots"]) {
            config.sliding_window_knots = std::max(3, sw["knots"].as<int>());
        }
        if (sw["step_knots"]) {
            config.sliding_window_step_knots = std::max(1, sw["step_knots"].as<int>());
        }
        if (sw["solver_max_iterations"]) {
            config.solver_max_iterations_window = std::max(1, sw["solver_max_iterations"].as<int>());
        }
        if (sw["marginalization"]) {
            config.sliding_window_marginalization = sw["marginalization"].as<bool>();
        }
        if (sw["log_timing"]) {
            config.sliding_window_log_timing = sw["log_timing"].as<bool>();
        }
        if (sw["function_tolerance"]) {
            config.sliding_window_function_tolerance = sw["function_tolerance"].as<double>();
        }
        if (sw["gradient_tolerance"]) {
            config.sliding_window_gradient_tolerance = sw["gradient_tolerance"].as<double>();
        }
        if (sw["adaptive_solver_iterations"]) {
            config.sliding_window_adaptive_solver_iterations = sw["adaptive_solver_iterations"].as<bool>();
        }
        if (sw["adaptive_solver_min_iterations"]) {
            config.sliding_window_adaptive_solver_min_iterations =
                std::max(1, sw["adaptive_solver_min_iterations"].as<int>());
        }
    }
    if (cfg["rtk_outage"]) {
        const YAML::Node outage = cfg["rtk_outage"];
        if (outage["ranges"]) {
            config.rtk_outage_ranges = ParseRtkOutageRanges(outage["ranges"]);
        }
        if (outage["recovery_horizon_s"]) {
            config.rtk_recovery_horizon_s = std::max(0.0, outage["recovery_horizon_s"].as<double>());
        }
        if (outage["freeze_imu_error_params_in_outage"]) {
            config.freeze_imu_error_params_in_outage = outage["freeze_imu_error_params_in_outage"].as<bool>();
        }
        if (outage["retro_opt_mode"]) {
            config.retro_opt_mode = outage["retro_opt_mode"].as<std::string>();
            std::transform(
                config.retro_opt_mode.begin(),
                config.retro_opt_mode.end(),
                config.retro_opt_mode.begin(),
                [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        }
    }
    if (cfg["output_query_dt_s"]) {
        config.output_query_dt_s = cfg["output_query_dt_s"].as<double>();
    }
    if (cfg["initpos"] && cfg["initvel"] && cfg["initatt"]) {
        const auto initpos = cfg["initpos"].as<std::vector<double>>();
        const auto initvel = cfg["initvel"].as<std::vector<double>>();
        const auto initatt = cfg["initatt"].as<std::vector<double>>();
        if (initpos.size() == 3 && initvel.size() == 3 && initatt.size() == 3) {
            config.use_explicit_init_state = true;
            config.init_pos_blh = Vector3d(initpos[0] * kDegToRad, initpos[1] * kDegToRad, initpos[2]);
            config.init_vel_ned = Vector3d(initvel[0], initvel[1], initvel[2]);
            config.init_att_rpy_rad =
                Vector3d(initatt[0] * kDegToRad, initatt[1] * kDegToRad, initatt[2] * kDegToRad);
        }
    }
    if (cfg["initgyrbias"]) {
        const auto initbg = cfg["initgyrbias"].as<std::vector<double>>();
        if (initbg.size() == 3) {
            config.init_bg_rps = Vector3d(initbg[0], initbg[1], initbg[2]) * (kDegToRad / 3600.0);
        }
    }
    if (cfg["initaccbias"]) {
        const auto initba = cfg["initaccbias"].as<std::vector<double>>();
        if (initba.size() == 3) {
            config.init_ba_mps2 = Vector3d(initba[0], initba[1], initba[2]) * 1.0e-5;
        }
    }
    if (cfg["initgyrscale"]) {
        const auto initsg = cfg["initgyrscale"].as<std::vector<double>>();
        if (initsg.size() == 3) {
            config.init_sg = Vector3d(initsg[0], initsg[1], initsg[2]);
        }
    }
    if (cfg["initaccscale"]) {
        const auto initsa = cfg["initaccscale"].as<std::vector<double>>();
        if (initsa.size() == 3) {
            config.init_sa = Vector3d(initsa[0], initsa[1], initsa[2]);
        }
    }
    if (cfg["body_frame"]) {
        const YAML::Node body = cfg["body_frame"];
        if (body["q_body_imu_xyzw"]) {
            const auto v = body["q_body_imu_xyzw"].as<std::vector<double>>();
            if (v.size() == 4) {
                config.body_frame.q_body_imu = Eigen::Quaterniond(v[3], v[0], v[1], v[2]).normalized();
            }
        }
        if (body["q_body_imu_prior_sigma_rad"]) {
            config.body_frame.q_body_imu_prior_sigma_rad = body["q_body_imu_prior_sigma_rad"].as<double>();
        }
        if (body["nhc_file"]) {
            std::filesystem::path nhc_path = body["nhc_file"].as<std::string>();
            if (nhc_path.is_relative()) {
                nhc_path = config_dir / nhc_path;
            }
            config.body_frame.nhc_file = nhc_path.lexically_normal().string();
        }
        if (body["enable_nhc"]) {
            config.body_frame.enable_nhc = body["enable_nhc"].as<bool>();
        }
        if (body["estimate_q_body_imu"]) {
            config.body_frame.estimate_q_body_imu = body["estimate_q_body_imu"].as<bool>();
        }
        if (body["nhc_enable_vx"]) {
            config.body_frame.nhc_enable_vx = body["nhc_enable_vx"].as<bool>();
        }
        if (body["nhc_enable_vy"]) {
            config.body_frame.nhc_enable_vy = body["nhc_enable_vy"].as<bool>();
        }
        if (body["nhc_enable_vz"]) {
            config.body_frame.nhc_enable_vz = body["nhc_enable_vz"].as<bool>();
        }
        if (body["nhc_target_vx_mps"]) {
            config.body_frame.nhc_target_vx_mps = body["nhc_target_vx_mps"].as<double>();
        }
        if (body["nhc_target_vy_mps"]) {
            config.body_frame.nhc_target_vy_mps = body["nhc_target_vy_mps"].as<double>();
        }
        if (body["nhc_target_vz_mps"]) {
            config.body_frame.nhc_target_vz_mps = body["nhc_target_vz_mps"].as<double>();
        }
        if (body["nhc_sigma_vx_mps"]) {
            config.body_frame.nhc_sigma_vx_mps = body["nhc_sigma_vx_mps"].as<double>();
        }
        if (body["nhc_sigma_vy_mps"]) {
            config.body_frame.nhc_sigma_vy_mps = body["nhc_sigma_vy_mps"].as<double>();
        }
        if (body["nhc_sigma_vz_mps"]) {
            config.body_frame.nhc_sigma_vz_mps = body["nhc_sigma_vz_mps"].as<double>();
        }
    }

    const YAML::Node imu = cfg["imu_main"];
    std::filesystem::path imu_path = imu["file"].as<std::string>();
    if (imu_path.is_relative()) {
        imu_path = config_dir / imu_path;
    }
    config.imu_main.file = imu_path.lexically_normal().string();
    config.imu_main.columns = imu["columns"] ? imu["columns"].as<int>() : 7;
    config.imu_main.rate_hz = imu["rate_hz"] ? imu["rate_hz"].as<double>() : 0.0;
    if (imu["values_are_increments"]) {
        config.imu_main.values_are_increments = imu["values_are_increments"].as<bool>();
    }
    if (imu["antlever"]) {
        const auto v = imu["antlever"].as<std::vector<double>>();
        if (v.size() == 3) {
            config.imu_main.antlever = Vector3d(v[0], v[1], v[2]);
        }
    }

    imu_extrinsics.lever_arm = config.imu_main.antlever;
    imu_extrinsics.q_body_imu = config.body_frame.q_body_imu;
    return true;
}

bool LoadMeasurementBundle(
    AppConfig& config,
    GnssMeasurementArray& gnss,
    ImuMeasurementArray& imu,
    NhcMeasurementArray& nhc,
    bool fill_default_time_window_if_zero) {
    gnss = io::LoadGnssFile(config.gnss_file);
    imu = io::LoadImuFile(config.imu_main.file, config.imu_main.values_are_increments);
    if (config.body_frame.enable_nhc && !config.body_frame.nhc_file.empty()) {
        nhc = io::LoadNhcFile(config.body_frame.nhc_file);
    } else {
        nhc.clear();
    }
    if (gnss.empty()) {
        LOG(ERROR) << "No GNSS measurements loaded from " << config.gnss_file;
        return false;
    }
    if (imu.empty()) {
        LOG(ERROR) << "No IMU measurements loaded from " << config.imu_main.file;
        return false;
    }
    if (fill_default_time_window_if_zero && config.start_time == 0.0 && config.end_time == 0.0) {
        config.start_time = std::max(gnss.front().time, imu.front().time);
        config.end_time = std::min(gnss.back().time, imu.back().time);
    }
    return true;
}

void TrimNavMeasurementsToConfigWindow(
    const AppConfig& config,
    GnssMeasurementArray& gnss,
    ImuMeasurementArray& imu,
    NhcMeasurementArray& nhc) {
    auto in_window = [&](double t) { return t >= config.start_time && t <= config.end_time; };
    gnss.erase(
        std::remove_if(gnss.begin(), gnss.end(), [&](const GnssMeasurement& m) { return !in_window(m.time); }),
        gnss.end());
    imu.erase(
        std::remove_if(imu.begin(), imu.end(), [&](const ImuMeasurement& m) { return !in_window(m.time); }),
        imu.end());
    nhc.erase(
        std::remove_if(nhc.begin(), nhc.end(), [&](const NhcMeasurement& m) { return !in_window(m.time); }),
        nhc.end());
}

}  // namespace ct_fgo_sim
