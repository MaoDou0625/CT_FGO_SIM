#include "ct_fgo_sim/core/factor_graph_session.h"

#include "ct_fgo_sim/core/marginalization_frontier.h"
#include "ct_fgo_sim/core/spline_helpers.h"
#include "ct_fgo_sim/core/system.h"
#include "ct_fgo_sim/factors/error_state_gnss_factor.h"
#include "ct_fgo_sim/factors/error_state_interval_factor.h"
#include "ct_fgo_sim/factors/error_state_nhc_factor.h"
#include "ct_fgo_sim/factors/quaternion_prior_factor.h"
#include "ct_fgo_sim/navigation/earth.h"
#include "ct_fgo_sim/navigation/mechanization.h"

#include <ceres/ceres.h>
#include <glog/logging.h>
#include <sophus/ceres_manifold.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <thread>

namespace ct_fgo_sim {

namespace {

struct YawBiasPriorFactor {
    explicit YawBiasPriorFactor(double sigma_rad)
        : inv_sigma_(1.0 / std::max(1.0e-6, sigma_rad)) {}
    template <typename T>
    bool operator()(const T* const yaw_bias, T* residuals) const {
        residuals[0] = T(inv_sigma_) * yaw_bias[0];
        return true;
    }
    double inv_sigma_ = 1.0;
};

}  // namespace

bool BuildAndSolveFactorGraph(FactorGraphSession& session) {
    if (!session.config || !session.origin_blh || !session.gnss || !session.imu || !session.nhc ||
        !session.control_points || !session.delta_theta_nodes || !session.delta_vel_nodes ||
        !session.delta_pos_nodes || !session.delta_bg_nodes || !session.delta_ba_nodes ||
        !session.delta_sg_nodes || !session.delta_sa_nodes ||
        !session.lever_arm || !session.time_offset_s || !session.q_body_imu || !session.nominal_nav ||
        !session.interval_cache || (session.config && session.config->yaw_bias_enable && !session.yaw_bias_rad)) {
        LOG(ERROR) << "BuildAndSolveFactorGraph: incomplete session";
        return false;
    }

    AppConfig& config_ = *session.config;
    Vector3d& origin_blh_ = *session.origin_blh;
    GnssMeasurementArray& gnss_ = *session.gnss;
    NhcMeasurementArray& nhc_ = *session.nhc;
    spline::ControlPointArray& control_points_ = *session.control_points;
    AlignedVec3Array& delta_theta_nodes_ = *session.delta_theta_nodes;
    AlignedVec3Array& delta_vel_nodes_ = *session.delta_vel_nodes;
    AlignedVec3Array& delta_pos_nodes_ = *session.delta_pos_nodes;
    AlignedVec3Array& delta_bg_nodes_ = *session.delta_bg_nodes;
    AlignedVec3Array& delta_ba_nodes_ = *session.delta_ba_nodes;
    AlignedVec3Array& delta_sg_nodes_ = *session.delta_sg_nodes;
    AlignedVec3Array& delta_sa_nodes_ = *session.delta_sa_nodes;
    Vector3d& lever_arm_ = *session.lever_arm;
    double& time_offset_s_ = *session.time_offset_s;
    Eigen::Quaterniond& q_body_imu_ = *session.q_body_imu;
    double* yaw_bias_rad = session.yaw_bias_rad;
    NominalNavStates& nominal_nav_ = *session.nominal_nav;
    IntervalPropagationCache& interval_cache_ = *session.interval_cache;

    if (control_points_.size() < 2) {
        LOG(ERROR) << "Need at least 2 control points to build the problem";
        return false;
    }

    const bool windowed =
        session.window_knot_lo >= 0 && session.window_knot_hi >= session.window_knot_lo;
    const int n_knots = static_cast<int>(control_points_.size());
    int k_lo = 0;
    int k_hi = n_knots - 1;
    if (windowed) {
        k_lo = std::clamp(session.window_knot_lo, 0, n_knots - 1);
        k_hi = std::clamp(session.window_knot_hi, k_lo, n_knots - 1);
        if (k_hi - k_lo < 1) {
            LOG(ERROR) << "Sliding window must span at least two knots";
            return false;
        }
    }

    ceres::Problem problem;
    for (int k = (windowed ? k_lo : 0); k <= (windowed ? k_hi : n_knots - 1); ++k) {
        problem.AddParameterBlock(delta_theta_nodes_[static_cast<size_t>(k)].data(), 3);
        problem.AddParameterBlock(delta_vel_nodes_[static_cast<size_t>(k)].data(), 3);
        problem.AddParameterBlock(delta_pos_nodes_[static_cast<size_t>(k)].data(), 3);
        problem.AddParameterBlock(delta_bg_nodes_[static_cast<size_t>(k)].data(), 3);
        problem.AddParameterBlock(delta_ba_nodes_[static_cast<size_t>(k)].data(), 3);
        problem.AddParameterBlock(delta_sg_nodes_[static_cast<size_t>(k)].data(), 3);
        problem.AddParameterBlock(delta_sa_nodes_[static_cast<size_t>(k)].data(), 3);
    }
    problem.AddParameterBlock(&time_offset_s_, 1);
    if (config_.yaw_bias_enable && yaw_bias_rad) {
        problem.AddParameterBlock(yaw_bias_rad, 1);
    }
    problem.AddParameterBlock(q_body_imu_.coeffs().data(), 4, new ceres::EigenQuaternionManifold);

    if (!windowed || k_lo == 0) {
        problem.SetParameterBlockConstant(delta_theta_nodes_.front().data());
        problem.SetParameterBlockConstant(delta_vel_nodes_.front().data());
        problem.SetParameterBlockConstant(delta_pos_nodes_.front().data());
        problem.SetParameterBlockConstant(delta_bg_nodes_.front().data());
        problem.SetParameterBlockConstant(delta_ba_nodes_.front().data());
        problem.SetParameterBlockConstant(delta_sg_nodes_.front().data());
        problem.SetParameterBlockConstant(delta_sa_nodes_.front().data());
    }
    problem.SetParameterBlockConstant(&time_offset_s_);
    if (!(config_.body_frame.enable_nhc && config_.body_frame.estimate_q_body_imu)) {
        problem.SetParameterBlockConstant(q_body_imu_.coeffs().data());
    }
    if (config_.yaw_bias_enable && yaw_bias_rad) {
        double lb = -std::numeric_limits<double>::infinity();
        double ub = std::numeric_limits<double>::infinity();
        if (config_.yaw_bias_max_abs_rad > 0.0) {
            lb = -config_.yaw_bias_max_abs_rad;
            ub = config_.yaw_bias_max_abs_rad;
        }
        if (session.has_yaw_bias_step_limit && session.yaw_bias_step_limit_rad > 0.0) {
            const double d = session.yaw_bias_step_limit_rad;
            lb = std::max(lb, session.yaw_bias_center_rad - d);
            ub = std::min(ub, session.yaw_bias_center_rad + d);
        }
        if (lb > ub) {
            const double mid = 0.5 * (lb + ub);
            lb = mid;
            ub = mid;
        }
        problem.SetParameterLowerBound(yaw_bias_rad, 0, lb);
        problem.SetParameterUpperBound(yaw_bias_rad, 0, ub);
    }

    problem.AddResidualBlock(
        factors::QuaternionPriorFactor::Create(
            config_.body_frame.q_body_imu,
            config_.body_frame.q_body_imu_prior_sigma_rad),
        nullptr,
        q_body_imu_.coeffs().data());
    if (config_.yaw_bias_enable && yaw_bias_rad) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<YawBiasPriorFactor, 1, 1>(
                new YawBiasPriorFactor(config_.yaw_bias_prior_sigma_rad)),
            nullptr,
            yaw_bias_rad);
    }

    if (session.marginalization_frontier && session.marginalization_frontier->valid &&
        session.marginalization_frontier->anchor_knot_index == k_lo && k_lo > 0) {
        ceres::CostFunction* marg_cost = CreateMarginalizationPriorCost(*session.marginalization_frontier);
        if (!marg_cost) {
            LOG(ERROR) << "Sliding window prior is marked valid but prior cost creation failed at k_lo=" << k_lo;
            return false;
        }
        const int mk = session.marginalization_frontier->anchor_knot_index;
        problem.AddResidualBlock(
            marg_cost,
            nullptr,
            delta_theta_nodes_[static_cast<size_t>(mk)].data(),
            delta_vel_nodes_[static_cast<size_t>(mk)].data(),
            delta_pos_nodes_[static_cast<size_t>(mk)].data(),
            delta_bg_nodes_[static_cast<size_t>(mk)].data(),
            delta_ba_nodes_[static_cast<size_t>(mk)].data(),
            delta_sg_nodes_[static_cast<size_t>(mk)].data(),
            delta_sa_nodes_[static_cast<size_t>(mk)].data());
    }

    auto interval_in_window = [&](int i) { return i >= k_lo && i + 1 <= k_hi; };

    if (windowed && k_lo > 0 &&
        !(session.marginalization_frontier && session.marginalization_frontier->valid &&
          session.marginalization_frontier->anchor_knot_index == k_lo)) {
        LOG(WARNING) << "Sliding window with k_lo=" << k_lo
                     << " but no marginalization prior on that knot; left edge may be weakly constrained.";
    }

    int gnss_horizontal_factor_count = 0;
    int gnss_vertical_factor_count = 0;
    const double max_available_time =
        nominal_nav_.empty() ? -std::numeric_limits<double>::infinity() : nominal_nav_.back().time;
    constexpr double kCausalTimeTol = 1.0e-6;
    if (config_.use_gnss_factors) {
        auto gnss_begin = gnss_.cbegin();
        auto gnss_end = gnss_.cend();
        if (windowed) {
            const double window_time_lo = control_points_[static_cast<size_t>(k_lo)].Timestamp() - kCausalTimeTol;
            const double window_time_hi =
                std::min(control_points_[static_cast<size_t>(k_hi)].Timestamp(), max_available_time) + kCausalTimeTol;
            gnss_begin = std::lower_bound(
                gnss_.cbegin(),
                gnss_.cend(),
                window_time_lo,
                [](const GnssMeasurement& m, double t) { return m.time < t; });
            gnss_end = std::upper_bound(
                gnss_begin,
                gnss_.cend(),
                window_time_hi,
                [](double t, const GnssMeasurement& m) { return t < m.time; });
        }
        for (auto it = gnss_begin; it != gnss_end; ++it) {
            const auto& gnss = *it;
            // In causal sliding, only consume exteroceptive measurements that are
            // already reachable by the current nominal/cached IMU prefix.
            if (windowed && gnss.time > max_available_time + kCausalTimeTol) {
                continue;
            }
            const int start = FindNodeIntervalStart(control_points_, gnss.time);
            if (start < 0 || start + 1 >= static_cast<int>(control_points_.size())) {
                continue;
            }
            if (windowed && !interval_in_window(start)) {
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

            problem.AddResidualBlock(
                factors::ErrorStateGnssVerticalLeverArmFactor::Create(
                    u,
                    nominal_pos_ned,
                    nominal_state->q_nb,
                    lever_arm_,
                    meas_pos_ned,
                    config_.gnss_sigma_vertical_m),
                nullptr,
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
            size_t nhc_start_index = 0;
            size_t nhc_end_index = nhc_.size();
            if (windowed) {
                const double window_time_lo = control_points_[static_cast<size_t>(k_lo)].Timestamp() - kCausalTimeTol;
                const double window_time_hi =
                    std::min(control_points_[static_cast<size_t>(k_hi)].Timestamp(), max_available_time) + kCausalTimeTol;
                const auto begin_it = std::lower_bound(
                    nhc_.cbegin(),
                    nhc_.cend(),
                    window_time_lo,
                    [](const NhcMeasurement& m, double t) { return m.time < t; });
                const auto end_it = std::upper_bound(
                    begin_it,
                    nhc_.cend(),
                    window_time_hi,
                    [](double t, const NhcMeasurement& m) { return t < m.time; });
                nhc_start_index = static_cast<size_t>(std::distance(nhc_.cbegin(), begin_it));
                nhc_end_index = static_cast<size_t>(std::distance(nhc_.cbegin(), end_it));
            }
            if (nhc_stride > 1 && nhc_start_index < nhc_end_index) {
                const size_t rem = nhc_start_index % nhc_stride;
                if (rem != 0) {
                    nhc_start_index += (nhc_stride - rem);
                }
            }
            for (size_t nhc_index = nhc_start_index; nhc_index < nhc_end_index; nhc_index += nhc_stride) {
                const auto& nhc = nhc_[nhc_index];
                if (windowed && nhc.time > max_available_time + kCausalTimeTol) {
                    continue;
                }
                const int start = FindNodeIntervalStart(control_points_, nhc.time);
                if (start < 0 || start + 1 >= static_cast<int>(control_points_.size())) {
                    continue;
                }
                if (windowed && !interval_in_window(start)) {
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
        const int i_end = windowed ? std::min(k_hi - 1, static_cast<int>(control_points_.size()) - 2) : static_cast<int>(control_points_.size()) - 2;
        const int i_begin = windowed ? k_lo : 0;
        for (int i = i_begin; i <= i_end; ++i) {
            if (i < 0) {
                continue;
            }
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
                delta_sg_nodes_[i].data(),
                delta_sa_nodes_[i].data(),
                delta_theta_nodes_[i + 1].data(),
                delta_vel_nodes_[i + 1].data(),
                delta_pos_nodes_[i + 1].data(),
                delta_bg_nodes_[i + 1].data(),
                delta_ba_nodes_[i + 1].data(),
                delta_sg_nodes_[i + 1].data(),
                delta_sa_nodes_[i + 1].data());
            ++process_factor_count;
        }
    }

    ceres::Solver::Options options;
    if (windowed) {
        if (session.sliding_solver_max_iterations_override >= 1) {
            options.max_num_iterations = session.sliding_solver_max_iterations_override;
        } else {
            options.max_num_iterations = config_.solver_max_iterations_window;
        }
        if (config_.sliding_window_function_tolerance > 0.0) {
            options.function_tolerance = config_.sliding_window_function_tolerance;
        }
        if (config_.sliding_window_gradient_tolerance > 0.0) {
            options.gradient_tolerance = config_.sliding_window_gradient_tolerance;
        }
    } else {
        options.max_num_iterations = config_.solver_max_iterations;
    }
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.num_threads = std::max(1u, std::thread::hardware_concurrency());
    options.minimizer_progress_to_stdout = !windowed;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (windowed && session.window_solver_stats_out) {
        session.window_solver_stats_out->num_successful_steps = summary.num_successful_steps;
        session.window_solver_stats_out->initial_cost = summary.initial_cost;
        session.window_solver_stats_out->final_cost = summary.final_cost;
        session.window_solver_stats_out->termination_type = static_cast<int>(summary.termination_type);
    }

    const bool emit_window_logs = !windowed || config_.sliding_window_log_timing;
    if (emit_window_logs) {
        LOG(INFO) << "GNSS factors (horizontal / vertical): "
                  << gnss_horizontal_factor_count << " / " << gnss_vertical_factor_count;
        LOG(INFO) << "NHC factors: " << nhc_factor_count;
        if (config_.yaw_bias_enable && yaw_bias_rad) {
            LOG(INFO) << "Current yaw_bias_rad: " << *yaw_bias_rad;
        }
        LOG(INFO) << "Interval propagation factors: " << process_factor_count;
        if (windowed) {
            LOG(INFO) << "Window knots [" << k_lo << ", " << k_hi << "]"
                      << (session.marginalization_frontier && session.marginalization_frontier->valid
                              ? " (with marginalization prior)"
                              : "");
        }
        LOG(INFO) << summary.BriefReport();
    }
    return summary.termination_type != ceres::FAILURE;
}

}  // namespace ct_fgo_sim
