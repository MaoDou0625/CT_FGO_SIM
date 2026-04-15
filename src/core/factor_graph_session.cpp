#include "ct_fgo_sim/core/factor_graph_session.h"

#include "ct_fgo_sim/core/spline_helpers.h"
#include "ct_fgo_sim/core/system.h"
#include "ct_fgo_sim/factors/bias_random_walk_factor.h"
#include "ct_fgo_sim/factors/continuous_gnss_factor.h"
#include "ct_fgo_sim/factors/continuous_inertial_factor.h"
#include "ct_fgo_sim/factors/error_state_gnss_factor.h"
#include "ct_fgo_sim/factors/error_state_interval_factor.h"
#include "ct_fgo_sim/factors/error_state_nhc_factor.h"
#include "ct_fgo_sim/factors/quaternion_prior_factor.h"
#include "ct_fgo_sim/navigation/earth.h"

#include <ceres/ceres.h>
#include <glog/logging.h>
#include <sophus/ceres_manifold.hpp>

#include <algorithm>
#include <cmath>
#include <thread>

namespace ct_fgo_sim {

bool BuildAndSolveFactorGraph(FactorGraphSession& session) {
    if (!session.config || !session.origin_blh || !session.gnss || !session.imu || !session.nhc ||
        !session.control_points || !session.delta_theta_nodes || !session.delta_vel_nodes ||
        !session.delta_pos_nodes || !session.delta_bg_nodes || !session.delta_ba_nodes ||
        !session.lever_arm || !session.time_offset_s || !session.q_body_imu || !session.nominal_nav ||
        !session.interval_cache) {
        LOG(ERROR) << "BuildAndSolveFactorGraph: incomplete session";
        return false;
    }

    AppConfig& config_ = *session.config;
    Vector3d& origin_blh_ = *session.origin_blh;
    GnssMeasurementArray& gnss_ = *session.gnss;
    ImuMeasurementArray& imu_ = *session.imu;
    NhcMeasurementArray& nhc_ = *session.nhc;
    spline::ControlPointArray& control_points_ = *session.control_points;
    AlignedVec3Array& delta_theta_nodes_ = *session.delta_theta_nodes;
    AlignedVec3Array& delta_vel_nodes_ = *session.delta_vel_nodes;
    AlignedVec3Array& delta_pos_nodes_ = *session.delta_pos_nodes;
    AlignedVec3Array& delta_bg_nodes_ = *session.delta_bg_nodes;
    AlignedVec3Array& delta_ba_nodes_ = *session.delta_ba_nodes;
    Vector3d& lever_arm_ = *session.lever_arm;
    double& time_offset_s_ = *session.time_offset_s;
    Eigen::Quaterniond& q_body_imu_ = *session.q_body_imu;
    NominalNavStates& nominal_nav_ = *session.nominal_nav;
    IntervalPropagationCache& interval_cache_ = *session.interval_cache;

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
        problem.AddParameterBlock(lever_arm_.data(), 3);
        problem.AddParameterBlock(&time_offset_s_, 1);
        problem.AddParameterBlock(q_body_imu_.coeffs().data(), 4, new ceres::EigenQuaternionManifold);

        problem.SetParameterBlockConstant(control_points_.front().PoseData());
        problem.SetParameterBlockConstant(delta_bg_nodes_.front().data());
        problem.SetParameterBlockConstant(delta_ba_nodes_.front().data());
        problem.SetParameterBlockConstant(lever_arm_.data());
        problem.SetParameterBlockConstant(&time_offset_s_);
        problem.SetParameterBlockConstant(q_body_imu_.coeffs().data());

        int gnss_factor_count = 0;
        if (config_.use_gnss_factors) {
            for (const auto& gnss : gnss_) {
                const int start = FindSplineWindowStart(control_points_, gnss.time);
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

        int inertial_factor_count = 0;
        if (config_.use_imu_factors) {
            const size_t imu_stride = static_cast<size_t>(std::max(1, config_.imu_stride));
            for (size_t imu_index = 1; imu_index < imu_.size(); imu_index += imu_stride) {
                const auto& meas = imu_[imu_index];
                if (meas.dt <= 1.0e-9) {
                    continue;
                }
                const int start = FindSplineWindowStart(control_points_, meas.time);
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

        ceres::Solver::Options options;
        options.max_num_iterations = config_.solver_max_iterations;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.num_threads = std::max(1u, std::thread::hardware_concurrency());
        options.minimizer_progress_to_stdout = true;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        LOG(INFO) << "Direct spline GNSS factors: " << gnss_factor_count;
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

}  // namespace ct_fgo_sim
