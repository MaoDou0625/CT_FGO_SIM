#include "ct_fgo_sim/core/factor_graph_backend.h"

#include "ct_fgo_sim/core/spline_helpers.h"
#include "ct_fgo_sim/core/factor_graph_session.h"
#include "ct_fgo_sim/core/system.h"
#include "ct_fgo_sim/navigation/earth.h"
#include "ct_fgo_sim/navigation/mechanization.h"

#include <glog/logging.h>

#include <algorithm>
#include <cctype>
#include <limits>

#include <gtsam/base/Matrix.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/nonlinear/Values.h>

namespace ct_fgo_sim {

namespace {

std::string ToLowerAscii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

thread_local const char* g_last_backend_impl = "unknown";
thread_local const char* g_last_backend_fallback_reason = "";

// Interval propagation / ErrorStateIntervalFactor internal state order is:
//   [pos, vel, theta, bg, ba, sg, sa]
// Ceres parameter block order (and MarginalizationFrontier order) is:
//   [theta, vel, pos, bg, ba, sg, sa]
//
// We keep the GTSAM 21D variable in the Ceres block order to avoid fragile prior-frontier
// permutations, and permute into propagation order only inside ProcessFactorGtsam.
constexpr int kIdxTheta = 0;
constexpr int kIdxVel = 3;
constexpr int kIdxPos = 6;
constexpr int kIdxBg = 9;
constexpr int kIdxBa = 12;
constexpr int kIdxSg = 15;
constexpr int kIdxSa = 18;

using gtsam::Matrix;
using gtsam::Matrix21;
using gtsam::Vector;
using gtsam::Vector2;
using gtsam::Vector3;

Eigen::PermutationMatrix<21> CeresToPropagationPerm() {
    Eigen::PermutationMatrix<21> P;
    P.setIdentity();
    // ceres block order: [theta vel pos bg ba sg sa]
    // propagation/gtsam order: [pos vel theta bg ba sg sa]
    for (int i = 0; i < 21; ++i) {
        P.indices()[i] = i;
    }
    for (int i = 0; i < 3; ++i) {
        P.indices()[i] = 6 + i;
        P.indices()[3 + i] = 3 + i;
        P.indices()[6 + i] = i;
        P.indices()[9 + i] = 9 + i;
        P.indices()[12 + i] = 12 + i;
        P.indices()[15 + i] = 15 + i;
        P.indices()[18 + i] = 18 + i;
    }
    return P;
}

gtsam::Vector ToStateVec21(
    const Vector3d& dtheta,
    const Vector3d& dvel,
    const Vector3d& dpos,
    const Vector3d& dbg,
    const Vector3d& dba,
    const Vector3d& dsg,
    const Vector3d& dsa) {
    gtsam::Vector x(21);
    x.segment<3>(kIdxTheta) = dtheta;
    x.segment<3>(kIdxVel) = dvel;
    x.segment<3>(kIdxPos) = dpos;
    x.segment<3>(kIdxBg) = dbg;
    x.segment<3>(kIdxBa) = dba;
    x.segment<3>(kIdxSg) = dsg;
    x.segment<3>(kIdxSa) = dsa;
    return x;
}

class ProcessFactorGtsam : public gtsam::NoiseModelFactor2<gtsam::Vector, gtsam::Vector> {
public:
    ProcessFactorGtsam(
        gtsam::Key key_i,
        gtsam::Key key_j,
        const MatrixErrorState& phi,
        const MatrixErrorState& P_ceres_to_prop,
        const gtsam::SharedNoiseModel& model)
        : gtsam::NoiseModelFactor2<gtsam::Vector, gtsam::Vector>(model, key_i, key_j),
          phi_(phi),
          P_(P_ceres_to_prop) {}

    gtsam::Vector evaluateError(
        const gtsam::Vector& xi,
        const gtsam::Vector& xj,
        boost::optional<gtsam::Matrix&> H1 = boost::none,
        boost::optional<gtsam::Matrix&> H2 = boost::none) const override {
        const Vector xi_prop = P_ * xi;
        const Vector xj_prop = P_ * xj;
        const Vector err = xj_prop - phi_ * xi_prop;

        if (H1) {
            // err = P*xj - phi*(P*xi)
            *H1 = -(phi_ * P_);
        }
        if (H2) {
            *H2 = P_;
        }
        return err;
    }

private:
    MatrixErrorState phi_;
    MatrixErrorState P_;
};

class MarginalPriorFactorGtsam : public gtsam::NoiseModelFactor1<gtsam::Vector> {
public:
    MarginalPriorFactorGtsam(
        gtsam::Key key,
        const VectorErrorState& x0,
        const MatrixErrorState& H,
        const VectorErrorState& g)
        : gtsam::NoiseModelFactor1<gtsam::Vector>(
              gtsam::noiseModel::Isotropic::Sigma(21, 1.0), key),
          x0_(x0) {
        // Match Ceres-side MarginalPriorCostFunction for numerical equivalence:
        //  - eps diagonal damping: 1e-9
        //  - if LLT fails: retry with 1e-6 damping
        //  - if still fails: fall back to identity/zero (same as Ceres behavior).
        const double eps = 1.0e-9;
        MatrixErrorState Hsym = 0.5 * (H + H.transpose());

        MatrixErrorState Hwork = Hsym;
        Hwork.diagonal().array() += eps;

        Eigen::LLT<MatrixErrorState> llt(Hwork);
        if (llt.info() != Eigen::Success) {
            Hwork = Hsym;
            Hwork.diagonal().array() += 1.0e-6;
            llt.compute(Hwork);
        }

        if (llt.info() == Eigen::Success) {
            const MatrixErrorState L = llt.matrixL();
            Lt_ = L.transpose();
            c_ = L.triangularView<Eigen::Lower>().solve(g);
        } else {
            Lt_.setIdentity();
            c_.setZero();
        }
    }

    gtsam::Vector evaluateError(
        const gtsam::Vector& x,
        boost::optional<gtsam::Matrix&> H1 = boost::none) const override {
        if (H1) {
            *H1 = Lt_;
        }
        return Lt_ * (x - x0_) + c_;
    }

private:
    VectorErrorState x0_ = VectorErrorState::Zero();
    MatrixErrorState Lt_ = MatrixErrorState::Identity();
    VectorErrorState c_ = VectorErrorState::Zero();
};

class GnssHorizontalFactorGtsam : public gtsam::NoiseModelFactor2<gtsam::Vector, gtsam::Vector> {
public:
    GnssHorizontalFactorGtsam(
        gtsam::Key key_i,
        gtsam::Key key_j,
        double u,
        const Vector3d& nominal_pos_ned,
        const Vector3d& meas_pos_ned,
        const gtsam::SharedNoiseModel& model)
        : gtsam::NoiseModelFactor2<gtsam::Vector, gtsam::Vector>(model, key_i, key_j),
          u_(u),
          nominal_pos_ned_(nominal_pos_ned),
          meas_pos_ned_(meas_pos_ned) {}

    gtsam::Vector evaluateError(
        const gtsam::Vector& xi,
        const gtsam::Vector& xj,
        boost::optional<gtsam::Matrix&> H1 = boost::none,
        boost::optional<gtsam::Matrix&> H2 = boost::none) const override {
        const Vector3 delta_pos = (1.0 - u_) * xi.segment<3>(kIdxPos) + u_ * xj.segment<3>(kIdxPos);
        const Vector3 pred = nominal_pos_ned_ + delta_pos;
        Vector2 err = pred.head<2>() - meas_pos_ned_.head<2>();
        if (H1) {
            H1->setZero(2, 21);
            H1->block<2, 2>(0, kIdxPos) = Matrix::Identity(2, 2) * (1.0 - u_);
        }
        if (H2) {
            H2->setZero(2, 21);
            H2->block<2, 2>(0, kIdxPos) = Matrix::Identity(2, 2) * u_;
        }
        return err;
    }

private:
    double u_ = 0.0;
    Vector3d nominal_pos_ned_ = Vector3d::Zero();
    Vector3d meas_pos_ned_ = Vector3d::Zero();
};

class GnssVerticalFactorGtsam : public gtsam::NoiseModelFactor2<gtsam::Vector, gtsam::Vector> {
public:
    GnssVerticalFactorGtsam(
        gtsam::Key key_i,
        gtsam::Key key_j,
        double u,
        const Vector3d& nominal_pos_ned,
        const Vector3d& meas_pos_ned,
        double sigma_vertical_m,
        const gtsam::SharedNoiseModel& model)
        : gtsam::NoiseModelFactor2<gtsam::Vector, gtsam::Vector>(model, key_i, key_j),
          u_(u),
          nominal_pos_ned_(nominal_pos_ned),
          meas_pos_ned_(meas_pos_ned),
          inv_sigma_vertical_(1.0 / std::max(1.0e-6, sigma_vertical_m)) {}

    gtsam::Vector evaluateError(
        const gtsam::Vector& xi,
        const gtsam::Vector& xj,
        boost::optional<gtsam::Matrix&> H1 = boost::none,
        boost::optional<gtsam::Matrix&> H2 = boost::none) const override {
        const double dz = (1.0 - u_) * xi(kIdxPos + 2) + u_ * xj(kIdxPos + 2);
        const double r0 = nominal_pos_ned_.z() + dz - meas_pos_ned_.z();
        const double r = inv_sigma_vertical_ * r0;

        gtsam::Vector1 err;
        err(0) = r;
        if (H1) {
            H1->setZero(1, 21);
            (*H1)(0, kIdxPos + 2) = inv_sigma_vertical_ * (1.0 - u_);
        }
        if (H2) {
            H2->setZero(1, 21);
            (*H2)(0, kIdxPos + 2) = inv_sigma_vertical_ * u_;
        }
        return err;
    }

private:
    double u_ = 0.0;
    Vector3d nominal_pos_ned_ = Vector3d::Zero();
    Vector3d meas_pos_ned_ = Vector3d::Zero();
    double inv_sigma_vertical_ = 1.0;
};

bool BuildAndSolveFactorGraphGtsamBatch(FactorGraphSession& session) {
    // GTSAM sliding + marginalization: 21D knot states + Ceres-built MarginalizationFrontier on k_lo.
    // Done: IMU interval (ProcessFactorGtsam), GNSS H/V (simplified vs Ceres lever-arm), marginal prior.
    // Still gated below (fallback): NHC with body axes, q_body_imu estimation, non-zero lever arm.
    if (!session.config || !session.origin_blh || !session.gnss || !session.control_points || !session.nominal_nav ||
        !session.interval_cache || !session.delta_theta_nodes || !session.delta_vel_nodes || !session.delta_pos_nodes ||
        !session.delta_bg_nodes || !session.delta_ba_nodes || !session.delta_sg_nodes || !session.delta_sa_nodes) {
        g_last_backend_fallback_reason = "incomplete_session";
        LOG(ERROR) << "BuildAndSolveFactorGraphGtsamBatch: incomplete session";
        return false;
    }
    AppConfig& config = *session.config;
    const bool windowed = session.window_knot_lo >= 0 && session.window_knot_hi >= session.window_knot_lo;
    if (config.yaw_bias_enable && !session.yaw_bias_rad) {
        g_last_backend_fallback_reason = "gtsam_batch_missing_yaw_bias_ptr";
        return false;
    }
    const bool nhc_axis_enabled =
        config.body_frame.nhc_enable_vx || config.body_frame.nhc_enable_vy || config.body_frame.nhc_enable_vz;
    if (config.body_frame.enable_nhc && nhc_axis_enabled) {
        g_last_backend_fallback_reason = "gtsam_batch_no_nhc";
        return false;
    }
    if (config.body_frame.enable_nhc && config.body_frame.estimate_q_body_imu) {
        g_last_backend_fallback_reason = "gtsam_batch_no_q_body_imu_estimation";
        return false;
    }
    if (session.lever_arm && session.lever_arm->norm() > 1.0e-9) {
        g_last_backend_fallback_reason = "gtsam_batch_no_lever_arm";
        return false;
    }
    if (session.control_points->size() < 2) {
        g_last_backend_fallback_reason = "insufficient_knots";
        return false;
    }
    if (!config.use_imu_factors) {
        g_last_backend_fallback_reason = "gtsam_batch_requires_imu_factors";
        return false;
    }

    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initial;

    auto& cp = *session.control_points;
    auto& dtheta = *session.delta_theta_nodes;
    auto& dvel = *session.delta_vel_nodes;
    auto& dpos = *session.delta_pos_nodes;
    auto& dbg = *session.delta_bg_nodes;
    auto& dba = *session.delta_ba_nodes;
    auto& dsg = *session.delta_sg_nodes;
    auto& dsa = *session.delta_sa_nodes;
    auto& cache = *session.interval_cache;
    auto& gnss = *session.gnss;
    auto& nominal = *session.nominal_nav;
    auto& origin_blh = *session.origin_blh;
    const int n_knots = static_cast<int>(cp.size());
    int k_lo = 0;
    int k_hi = n_knots - 1;
    if (windowed) {
        k_lo = std::clamp(session.window_knot_lo, 0, n_knots - 1);
        k_hi = std::clamp(session.window_knot_hi, k_lo, n_knots - 1);
        if (k_hi - k_lo < 1) {
            g_last_backend_fallback_reason = "invalid_window_span";
            return false;
        }
    }

    for (int k = k_lo; k <= k_hi; ++k) {
        const gtsam::Vector x0 = ToStateVec21(
            dtheta[static_cast<size_t>(k)],
            dvel[static_cast<size_t>(k)],
            dpos[static_cast<size_t>(k)],
            dbg[static_cast<size_t>(k)],
            dba[static_cast<size_t>(k)],
            dsg[static_cast<size_t>(k)],
            dsa[static_cast<size_t>(k)]);
        if (!x0.allFinite()) {
            g_last_backend_fallback_reason = "non_finite_initial_state";
            return false;
        }
        initial.insert(gtsam::Symbol('x', static_cast<uint64_t>(k)), x0);
    }

    const gtsam::Key yaw_bias_key = gtsam::Symbol('b', 0);
    if (config.yaw_bias_enable && session.yaw_bias_rad) {
        gtsam::Vector y0(1);
        y0 << *session.yaw_bias_rad;
        initial.insert(yaw_bias_key, y0);
        graph.add(gtsam::PriorFactor<gtsam::Vector>(
            yaw_bias_key,
            gtsam::Vector::Zero(1),
            gtsam::noiseModel::Isotropic::Sigma(1, std::max(1.0e-12, config.yaw_bias_prior_sigma_rad))));
    }

    if (!windowed || k_lo == 0) {
        graph.add(gtsam::PriorFactor<gtsam::Vector>(
            gtsam::Symbol('x', static_cast<uint64_t>(k_lo)),
            initial.at<gtsam::Vector>(gtsam::Symbol('x', static_cast<uint64_t>(k_lo))),
            gtsam::noiseModel::Isotropic::Sigma(21, 1.0e-9)));
    }

    if (session.marginalization_frontier && session.marginalization_frontier->valid &&
        session.marginalization_frontier->anchor_knot_index == k_lo && k_lo > 0) {
        // GTSAM 21D variable is in Ceres block order, so the frontier can be applied directly.
        graph.add(boost::make_shared<MarginalPriorFactorGtsam>(
            gtsam::Symbol('x', static_cast<uint64_t>(k_lo)),
            session.marginalization_frontier->x0,
            session.marginalization_frontier->H,
            session.marginalization_frontier->g));
    } else if (windowed && k_lo > 0) {
        // Keep the left edge of non-marginalized windows anchored to the incoming trajectory estimate.
        graph.add(gtsam::PriorFactor<gtsam::Vector>(
            gtsam::Symbol('x', static_cast<uint64_t>(k_lo)),
            initial.at<gtsam::Vector>(gtsam::Symbol('x', static_cast<uint64_t>(k_lo))),
            gtsam::noiseModel::Isotropic::Sigma(21, 1.0e-5)));
    }

    // Permutation from Ceres block order to propagation internal order inside the interval factors.
    const auto P_c2p = CeresToPropagationPerm();
    const MatrixErrorState P_c2p_dense = P_c2p.toDenseMatrix();

    if (config.use_imu_factors) {
        for (int i = k_lo; i <= k_hi - 1 && i < static_cast<int>(cache.knot_intervals.size()); ++i) {
            const auto& knot = cache.knot_intervals[static_cast<size_t>(i)];
            if (!knot.valid) {
                g_last_backend_fallback_reason = "invalid_interval_cache";
                return false;
            }
            if (!knot.phi.allFinite() || !knot.sqrt_info.allFinite()) {
                g_last_backend_fallback_reason = "non_finite_interval_cache";
                return false;
            }
            auto model = gtsam::noiseModel::Gaussian::SqrtInformation(knot.sqrt_info);
            graph.add(boost::make_shared<ProcessFactorGtsam>(
                gtsam::Symbol('x', static_cast<uint64_t>(i)),
                gtsam::Symbol('x', static_cast<uint64_t>(i + 1)),
                knot.phi,
                P_c2p_dense,
                model));
        }
    }

    if (config.use_gnss_factors) {
        const auto interval_in_window = [&](int i) { return i >= k_lo && i + 1 <= k_hi; };
        const double kCausalTimeTol = 1.0e-6;
        const double max_available_time = nominal.empty() ? -std::numeric_limits<double>::infinity() : nominal.back().time;
        const double window_time_hi =
            windowed ? (std::min(cp[static_cast<size_t>(k_hi)].Timestamp(), max_available_time) + kCausalTimeTol)
                     : std::numeric_limits<double>::infinity();
        for (const auto& g : gnss) {
            if (windowed && g.time > window_time_hi) {
                break;
            }
            const int start = FindNodeIntervalStart(cp, g.time);
            if (start < 0 || start + 1 >= static_cast<int>(cp.size())) {
                continue;
            }
            if (windowed && !interval_in_window(start)) {
                continue;
            }
            const auto nominal_state = EvaluateNominalState(nominal, g.time);
            if (!nominal_state) {
                continue;
            }
            const double dt = cp[start + 1].Timestamp() - cp[start].Timestamp();
            if (dt <= 1.0e-9) {
                continue;
            }
            const double u = std::clamp((g.time - cp[start].Timestamp()) / dt, 0.0, 1.0);
            const Vector3d nominal_pos_ned = Earth::GlobalToLocal(origin_blh, nominal_state->blh);
            const Vector3d meas_pos_ned = Earth::GlobalToLocal(origin_blh, g.blh);
            graph.add(boost::make_shared<GnssHorizontalFactorGtsam>(
                gtsam::Symbol('x', static_cast<uint64_t>(start)),
                gtsam::Symbol('x', static_cast<uint64_t>(start + 1)),
                u,
                nominal_pos_ned,
                meas_pos_ned,
                gtsam::noiseModel::Isotropic::Sigma(2, std::max(1.0e-6, config.gnss_sigma_horizontal_m))));

            const double sigma_vertical = std::max(1.0e-6, config.gnss_sigma_vertical_m);
            const gtsam::SharedNoiseModel v_model = gtsam::noiseModel::Isotropic::Sigma(1, 1.0);
            graph.add(boost::make_shared<GnssVerticalFactorGtsam>(
                gtsam::Symbol('x', static_cast<uint64_t>(start)),
                gtsam::Symbol('x', static_cast<uint64_t>(start + 1)),
                u,
                nominal_pos_ned,
                meas_pos_ned,
                sigma_vertical,
                v_model));
        }
    }

    gtsam::LevenbergMarquardtParams params;
    if (windowed) {
        params.maxIterations =
            std::max(1, session.sliding_solver_max_iterations_override >= 1
                            ? session.sliding_solver_max_iterations_override
                            : config.solver_max_iterations_window);
    } else {
        params.maxIterations = std::max(1, config.solver_max_iterations);
    }
    params.verbosityLM = (config.gtsam_verbose_optimizer ? gtsam::LevenbergMarquardtParams::SUMMARY
                                                         : gtsam::LevenbergMarquardtParams::SILENT);
    if (windowed) {
        params.linearSolverType = gtsam::NonlinearOptimizerParams::SEQUENTIAL_QR;
        params.diagonalDamping = true;
        params.lambdaInitial = 1.0e-2;
    }
    gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial, params);
    const gtsam::Values result = optimizer.optimize();
    if (windowed && session.window_solver_stats_out) {
        session.window_solver_stats_out->initial_cost = graph.error(initial);
        session.window_solver_stats_out->final_cost = graph.error(result);
        session.window_solver_stats_out->num_successful_steps =
            std::max(1, static_cast<int>(optimizer.iterations()));
        // 0 = success for sliding-window adaptive heuristics (historically matched Ceres CONVERGENCE).
        session.window_solver_stats_out->termination_type = 0;
    }
    for (int k = k_lo; k <= k_hi; ++k) {
        const gtsam::Vector x = result.at<gtsam::Vector>(gtsam::Symbol('x', static_cast<uint64_t>(k)));
        if (!x.allFinite()) {
            g_last_backend_fallback_reason = "gtsam_non_finite_state";
            return false;
        }
        dtheta[static_cast<size_t>(k)] = x.segment<3>(kIdxTheta);
        dvel[static_cast<size_t>(k)] = x.segment<3>(kIdxVel);
        dpos[static_cast<size_t>(k)] = x.segment<3>(kIdxPos);
        dbg[static_cast<size_t>(k)] = x.segment<3>(kIdxBg);
        dba[static_cast<size_t>(k)] = x.segment<3>(kIdxBa);
        dsg[static_cast<size_t>(k)] = x.segment<3>(kIdxSg);
        dsa[static_cast<size_t>(k)] = x.segment<3>(kIdxSa);
    }
    if (config.yaw_bias_enable && session.yaw_bias_rad) {
        const gtsam::Vector y = result.at<gtsam::Vector>(yaw_bias_key);
        if (!y.allFinite()) {
            g_last_backend_fallback_reason = "gtsam_non_finite_yaw_bias";
            return false;
        }
        double yv = y(0);
        double lb = -std::numeric_limits<double>::infinity();
        double ub = std::numeric_limits<double>::infinity();
        if (config.yaw_bias_max_abs_rad > 0.0) {
            lb = -config.yaw_bias_max_abs_rad;
            ub = config.yaw_bias_max_abs_rad;
        }
        if (session.has_yaw_bias_step_limit && session.yaw_bias_step_limit_rad > 0.0) {
            const double d = session.yaw_bias_step_limit_rad;
            lb = std::max(lb, session.yaw_bias_center_rad - d);
            ub = std::min(ub, session.yaw_bias_center_rad + d);
        }
        *session.yaw_bias_rad = std::min(ub, std::max(lb, yv));
    }
    g_last_backend_fallback_reason = "";
    return true;
}

bool BuildAndSolveFactorGraphGtsamDispatch(FactorGraphSession& session) {
    try {
        if (BuildAndSolveFactorGraphGtsamBatch(session)) {
            g_last_backend_impl = "gtsam_batch";
            return true;
        }
    } catch (const std::exception& ex) {
        LOG(ERROR) << "GTSAM batch solve threw exception: " << ex.what();
        g_last_backend_fallback_reason = "gtsam_exception";
    } catch (...) {
        LOG(ERROR) << "GTSAM batch solve threw unknown exception";
        g_last_backend_fallback_reason = "gtsam_exception";
    }
    LOG(ERROR) << "GTSAM backend failed: " << g_last_backend_fallback_reason;
    g_last_backend_impl = "gtsam_failed";
    return false;
}

}  // namespace

GraphBackend ParseGraphBackend(const std::string& value) {
    const std::string lower = ToLowerAscii(value);
    if (lower == "ceres") {
        LOG(WARNING) << "backend 'ceres' is no longer supported; using GTSAM only.";
    } else if (!lower.empty() && lower != "gtsam") {
        LOG(WARNING) << "Unknown backend '" << value << "'; using GTSAM.";
    }
    return GraphBackend::Gtsam;
}

const char* GraphBackendName(GraphBackend backend) {
    switch (backend) {
        case GraphBackend::Gtsam:
            return "gtsam";
    }
    return "gtsam";
}

bool BuildAndSolveFactorGraphWithBackend(FactorGraphSession& session, GraphBackend backend) {
    (void)backend;
    return BuildAndSolveFactorGraphGtsamDispatch(session);
}

const char* ActiveGraphBackendImpl(GraphBackend backend) {
    (void)backend;
    return "gtsam_batch";
}

bool IsGtsamBackendAvailable() {
    return true;
}

const char* LastBackendImpl() {
    return g_last_backend_impl;
}

const char* LastBackendFallbackReason() {
    return g_last_backend_fallback_reason;
}

}  // namespace ct_fgo_sim

