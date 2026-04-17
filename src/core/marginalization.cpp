#include "ct_fgo_sim/core/marginalization_frontier.h"

#include "ct_fgo_sim/core/factor_graph_session.h"
#include "ct_fgo_sim/core/spline_helpers.h"
#include "ct_fgo_sim/core/system.h"
#include "ct_fgo_sim/factors/error_state_gnss_factor.h"
#include "ct_fgo_sim/factors/error_state_interval_factor.h"
#include "ct_fgo_sim/navigation/earth.h"
#include "ct_fgo_sim/navigation/mechanization.h"
#include "ct_fgo_sim/types.h"

#include <ceres/ceres.h>
#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

namespace ct_fgo_sim {

namespace {

using Matrix15d = MatrixErrorState;
using Vector15d = VectorErrorState;
using Matrix30d = MatrixErrorState2;
using Vector30d = VectorErrorState2;

/// Stacks one knot's Ceres blocks into a 21-vector: [theta, vel, pos, bg, ba, sg, sa] (same column order as
/// `ErrorStateIntervalFactor` Jacobians from `CostFunction::Evaluate`).
Vector15d StackCeres15(
    const double* dtheta,
    const double* dv,
    const double* dp,
    const double* dbg,
    const double* dba,
    const double* dsg,
    const double* dsa) {
    Vector15d v;
    v.segment<3>(0) = Eigen::Map<const Vector3d>(dtheta);
    v.segment<3>(3) = Eigen::Map<const Vector3d>(dv);
    v.segment<3>(6) = Eigen::Map<const Vector3d>(dp);
    v.segment<3>(9) = Eigen::Map<const Vector3d>(dbg);
    v.segment<3>(12) = Eigen::Map<const Vector3d>(dba);
    v.segment<3>(15) = Eigen::Map<const Vector3d>(dsg);
    v.segment<3>(18) = Eigen::Map<const Vector3d>(dsa);
    return v;
}

bool AppendIntervalHessian30(
    const MatrixErrorState& phi,
    const MatrixErrorState& sqrt_info,
    const double* p10[14],
    Matrix30d& H,
    Vector30d& g) {
    std::unique_ptr<ceres::CostFunction> cost(
        factors::ErrorStateIntervalFactor::Create(phi, sqrt_info));
    double residuals[21] = {};
    double r0[21 * 3] = {};
    double r1[21 * 3] = {};
    double r2[21 * 3] = {};
    double r3[21 * 3] = {};
    double r4[21 * 3] = {};
    double r5[21 * 3] = {};
    double r6[21 * 3] = {};
    double r7[21 * 3] = {};
    double r8[21 * 3] = {};
    double r9[21 * 3] = {};
    double r10[21 * 3] = {};
    double r11[21 * 3] = {};
    double r12[21 * 3] = {};
    double r13[21 * 3] = {};
    double* jacobians[14] = {r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13};
    if (!cost->Evaluate(p10, residuals, jacobians)) {
        return false;
    }
    Eigen::Matrix<double, 21, 42> J;
    for (int b = 0; b < 14; ++b) {
        J.block(0, 3 * b, 21, 3) = Eigen::Map<Eigen::Matrix<double, 21, 3, Eigen::RowMajor>>(jacobians[b]);
    }
    H += J.transpose() * J;
    Eigen::Map<Vector15d> rmap(residuals);
    g += J.transpose() * rmap;
    return true;
}

bool AppendGnssHessian30(
    ceres::CostFunction* cost,
    int residual_dim,
    const double* const p4[4],
    Matrix30d& H,
    Vector30d& g) {
    std::vector<double> residuals(static_cast<size_t>(residual_dim), 0.0);
    Eigen::MatrixXd J(residual_dim, 12);
    std::vector<double*> jac_rows(4);
    std::vector<std::vector<double>> jac_storage(4);
    for (int i = 0; i < 4; ++i) {
        jac_storage[i].assign(static_cast<size_t>(residual_dim * 3), 0.0);
        jac_rows[i] = jac_storage[i].data();
    }
    if (!cost->Evaluate(p4, residuals.data(), jac_rows.data())) {
        return false;
    }
    for (int i = 0; i < 4; ++i) {
        J.block(0, 3 * i, residual_dim, 3) =
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>>(jac_rows[i], residual_dim, 3);
    }
    Eigen::Matrix<double, Eigen::Dynamic, 42> Jbig(residual_dim, 42);
    Jbig.setZero();
    // CostFunction parameter order: dp_i, dp_j, dtheta_i, dtheta_j.
    // 42-column layout matches interval: knot0 [theta..sa], knot1 [theta..sa].
    Jbig.block(0, 6, residual_dim, 3) = J.block(0, 0, residual_dim, 3);           // dp_i
    Jbig.block(0, 21 + 6, residual_dim, 3) = J.block(0, 3, residual_dim, 3);      // dp_j
    Jbig.block(0, 0, residual_dim, 3) = J.block(0, 6, residual_dim, 3);           // dtheta_i
    Jbig.block(0, 21 + 0, residual_dim, 3) = J.block(0, 9, residual_dim, 3);      // dtheta_j
    Eigen::Map<Eigen::VectorXd> r_vec(residuals.data(), residual_dim);
    H += Jbig.transpose() * Jbig;
    g += Jbig.transpose() * r_vec;
    return true;
}

class MarginalPriorCostFunction : public ceres::SizedCostFunction<21, 3, 3, 3, 3, 3, 3, 3> {
public:
    MarginalPriorCostFunction(Matrix15d H, Vector15d g_lin, Vector15d x0)
        : x0_(std::move(x0)) {
        const double eps = 1.0e-9;
        const Matrix15d Hsym = 0.5 * (H + H.transpose());
        Matrix15d Hwork = Hsym;
        Hwork.diagonal().array() += eps;
        Eigen::LLT<Matrix15d> llt(Hwork);
        if (llt.info() != Eigen::Success) {
            Hwork = Hsym;
            Hwork.diagonal().array() += 1.0e-6;
            llt.compute(Hwork);
            if (llt.info() != Eigen::Success) {
                LOG(ERROR) << "MarginalPriorCostFunction: LLT failed after damping";
                Lt_.setIdentity();
                c_.setZero();
                return;
            }
        }
        const Matrix15d L = llt.matrixL();
        Lt_ = L.transpose();
        c_ = L.triangularView<Eigen::Lower>().solve(g_lin);
    }

    bool Evaluate(
        double const* const* parameters,
        double* residuals,
        double** jacobians) const override {
        const Vector15d v = StackCeres15(parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], parameters[6]);
        const Vector15d dx = v - x0_;
        Eigen::Map<Vector15d> r(residuals);
        r = Lt_ * dx + c_;
        if (!jacobians) {
            return true;
        }
        for (int k = 0; k < 7; ++k) {
            if (jacobians[k]) {
                Eigen::Map<Eigen::Matrix<double, 21, 3, Eigen::RowMajor>> Jk(jacobians[k]);
                Jk = Lt_.block(0, 3 * k, 21, 3);
            }
        }
        return true;
    }

private:
    Matrix15d Lt_ = Matrix15d::Identity();
    Vector15d c_ = Vector15d::Zero();
    Vector15d x0_ = Vector15d::Zero();
};

}  // namespace

ceres::CostFunction* CreateMarginalizationPriorCost(const MarginalizationFrontier& frontier) {
    if (!frontier.valid) {
        return nullptr;
    }
    return new MarginalPriorCostFunction(frontier.H, frontier.g, frontier.x0);
}

bool MarginalizeOldestKnotTwoKnotWindow(
    int k_drop,
    const FactorGraphSession& session,
    const MarginalizationFrontier* prior_on_k_drop,
    MarginalizationFrontier& out_on_k_drop_plus_1) {
    out_on_k_drop_plus_1.reset();
    if (!session.config || !session.control_points || !session.delta_theta_nodes || !session.delta_vel_nodes ||
        !session.delta_pos_nodes || !session.delta_bg_nodes || !session.delta_ba_nodes ||
        !session.delta_sg_nodes || !session.delta_sa_nodes || !session.interval_cache ||
        !session.origin_blh || !session.nominal_nav || !session.lever_arm) {
        LOG(ERROR) << "MarginalizeOldestKnotTwoKnotWindow: incomplete session";
        return false;
    }
    const auto& control_points = *session.control_points;
    const int n = static_cast<int>(control_points.size());
    if (k_drop < 0 || k_drop + 1 >= n) {
        return false;
    }
    if (k_drop >= static_cast<int>(session.interval_cache->knot_intervals.size())) {
        return false;
    }
    const auto& knot_iv = session.interval_cache->knot_intervals[static_cast<size_t>(k_drop)];
    if (!knot_iv.valid) {
        LOG(WARNING) << "MarginalizeOldestKnotTwoKnotWindow: invalid knot interval at " << k_drop;
        return false;
    }

    const double* p10[14] = {
        (*session.delta_theta_nodes)[static_cast<size_t>(k_drop)].data(),
        (*session.delta_vel_nodes)[static_cast<size_t>(k_drop)].data(),
        (*session.delta_pos_nodes)[static_cast<size_t>(k_drop)].data(),
        (*session.delta_bg_nodes)[static_cast<size_t>(k_drop)].data(),
        (*session.delta_ba_nodes)[static_cast<size_t>(k_drop)].data(),
        (*session.delta_sg_nodes)[static_cast<size_t>(k_drop)].data(),
        (*session.delta_sa_nodes)[static_cast<size_t>(k_drop)].data(),
        (*session.delta_theta_nodes)[static_cast<size_t>(k_drop + 1)].data(),
        (*session.delta_vel_nodes)[static_cast<size_t>(k_drop + 1)].data(),
        (*session.delta_pos_nodes)[static_cast<size_t>(k_drop + 1)].data(),
        (*session.delta_bg_nodes)[static_cast<size_t>(k_drop + 1)].data(),
        (*session.delta_ba_nodes)[static_cast<size_t>(k_drop + 1)].data(),
        (*session.delta_sg_nodes)[static_cast<size_t>(k_drop + 1)].data(),
        (*session.delta_sa_nodes)[static_cast<size_t>(k_drop + 1)].data(),
    };

    Matrix30d H = Matrix30d::Zero();
    Vector30d g = Vector30d::Zero();
    if (session.config->use_imu_factors) {
        if (!AppendIntervalHessian30(knot_iv.phi, knot_iv.sqrt_info, p10, H, g)) {
            return false;
        }
    }

    if (prior_on_k_drop && prior_on_k_drop->valid && prior_on_k_drop->anchor_knot_index == k_drop) {
        const Vector15d v_drop = StackCeres15(p10[0], p10[1], p10[2], p10[3], p10[4], p10[5], p10[6]);
        H.block<21, 21>(0, 0) += prior_on_k_drop->H;
        g.head<21>() +=
            prior_on_k_drop->H * (v_drop - prior_on_k_drop->x0) + prior_on_k_drop->g;
    }

    if (session.config->use_gnss_factors && session.gnss) {
        AppConfig& cfg = *session.config;
        constexpr double kGnssTimeTol = 1.0e-6;
        const double t_window_lo = control_points[static_cast<size_t>(k_drop)].Timestamp() - kGnssTimeTol;
        const double t_window_hi = control_points[static_cast<size_t>(k_drop + 1)].Timestamp() + kGnssTimeTol;
        auto gnss_begin = std::lower_bound(
            session.gnss->begin(),
            session.gnss->end(),
            t_window_lo,
            [](const GnssMeasurement& m, double t) { return m.time < t; });
        auto gnss_end = std::upper_bound(
            gnss_begin,
            session.gnss->end(),
            t_window_hi,
            [](double t, const GnssMeasurement& m) { return t < m.time; });
        for (auto it = gnss_begin; it != gnss_end; ++it) {
            const auto& gnss = *it;
            const int start = FindNodeIntervalStart(control_points, gnss.time);
            if (start != k_drop) {
                continue;
            }
            if (start < 0 || start + 1 >= n) {
                continue;
            }
            const double dt = control_points[static_cast<size_t>(start + 1)].Timestamp() -
                control_points[static_cast<size_t>(start)].Timestamp();
            if (dt <= 1.0e-9) {
                continue;
            }
            const auto nominal_state = EvaluateNominalState(*session.nominal_nav, gnss.time);
            if (!nominal_state) {
                continue;
            }
            const double u = std::clamp(
                (gnss.time - control_points[static_cast<size_t>(start)].Timestamp()) / dt,
                0.0,
                1.0);
            const Vector3d nominal_pos_ned = Earth::GlobalToLocal(*session.origin_blh, nominal_state->blh);
            const Vector3d meas_pos_ned = Earth::GlobalToLocal(*session.origin_blh, gnss.blh);
            const double* p4h[4] = {
                (*session.delta_pos_nodes)[static_cast<size_t>(start)].data(),
                (*session.delta_pos_nodes)[static_cast<size_t>(start + 1)].data(),
                (*session.delta_theta_nodes)[static_cast<size_t>(start)].data(),
                (*session.delta_theta_nodes)[static_cast<size_t>(start + 1)].data(),
            };
            std::unique_ptr<ceres::CostFunction> ch(factors::ErrorStateGnssHorizontalLeverArmFactor::Create(
                u,
                nominal_pos_ned,
                nominal_state->q_nb,
                *session.lever_arm,
                meas_pos_ned,
                cfg.gnss_sigma_horizontal_m));
            AppendGnssHessian30(ch.get(), 2, p4h, H, g);
            std::unique_ptr<ceres::CostFunction> cv(factors::ErrorStateGnssVerticalLeverArmFactor::Create(
                u,
                nominal_pos_ned,
                nominal_state->q_nb,
                *session.lever_arm,
                meas_pos_ned,
                cfg.gnss_sigma_vertical_m));
            AppendGnssHessian30(cv.get(), 1, p4h, H, g);
        }
    }

    {
        const Matrix30d Hsym = 0.5 * (H + H.transpose());
        H = Hsym;
    }
    if (k_drop == 0) {
        // Approximate a fixed left knot (aligned with Ceres SetParameterBlockConstant on knot 0).
        H.block<21, 21>(0, 0).diagonal().array() += 1.0e8;
    }

    const Matrix15d Hmm = H.block<21, 21>(0, 0);
    const Matrix15d Hmr = H.block<21, 21>(0, 21);
    const Matrix15d Hrr = H.block<21, 21>(21, 21);
    const Vector15d gm = g.head<21>();
    const Vector15d gr = g.tail<21>();

    Eigen::LDLT<Matrix15d> ldlt(Hmm);
    if (ldlt.info() != Eigen::Success) {
        LOG(WARNING) << "MarginalizeOldestKnotTwoKnotWindow: LDLT failed on Hmm; adding damping";
        Eigen::LDLT<Matrix15d> ldlt2(Hmm + 1.0e-3 * Matrix15d::Identity());
        if (ldlt2.info() != Eigen::Success) {
            return false;
        }
        ldlt = ldlt2;
    }
    const Matrix15d Hmm_inv_Hmr = ldlt.solve(Hmr);
    const Vector15d Hmm_inv_gm = ldlt.solve(gm);
    const Matrix15d H_red = Hrr - Hmr.transpose() * Hmm_inv_Hmr;
    const Vector15d g_red = gr - Hmr.transpose() * Hmm_inv_gm;

    const Vector15d x1 = StackCeres15(p10[7], p10[8], p10[9], p10[10], p10[11], p10[12], p10[13]);

    out_on_k_drop_plus_1.valid = true;
    out_on_k_drop_plus_1.anchor_knot_index = k_drop + 1;
    out_on_k_drop_plus_1.x0 = x1;
    out_on_k_drop_plus_1.H = H_red;
    out_on_k_drop_plus_1.g = g_red;
    return true;
}

}  // namespace ct_fgo_sim
