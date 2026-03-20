#pragma once

#include <ceres/ceres.h>
#include <Eigen/Core>

namespace ct_fgo_sim::factors {

struct ErrorStateProcessFactor {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ErrorStateProcessFactor(
        double dt,
        const Eigen::Vector3d& nominal_specific_force_ned,
        double sigma_theta,
        double sigma_v,
        double sigma_p,
        double sigma_bg,
        double sigma_ba)
        : dt_(dt),
          nominal_specific_force_ned_(nominal_specific_force_ned),
          sigma_theta_(sigma_theta),
          sigma_v_(sigma_v),
          sigma_p_(sigma_p),
          sigma_bg_(sigma_bg),
          sigma_ba_(sigma_ba) {}

    template <typename T>
    bool operator()(
        const T* const dtheta_i, const T* const dv_i, const T* const dp_i, const T* const dbg_i, const T* const dba_i,
        const T* const dtheta_j, const T* const dv_j, const T* const dp_j, const T* const dbg_j, const T* const dba_j,
        T* residuals) const {
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        const Eigen::Map<const Vec3T> phi_i(dtheta_i);
        const Eigen::Map<const Vec3T> vel_i(dv_i);
        const Eigen::Map<const Vec3T> pos_i(dp_i);
        const Eigen::Map<const Vec3T> bg_i(dbg_i);
        const Eigen::Map<const Vec3T> ba_i(dba_i);
        const Eigen::Map<const Vec3T> phi_j(dtheta_j);
        const Eigen::Map<const Vec3T> vel_j(dv_j);
        const Eigen::Map<const Vec3T> pos_j(dp_j);
        const Eigen::Map<const Vec3T> bg_j(dbg_j);
        const Eigen::Map<const Vec3T> ba_j(dba_j);

        const Vec3T f_n = nominal_specific_force_ned_.cast<T>();
        Eigen::Matrix<T, 3, 3> skew_f = Eigen::Matrix<T, 3, 3>::Zero();
        skew_f(0, 1) = -f_n.z();
        skew_f(0, 2) = f_n.y();
        skew_f(1, 0) = f_n.z();
        skew_f(1, 2) = -f_n.x();
        skew_f(2, 0) = -f_n.y();
        skew_f(2, 1) = f_n.x();

        Eigen::Map<Eigen::Matrix<T, 15, 1>> res(residuals);
        // Minimal NED error-state skeleton around nominal mechanization:
        // dtheta_dot ~= -dbg
        // dv_dot     ~= -(f_n x dtheta) - dba
        // dp_dot     ~= dv
        res.template segment<3>(0) = (phi_j - (phi_i - T(dt_) * bg_i)) / T(sigma_theta_);
        res.template segment<3>(3) =
            (vel_j - (vel_i - T(dt_) * (skew_f * phi_i + ba_i))) / T(sigma_v_);
        res.template segment<3>(6) = (pos_j - (pos_i + T(dt_) * vel_i)) / T(sigma_p_);
        res.template segment<3>(9) = (bg_j - bg_i) / T(sigma_bg_);
        res.template segment<3>(12) = (ba_j - ba_i) / T(sigma_ba_);
        return true;
    }

    static ceres::CostFunction* Create(
        double dt,
        const Eigen::Vector3d& nominal_specific_force_ned,
        double sigma_theta,
        double sigma_v,
        double sigma_p,
        double sigma_bg,
        double sigma_ba) {
        return new ceres::AutoDiffCostFunction<
            ErrorStateProcessFactor, 15,
            3, 3, 3, 3, 3,
            3, 3, 3, 3, 3>(
            new ErrorStateProcessFactor(
                dt, nominal_specific_force_ned, sigma_theta, sigma_v, sigma_p, sigma_bg, sigma_ba));
    }

private:
    double dt_;
    Eigen::Vector3d nominal_specific_force_ned_;
    double sigma_theta_;
    double sigma_v_;
    double sigma_p_;
    double sigma_bg_;
    double sigma_ba_;
};

}  // namespace ct_fgo_sim::factors
