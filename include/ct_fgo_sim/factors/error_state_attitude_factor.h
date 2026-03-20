#pragma once

#include <ceres/ceres.h>
#include <Eigen/Core>

namespace ct_fgo_sim::factors {

struct ErrorStateAttitudeFactor {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ErrorStateAttitudeFactor(
        double u,
        double dt,
        const Eigen::Vector3d& gyro_meas,
        const Eigen::Vector3d& nominal_gyro_body,
        const Eigen::Vector3d& nominal_bg,
        double sigma_g)
        : u_(u),
          dt_(dt),
          gyro_meas_(gyro_meas),
          nominal_gyro_body_(nominal_gyro_body),
          nominal_bg_(nominal_bg),
          sigma_g_(sigma_g) {}

    template <typename T>
    bool operator()(
        const T* const dtheta_i,
        const T* const dtheta_j,
        const T* const dbg_i,
        const T* const dbg_j,
        const T* const td_ptr,
        T* residuals) const {
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        const Eigen::Map<const Vec3T> delta_theta_i(dtheta_i);
        const Eigen::Map<const Vec3T> delta_theta_j(dtheta_j);
        const Eigen::Map<const Vec3T> delta_bg_i(dbg_i);
        const Eigen::Map<const Vec3T> delta_bg_j(dbg_j);
        const T td = td_ptr[0];
        const T u = T(u_) + td / T(dt_);
        const Vec3T delta_bg = (T(1.0) - u) * delta_bg_i + u * delta_bg_j;
        const Vec3T delta_theta_rate = (delta_theta_j - delta_theta_i) / T(dt_);
        const Vec3T gyro_pred =
            nominal_gyro_body_.cast<T>() + nominal_bg_.cast<T>() + delta_bg + delta_theta_rate;
        Eigen::Map<Vec3T> res(residuals);
        res = (gyro_meas_.cast<T>() - gyro_pred) / T(sigma_g_);
        return true;
    }

    static ceres::CostFunction* Create(
        double u,
        double dt,
        const Eigen::Vector3d& gyro_meas,
        const Eigen::Vector3d& nominal_gyro_body,
        const Eigen::Vector3d& nominal_bg,
        double sigma_g) {
        return new ceres::AutoDiffCostFunction<ErrorStateAttitudeFactor, 3, 3, 3, 3, 3, 1>(
            new ErrorStateAttitudeFactor(u, dt, gyro_meas, nominal_gyro_body, nominal_bg, sigma_g));
    }

private:
    double u_;
    double dt_;
    Eigen::Vector3d gyro_meas_;
    Eigen::Vector3d nominal_gyro_body_;
    Eigen::Vector3d nominal_bg_;
    double sigma_g_;
};

}  // namespace ct_fgo_sim::factors
