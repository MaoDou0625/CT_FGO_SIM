#pragma once

#include <ceres/ceres.h>
#include <Eigen/Core>

namespace ct_fgo_sim::factors {

struct ErrorStateGnssFactor {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ErrorStateGnssFactor(
        double u,
        const Eigen::Vector3d& nominal_pos_enu,
        const Eigen::Vector3d& pos_meas_enu,
        const Eigen::Matrix3d& sqrt_info)
        : u_(u),
          nominal_pos_enu_(nominal_pos_enu),
          pos_meas_enu_(pos_meas_enu),
          sqrt_info_(sqrt_info) {}

    template <typename T>
    bool operator()(const T* const dp_i, const T* const dp_j, T* residuals) const {
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        const Eigen::Map<const Vec3T> delta_p_i(dp_i);
        const Eigen::Map<const Vec3T> delta_p_j(dp_j);
        const Vec3T delta_p = (T(1.0) - T(u_)) * delta_p_i + T(u_) * delta_p_j;
        const Vec3T pos_pred = nominal_pos_enu_.cast<T>() + delta_p;
        Eigen::Map<Vec3T> res(residuals);
        res = sqrt_info_.cast<T>() * (pos_pred - pos_meas_enu_.cast<T>());
        return true;
    }

    static ceres::CostFunction* Create(
        double u,
        const Eigen::Vector3d& nominal_pos_enu,
        const Eigen::Vector3d& pos_meas_enu,
        const Eigen::Matrix3d& sqrt_info) {
        return new ceres::AutoDiffCostFunction<ErrorStateGnssFactor, 3, 3, 3>(
            new ErrorStateGnssFactor(u, nominal_pos_enu, pos_meas_enu, sqrt_info));
    }

private:
    double u_;
    Eigen::Vector3d nominal_pos_enu_;
    Eigen::Vector3d pos_meas_enu_;
    Eigen::Matrix3d sqrt_info_;
};

}  // namespace ct_fgo_sim::factors
