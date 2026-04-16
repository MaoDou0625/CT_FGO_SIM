#pragma once

#include <ceres/ceres.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/so3.hpp>

#include <algorithm>
#include <cmath>

namespace ct_fgo_sim::factors {

struct ErrorStateYawBiasFactor {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ErrorStateYawBiasFactor(
        double u,
        const Eigen::Quaterniond& nominal_q_nb,
        double heading_meas_rad,
        double sigma_heading_rad)
        : u_(u),
          nominal_q_nb_(nominal_q_nb),
          heading_meas_rad_(heading_meas_rad),
          inv_sigma_heading_(1.0 / std::max(1.0e-6, sigma_heading_rad)) {}

    template <typename T>
    bool operator()(
        const T* const dtheta_i,
        const T* const dtheta_j,
        const T* const yaw_bias_rad,
        T* residuals) const {
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        const Eigen::Map<const Vec3T> delta_theta_i(dtheta_i);
        const Eigen::Map<const Vec3T> delta_theta_j(dtheta_j);
        const T one_minus_u = T(1.0) - T(u_);
        const Vec3T delta_theta = one_minus_u * delta_theta_i + T(u_) * delta_theta_j;

        const Sophus::SO3<T> nominal_so3(nominal_q_nb_.cast<T>());
        const Eigen::Quaternion<T> q_yaw(Eigen::AngleAxis<T>(yaw_bias_rad[0], Vec3T::UnitZ()));
        const Sophus::SO3<T> yaw_so3(q_yaw);
        const Sophus::SO3<T> full_so3 = yaw_so3 * nominal_so3 * Sophus::SO3<T>::exp(delta_theta);
        const Eigen::Matrix<T, 3, 1> forward_n = full_so3.matrix().col(0);
        const T heading_pred = ceres::atan2(forward_n.y(), forward_n.x());
        const T dpsi = heading_pred - T(heading_meas_rad_);
        const T wrapped = ceres::atan2(ceres::sin(dpsi), ceres::cos(dpsi));
        residuals[0] = T(inv_sigma_heading_) * wrapped;
        return true;
    }

    static ceres::CostFunction* Create(
        double u,
        const Eigen::Quaterniond& nominal_q_nb,
        double heading_meas_rad,
        double sigma_heading_rad) {
        return new ceres::AutoDiffCostFunction<ErrorStateYawBiasFactor, 1, 3, 3, 1>(
            new ErrorStateYawBiasFactor(u, nominal_q_nb, heading_meas_rad, sigma_heading_rad));
    }

private:
    double u_ = 0.0;
    Eigen::Quaterniond nominal_q_nb_ = Eigen::Quaterniond::Identity();
    double heading_meas_rad_ = 0.0;
    double inv_sigma_heading_ = 1.0;
};

}  // namespace ct_fgo_sim::factors
