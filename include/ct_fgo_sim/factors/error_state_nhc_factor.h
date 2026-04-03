#pragma once

#include <ceres/ceres.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/so3.hpp>

namespace ct_fgo_sim::factors {

struct ErrorStateBodyVelocityNhcFactor {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ErrorStateBodyVelocityNhcFactor(
        double u,
        const Eigen::Quaterniond& nominal_q_nb,
        const Eigen::Vector3d& nominal_vel_ned,
        const Eigen::Vector3d& target_vel_body_mps,
        const Eigen::Vector3d& sigma_body_mps)
        : u_(u),
          nominal_q_nb_(nominal_q_nb),
          nominal_vel_ned_(nominal_vel_ned),
          target_vel_body_mps_(target_vel_body_mps) {
        inv_sigma_body_mps_.x() = sigma_body_mps.x() > 0.0 ? 1.0 / sigma_body_mps.x() : 0.0;
        inv_sigma_body_mps_.y() = sigma_body_mps.y() > 0.0 ? 1.0 / sigma_body_mps.y() : 0.0;
        inv_sigma_body_mps_.z() = sigma_body_mps.z() > 0.0 ? 1.0 / sigma_body_mps.z() : 0.0;
    }

    template <typename T>
    bool operator()(
        const T* const dtheta_i,
        const T* const dtheta_j,
        const T* const dvel_i,
        const T* const dvel_j,
        const T* const q_body_imu_ptr,
        T* residuals) const {
        using Vec3T = Eigen::Matrix<T, 3, 1>;

        const Eigen::Map<const Vec3T> delta_theta_i(dtheta_i);
        const Eigen::Map<const Vec3T> delta_theta_j(dtheta_j);
        const Eigen::Map<const Vec3T> delta_vel_i(dvel_i);
        const Eigen::Map<const Vec3T> delta_vel_j(dvel_j);
        const Eigen::Map<const Eigen::Quaternion<T>> q_body_imu(q_body_imu_ptr);

        const T one_minus_u = T(1.0) - T(u_);
        const Vec3T delta_theta = one_minus_u * delta_theta_i + T(u_) * delta_theta_j;
        const Vec3T delta_vel_ned = one_minus_u * delta_vel_i + T(u_) * delta_vel_j;

        const Sophus::SO3<T> q_nb_full = Sophus::SO3<T>(nominal_q_nb_.cast<T>()) * Sophus::SO3<T>::exp(delta_theta);
        const Vec3T full_vel_ned = nominal_vel_ned_.cast<T>() + delta_vel_ned;
        const Vec3T full_vel_imu = q_nb_full.inverse() * full_vel_ned;
        const Vec3T full_vel_body = q_body_imu * full_vel_imu;

        const Vec3T vel_error = full_vel_body - target_vel_body_mps_.cast<T>();
        Eigen::Map<Vec3T> res(residuals);
        res = inv_sigma_body_mps_.cast<T>().cwiseProduct(vel_error);
        return true;
    }

    static ceres::CostFunction* Create(
        double u,
        const Eigen::Quaterniond& nominal_q_nb,
        const Eigen::Vector3d& nominal_vel_ned,
        const Eigen::Vector3d& target_vel_body_mps,
        const Eigen::Vector3d& sigma_body_mps) {
        return new ceres::AutoDiffCostFunction<ErrorStateBodyVelocityNhcFactor, 3, 3, 3, 3, 3, 4>(
            new ErrorStateBodyVelocityNhcFactor(
                u, nominal_q_nb, nominal_vel_ned, target_vel_body_mps, sigma_body_mps));
    }

private:
    double u_;
    Eigen::Quaterniond nominal_q_nb_;
    Eigen::Vector3d nominal_vel_ned_;
    Eigen::Vector3d target_vel_body_mps_;
    Eigen::Vector3d inv_sigma_body_mps_;
};

}  // namespace ct_fgo_sim::factors
