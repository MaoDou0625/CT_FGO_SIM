#pragma once

#include <ceres/ceres.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace ct_fgo_sim::factors {

struct ErrorStateGnssHorizontalLeverArmFactor {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ErrorStateGnssHorizontalLeverArmFactor(
        double u,
        const Eigen::Vector3d& nominal_pos_ned,
        const Eigen::Quaterniond& nominal_q_nb,
        const Eigen::Vector3d& lever_arm,
        const Eigen::Vector3d& pos_meas_ned,
        double sigma_horizontal_m)
        : u_(u),
          nominal_pos_ned_(nominal_pos_ned),
          nominal_q_nb_(nominal_q_nb),
          lever_arm_(lever_arm),
          pos_meas_ned_(pos_meas_ned),
          inv_sigma_horizontal_(1.0 / std::max(1.0e-6, sigma_horizontal_m)) {}

    template <typename T>
    bool operator()(
        const T* const dp_i,
        const T* const dp_j,
        const T* const dtheta_i,
        const T* const dtheta_j,
        T* residuals) const {
        using Vec2T = Eigen::Matrix<T, 2, 1>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        const Eigen::Map<const Vec3T> delta_p_i(dp_i);
        const Eigen::Map<const Vec3T> delta_p_j(dp_j);
        const Eigen::Map<const Vec3T> delta_theta_i(dtheta_i);
        const Eigen::Map<const Vec3T> delta_theta_j(dtheta_j);

        const T one_minus_u = T(1.0) - T(u_);
        const Vec3T delta_p = one_minus_u * delta_p_i + T(u_) * delta_p_j;
        const Vec3T delta_theta = one_minus_u * delta_theta_i + T(u_) * delta_theta_j;

        const Vec3T lever_arm_nominal = nominal_q_nb_.cast<T>() * lever_arm_.cast<T>();
        const Vec3T lever_arm_error = nominal_q_nb_.cast<T>() * delta_theta.cross(lever_arm_.cast<T>());
        const Vec3T pos_pred = nominal_pos_ned_.cast<T>() + lever_arm_nominal + lever_arm_error + delta_p;
        const Vec3T pos_error = pos_pred - pos_meas_ned_.cast<T>();

        Eigen::Map<Vec2T> res(residuals);
        res << T(inv_sigma_horizontal_) * pos_error.x(),
               T(inv_sigma_horizontal_) * pos_error.y();
        return true;
    }

    static ceres::CostFunction* Create(
        double u,
        const Eigen::Vector3d& nominal_pos_ned,
        const Eigen::Quaterniond& nominal_q_nb,
        const Eigen::Vector3d& lever_arm,
        const Eigen::Vector3d& pos_meas_ned,
        double sigma_horizontal_m) {
        return new ceres::AutoDiffCostFunction<ErrorStateGnssHorizontalLeverArmFactor, 2, 3, 3, 3, 3>(
            new ErrorStateGnssHorizontalLeverArmFactor(
                u, nominal_pos_ned, nominal_q_nb, lever_arm, pos_meas_ned, sigma_horizontal_m));
    }

private:
    double u_;
    Eigen::Vector3d nominal_pos_ned_;
    Eigen::Quaterniond nominal_q_nb_;
    Eigen::Vector3d lever_arm_;
    Eigen::Vector3d pos_meas_ned_;
    double inv_sigma_horizontal_;
};

struct ErrorStateGnssVerticalLeverArmFactor {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ErrorStateGnssVerticalLeverArmFactor(
        double u,
        const Eigen::Vector3d& nominal_pos_ned,
        const Eigen::Quaterniond& nominal_q_nb,
        const Eigen::Vector3d& lever_arm,
        const Eigen::Vector3d& pos_meas_ned,
        double sigma_vertical_m)
        : u_(u),
          nominal_pos_ned_(nominal_pos_ned),
          nominal_q_nb_(nominal_q_nb),
          lever_arm_(lever_arm),
          pos_meas_ned_(pos_meas_ned),
          inv_sigma_vertical_(1.0 / std::max(1.0e-6, sigma_vertical_m)) {}

    template <typename T>
    bool operator()(
        const T* const dp_i,
        const T* const dp_j,
        const T* const dtheta_i,
        const T* const dtheta_j,
        T* residuals) const {
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        const Eigen::Map<const Vec3T> delta_p_i(dp_i);
        const Eigen::Map<const Vec3T> delta_p_j(dp_j);
        const Eigen::Map<const Vec3T> delta_theta_i(dtheta_i);
        const Eigen::Map<const Vec3T> delta_theta_j(dtheta_j);

        const T one_minus_u = T(1.0) - T(u_);
        const Vec3T delta_p = one_minus_u * delta_p_i + T(u_) * delta_p_j;
        const Vec3T delta_theta = one_minus_u * delta_theta_i + T(u_) * delta_theta_j;

        const Vec3T lever_arm_nominal = nominal_q_nb_.cast<T>() * lever_arm_.cast<T>();
        const Vec3T lever_arm_error = nominal_q_nb_.cast<T>() * delta_theta.cross(lever_arm_.cast<T>());
        const Vec3T pos_pred = nominal_pos_ned_.cast<T>() + lever_arm_nominal + lever_arm_error + delta_p;

        residuals[0] = T(inv_sigma_vertical_) * (pos_pred.z() - T(pos_meas_ned_.z()));
        return true;
    }

    static ceres::CostFunction* Create(
        double u,
        const Eigen::Vector3d& nominal_pos_ned,
        const Eigen::Quaterniond& nominal_q_nb,
        const Eigen::Vector3d& lever_arm,
        const Eigen::Vector3d& pos_meas_ned,
        double sigma_vertical_m) {
        return new ceres::AutoDiffCostFunction<ErrorStateGnssVerticalLeverArmFactor, 1, 3, 3, 3, 3>(
            new ErrorStateGnssVerticalLeverArmFactor(
                u, nominal_pos_ned, nominal_q_nb, lever_arm, pos_meas_ned, sigma_vertical_m));
    }

private:
    double u_;
    Eigen::Vector3d nominal_pos_ned_;
    Eigen::Quaterniond nominal_q_nb_;
    Eigen::Vector3d lever_arm_;
    Eigen::Vector3d pos_meas_ned_;
    double inv_sigma_vertical_;
};

}  // namespace ct_fgo_sim::factors
