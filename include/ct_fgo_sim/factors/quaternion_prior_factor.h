#pragma once

#include <ceres/ceres.h>
#include <Eigen/Geometry>

namespace ct_fgo_sim::factors {

struct QuaternionPriorFactor {
    QuaternionPriorFactor(const Eigen::Quaterniond& q_prior, double sigma_rad)
        : q_prior_(q_prior), sigma_rad_(sigma_rad) {}

    template <typename T>
    bool operator()(const T* const q_ptr, T* residuals) const {
        const Eigen::Quaternion<T> q_est(q_ptr[3], q_ptr[0], q_ptr[1], q_ptr[2]);
        const Eigen::Quaternion<T> q_prior(T(q_prior_.w()), T(q_prior_.x()), T(q_prior_.y()), T(q_prior_.z()));
        const Eigen::Quaternion<T> q_err = q_prior.conjugate() * q_est;
        const Eigen::Matrix<T, 3, 1> rot_vec = T(2.0) * q_err.vec();
        residuals[0] = rot_vec.x() / T(sigma_rad_);
        residuals[1] = rot_vec.y() / T(sigma_rad_);
        residuals[2] = rot_vec.z() / T(sigma_rad_);
        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Quaterniond& q_prior, double sigma_rad) {
        return new ceres::AutoDiffCostFunction<QuaternionPriorFactor, 3, 4>(
            new QuaternionPriorFactor(q_prior, sigma_rad));
    }

private:
    Eigen::Quaterniond q_prior_;
    double sigma_rad_;
};

}  // namespace ct_fgo_sim::factors
