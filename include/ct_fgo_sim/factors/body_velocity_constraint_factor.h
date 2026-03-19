#pragma once

#include "ct_fgo_sim/spline/bspline_evaluator.h"

#include <ceres/ceres.h>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>

namespace ct_fgo_sim::factors {

struct BodyVelocityConstraintFactor {
    BodyVelocityConstraintFactor(
        double t_meas,
        double dt,
        double t0,
        const Eigen::Vector3i& enable_axes,
        const Eigen::Vector3d& target_body_velocity,
        const Eigen::Vector3d& sigma_body_velocity)
        : t_meas_(t_meas),
          dt_(dt),
          t0_(t0),
          enable_axes_(enable_axes),
          target_body_velocity_(target_body_velocity),
          sigma_body_velocity_(sigma_body_velocity) {}

    template <typename T>
    bool operator()(const T* const cp0, const T* const cp1, const T* const cp2, const T* const cp3,
                    const T* const q_body_imu_ptr,
                    T* residuals) const {
        using SE3T = Sophus::SE3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using QuatT = Eigen::Quaternion<T>;
        using ResultT = typename spline::BSplineEvaluator::Result<T>;

        const SE3T t0 = Eigen::Map<const SE3T>(cp0);
        const SE3T t1 = Eigen::Map<const SE3T>(cp1);
        const SE3T t2 = Eigen::Map<const SE3T>(cp2);
        const SE3T t3 = Eigen::Map<const SE3T>(cp3);
        const QuatT q_body_imu(q_body_imu_ptr[3], q_body_imu_ptr[0], q_body_imu_ptr[1], q_body_imu_ptr[2]);

        const T u = (T(t_meas_) - T(t0_)) / T(dt_);
        const ResultT result = spline::BSplineEvaluator::Evaluate(u, T(dt_), t0, t1, t2, t3);
        const Vec3T v_body_frame = q_body_imu * result.v_body;

        residuals[0] = enable_axes_.x() ? (v_body_frame.x() - T(target_body_velocity_.x())) / T(sigma_body_velocity_.x()) : T(0);
        residuals[1] = enable_axes_.y() ? (v_body_frame.y() - T(target_body_velocity_.y())) / T(sigma_body_velocity_.y()) : T(0);
        residuals[2] = enable_axes_.z() ? (v_body_frame.z() - T(target_body_velocity_.z())) / T(sigma_body_velocity_.z()) : T(0);
        return true;
    }

    static ceres::CostFunction* Create(
        double t_meas,
        double dt,
        double t0,
        const Eigen::Vector3i& enable_axes,
        const Eigen::Vector3d& target_body_velocity,
        const Eigen::Vector3d& sigma_body_velocity) {
        return new ceres::AutoDiffCostFunction<BodyVelocityConstraintFactor, 3, 7, 7, 7, 7, 4>(
            new BodyVelocityConstraintFactor(t_meas, dt, t0, enable_axes, target_body_velocity, sigma_body_velocity));
    }

private:
    double t_meas_;
    double dt_;
    double t0_;
    Eigen::Vector3i enable_axes_;
    Eigen::Vector3d target_body_velocity_;
    Eigen::Vector3d sigma_body_velocity_;
};

}  // namespace ct_fgo_sim::factors
