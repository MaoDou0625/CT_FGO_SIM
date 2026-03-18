#pragma once

#include "ct_fgo_sim/spline/bspline_evaluator.h"

#include <ceres/ceres.h>
#include <sophus/se3.hpp>

namespace ct_fgo_sim::factors {

struct ContinuousInertialFactor {
    ContinuousInertialFactor(
        double t_meas,
        const Eigen::Vector3d& accel_meas,
        const Eigen::Vector3d& gyro_meas,
        const Eigen::Vector3d& gravity_n,
        const Eigen::Vector3d& omega_ie_n,
        double dt,
        double t0,
        double sigma_a,
        double sigma_g)
        : t_meas_(t_meas),
          accel_meas_(accel_meas),
          gyro_meas_(gyro_meas),
          gravity_n_(gravity_n),
          omega_ie_n_(omega_ie_n),
          dt_(dt),
          t0_(t0),
          sigma_a_(sigma_a),
          sigma_g_(sigma_g) {}

    template <typename T>
    bool operator()(const T* const cp0, const T* const cp1, const T* const cp2, const T* const cp3,
                    const T* const bg0, const T* const bg1, const T* const bg2, const T* const bg3,
                    const T* const ba0, const T* const ba1, const T* const ba2, const T* const ba3,
                    const T* const lever_arm_ptr,
                    const T* const td_ptr,
                    T* residuals) const {
        using SE3T = Sophus::SE3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using ResultT = typename spline::BSplineEvaluator::Result<T>;

        Eigen::Map<const SE3T> t0(cp0);
        Eigen::Map<const SE3T> t1(cp1);
        Eigen::Map<const SE3T> t2(cp2);
        Eigen::Map<const SE3T> t3(cp3);
        Eigen::Map<const Vec3T> bg1_vec(bg1);
        Eigen::Map<const Vec3T> bg2_vec(bg2);
        Eigen::Map<const Vec3T> ba1_vec(ba1);
        Eigen::Map<const Vec3T> ba2_vec(ba2);
        Eigen::Map<const Vec3T> lever_arm(lever_arm_ptr);
        (void)bg0;
        (void)bg3;
        (void)ba0;
        (void)ba3;

        const T td = td_ptr[0];
        const T u = (T(t_meas_) + td - T(t0_)) / T(dt_);
        const ResultT result = spline::BSplineEvaluator::Evaluate(u, T(dt_), t0, t1, t2, t3);

        const Vec3T bg = bg1_vec * (T(1) - u) + bg2_vec * u;
        const Vec3T ba = ba1_vec * (T(1) - u) + ba2_vec * u;

        const Vec3T omega_ie_b = result.pose.so3().inverse() * omega_ie_n_.cast<T>();
        const Vec3T gyro_pred = result.w_body + bg + omega_ie_b;

        const Vec3T coriolis = T(2) * omega_ie_n_.cast<T>().cross(result.v_world);
        const Vec3T acc_n = result.a_world - gravity_n_.cast<T>() + coriolis;
        const Vec3T acc_b = result.pose.so3().inverse() * acc_n;
        const Vec3T acc_lever = result.alpha_body.cross(lever_arm) +
                                result.w_body.cross(result.w_body.cross(lever_arm));
        const Vec3T accel_pred = acc_b + acc_lever + ba;

        residuals[0] = (gyro_meas_.x() - gyro_pred.x()) / T(sigma_g_);
        residuals[1] = (gyro_meas_.y() - gyro_pred.y()) / T(sigma_g_);
        residuals[2] = (gyro_meas_.z() - gyro_pred.z()) / T(sigma_g_);
        residuals[3] = (accel_meas_.x() - accel_pred.x()) / T(sigma_a_);
        residuals[4] = (accel_meas_.y() - accel_pred.y()) / T(sigma_a_);
        residuals[5] = (accel_meas_.z() - accel_pred.z()) / T(sigma_a_);
        return true;
    }

    static ceres::CostFunction* Create(
        double t_meas,
        const Eigen::Vector3d& accel_meas,
        const Eigen::Vector3d& gyro_meas,
        const Eigen::Vector3d& gravity_n,
        const Eigen::Vector3d& omega_ie_n,
        double dt,
        double t0,
        double sigma_a,
        double sigma_g) {
        return new ceres::AutoDiffCostFunction<ContinuousInertialFactor, 6,
            7, 7, 7, 7,
            3, 3, 3, 3,
            3, 3, 3, 3,
            3, 1>(
            new ContinuousInertialFactor(
                t_meas, accel_meas, gyro_meas, gravity_n, omega_ie_n, dt, t0, sigma_a, sigma_g));
    }

private:
    double t_meas_;
    Eigen::Vector3d accel_meas_;
    Eigen::Vector3d gyro_meas_;
    Eigen::Vector3d gravity_n_;
    Eigen::Vector3d omega_ie_n_;
    double dt_;
    double t0_;
    double sigma_a_;
    double sigma_g_;
};

}  // namespace ct_fgo_sim::factors
