#pragma once

#include "ct_fgo_sim/spline/bspline_evaluator.h"

#include <ceres/ceres.h>
#include <sophus/se3.hpp>

namespace ct_fgo_sim::factors {

struct ContinuousGnssFactor {
    ContinuousGnssFactor(double t, double dt, double t0, const Eigen::Vector3d& pos_meas, const Eigen::Matrix3d& sqrt_info)
        : t_(t), dt_(dt), t0_(t0), pos_meas_(pos_meas), sqrt_info_(sqrt_info) {}

    template <typename T>
    bool operator()(const T* const p0, const T* const p1, const T* const p2, const T* const p3,
                    const T* const lever_arm_ptr, T* residuals) const {
        using SE3T = Sophus::SE3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using ResultT = typename spline::BSplineEvaluator::Result<T>;

        Eigen::Map<const SE3T> t0(p0);
        Eigen::Map<const SE3T> t1(p1);
        Eigen::Map<const SE3T> t2(p2);
        Eigen::Map<const SE3T> t3(p3);
        Eigen::Map<const Vec3T> lever_arm(lever_arm_ptr);

        const T u = (T(t_) - T(t0_)) / T(dt_);
        const ResultT result = spline::BSplineEvaluator::Evaluate(u, T(dt_), t0, t1, t2, t3);
        const Vec3T p = result.pose.translation() + result.pose.so3() * lever_arm;

        Eigen::Map<Vec3T> res(residuals);
        res = sqrt_info_.cast<T>() * (p - pos_meas_.cast<T>());
        return true;
    }

    static ceres::CostFunction* Create(
        double t, double dt, double t0, const Eigen::Vector3d& pos_meas, const Eigen::Matrix3d& sqrt_info) {
        return new ceres::AutoDiffCostFunction<ContinuousGnssFactor, 3, 7, 7, 7, 7, 3>(
            new ContinuousGnssFactor(t, dt, t0, pos_meas, sqrt_info));
    }

private:
    double t_;
    double dt_;
    double t0_;
    Eigen::Vector3d pos_meas_;
    Eigen::Matrix3d sqrt_info_;
};

}  // namespace ct_fgo_sim::factors
