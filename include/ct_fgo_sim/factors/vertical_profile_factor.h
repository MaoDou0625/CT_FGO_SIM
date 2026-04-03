#pragma once

#include "ct_fgo_sim/spline/bspline_evaluator.h"

#include <ceres/ceres.h>
#include <sophus/se3.hpp>

namespace ct_fgo_sim::factors {

struct ContinuousVerticalGnssFactor {
    ContinuousVerticalGnssFactor(
        double t,
        double dt,
        double t0,
        double meas_z_ned,
        double sigma_vertical_m)
        : t_(t),
          dt_(dt),
          t0_(t0),
          meas_z_ned_(meas_z_ned),
          inv_sigma_vertical_(1.0 / std::max(1.0e-6, sigma_vertical_m)) {}

    template <typename T>
    bool operator()(const T* const p0, const T* const p1, const T* const p2, const T* const p3,
                    const T* const dz0_ptr, const T* const dz1_ptr,
                    const T* const lever_arm_ptr, T* residuals) const {
        using SE3T = Sophus::SE3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using ResultT = typename spline::BSplineEvaluator::Result<T>;

        const SE3T t0 = Eigen::Map<const SE3T>(p0);
        const SE3T t1 = Eigen::Map<const SE3T>(p1);
        const SE3T t2 = Eigen::Map<const SE3T>(p2);
        const SE3T t3 = Eigen::Map<const SE3T>(p3);
        Eigen::Map<const Vec3T> lever_arm(lever_arm_ptr);

        const T u = (T(t_) - T(t0_)) / T(dt_);
        const ResultT result = spline::BSplineEvaluator::Evaluate(u, T(dt_), t0, t1, t2, t3);
        const T dz = (T(1.0) - u) * dz0_ptr[0] + u * dz1_ptr[0];
        const Vec3T p = result.pose.translation() + result.pose.so3() * lever_arm;
        residuals[0] = T(inv_sigma_vertical_) * (p.z() + dz - T(meas_z_ned_));
        return true;
    }

    static ceres::CostFunction* Create(
        double t,
        double dt,
        double t0,
        double meas_z_ned,
        double sigma_vertical_m) {
        return new ceres::AutoDiffCostFunction<ContinuousVerticalGnssFactor, 1, 7, 7, 7, 7, 1, 1, 3>(
            new ContinuousVerticalGnssFactor(t, dt, t0, meas_z_ned, sigma_vertical_m));
    }

private:
    double t_;
    double dt_;
    double t0_;
    double meas_z_ned_;
    double inv_sigma_vertical_;
};

struct VerticalSmoothnessFactor {
    explicit VerticalSmoothnessFactor(double sigma_m)
        : inv_sigma_(1.0 / std::max(1.0e-6, sigma_m)) {}

    template <typename T>
    bool operator()(const T* const zi, const T* const zj, T* residuals) const {
        residuals[0] = T(inv_sigma_) * (zj[0] - zi[0]);
        return true;
    }

    static ceres::CostFunction* Create(double sigma_m) {
        return new ceres::AutoDiffCostFunction<VerticalSmoothnessFactor, 1, 1, 1>(
            new VerticalSmoothnessFactor(sigma_m));
    }

private:
    double inv_sigma_;
};

struct VerticalZeroPriorFactor {
    explicit VerticalZeroPriorFactor(double sigma_m)
        : inv_sigma_(1.0 / std::max(1.0e-6, sigma_m)) {}

    template <typename T>
    bool operator()(const T* const zi, T* residuals) const {
        residuals[0] = T(inv_sigma_) * zi[0];
        return true;
    }

    static ceres::CostFunction* Create(double sigma_m) {
        return new ceres::AutoDiffCostFunction<VerticalZeroPriorFactor, 1, 1>(
            new VerticalZeroPriorFactor(sigma_m));
    }

private:
    double inv_sigma_;
};

}  // namespace ct_fgo_sim::factors
