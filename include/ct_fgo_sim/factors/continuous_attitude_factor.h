#pragma once

#include "ct_fgo_sim/spline/bspline_evaluator.h"

#include <ceres/ceres.h>
#include <sophus/se3.hpp>

namespace ct_fgo_sim::factors {

struct ContinuousAttitudeFactor {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ContinuousAttitudeFactor(
        double t_meas,
        const Eigen::Vector3d& gyro_meas,
        double dt,
        double t0,
        double sigma_g,
        const Sophus::SE3d& nominal_cp0,
        const Sophus::SE3d& nominal_cp1,
        const Sophus::SE3d& nominal_cp2,
        const Sophus::SE3d& nominal_cp3,
        const Eigen::Vector3d& nominal_gyro_center)
        : t_meas_(t_meas),
          gyro_meas_(gyro_meas),
          dt_(dt),
          t0_(t0),
          sigma_g_(sigma_g),
          nominal_cp0_(nominal_cp0),
          nominal_cp1_(nominal_cp1),
          nominal_cp2_(nominal_cp2),
          nominal_cp3_(nominal_cp3),
          nominal_gyro_center_(nominal_gyro_center) {}

    template <typename T>
    bool operator()(const T* const cp0, const T* const cp1, const T* const cp2, const T* const cp3,
                    const T* const bgi, const T* const bgj,
                    const T* const td_ptr,
                    T* residuals) const {
        using SE3T = Sophus::SE3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;

        const SE3T full0 = Eigen::Map<const SE3T>(cp0);
        const SE3T full1 = Eigen::Map<const SE3T>(cp1);
        const SE3T full2 = Eigen::Map<const SE3T>(cp2);
        const SE3T full3 = Eigen::Map<const SE3T>(cp3);
        const SE3T nominal0 = nominal_cp0_.template cast<T>();
        const SE3T nominal1 = nominal_cp1_.template cast<T>();
        const SE3T nominal2 = nominal_cp2_.template cast<T>();
        const SE3T nominal3 = nominal_cp3_.template cast<T>();

        const SE3T delta0 = nominal0.inverse() * full0;
        const SE3T delta1 = nominal1.inverse() * full1;
        const SE3T delta2 = nominal2.inverse() * full2;
        const SE3T delta3 = nominal3.inverse() * full3;

        Eigen::Map<const Vec3T> bg0_vec(bgi);
        Eigen::Map<const Vec3T> bg1_vec(bgj);
        const T td = td_ptr[0];
        const T u = (T(t_meas_) + td - T(t0_)) / T(dt_);
        const auto delta_result = spline::BSplineEvaluator::Evaluate(u, T(dt_), delta0, delta1, delta2, delta3);
        const Vec3T bg = bg0_vec * (T(1) - u) + bg1_vec * u;
        const Vec3T gyro_pred = nominal_gyro_center_.template cast<T>() + delta_result.w_body + bg;

        residuals[0] = (T(gyro_meas_.x()) - gyro_pred.x()) / T(sigma_g_);
        residuals[1] = (T(gyro_meas_.y()) - gyro_pred.y()) / T(sigma_g_);
        residuals[2] = (T(gyro_meas_.z()) - gyro_pred.z()) / T(sigma_g_);
        return true;
    }

    static ceres::CostFunction* Create(
        double t_meas,
        const Eigen::Vector3d& gyro_meas,
        double dt,
        double t0,
        double sigma_g,
        const Sophus::SE3d& nominal_cp0,
        const Sophus::SE3d& nominal_cp1,
        const Sophus::SE3d& nominal_cp2,
        const Sophus::SE3d& nominal_cp3,
        const Eigen::Vector3d& nominal_gyro_center) {
        return new ceres::AutoDiffCostFunction<ContinuousAttitudeFactor, 3,
            7, 7, 7, 7,
            3, 3,
            1>(
            new ContinuousAttitudeFactor(
                t_meas,
                gyro_meas,
                dt,
                t0,
                sigma_g,
                nominal_cp0,
                nominal_cp1,
                nominal_cp2,
                nominal_cp3,
                nominal_gyro_center));
    }

private:
    double t_meas_;
    Eigen::Vector3d gyro_meas_;
    double dt_;
    double t0_;
    double sigma_g_;
    Sophus::SE3d nominal_cp0_;
    Sophus::SE3d nominal_cp1_;
    Sophus::SE3d nominal_cp2_;
    Sophus::SE3d nominal_cp3_;
    Eigen::Vector3d nominal_gyro_center_;
};

}  // namespace ct_fgo_sim::factors
