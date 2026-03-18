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
        const Eigen::Vector3d& origin_blh,
        double dt,
        double t0,
        double sigma_a,
        double sigma_g)
        : t_meas_(t_meas),
          accel_meas_(accel_meas),
          gyro_meas_(gyro_meas),
          origin_blh_(origin_blh),
          dt_(dt),
          t0_(t0),
          sigma_a_(sigma_a),
          sigma_g_(sigma_g) {}

    template <typename T>
    bool operator()(const T* const cp0, const T* const cp1, const T* const cp2, const T* const cp3,
                    const T* const bgi, const T* const bgj,
                    const T* const bai, const T* const baj,
                    const T* const lever_arm_ptr,
                    const T* const td_ptr,
                    T* residuals) const {
        using SE3T = Sophus::SE3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using ResultT = typename spline::BSplineEvaluator::Result<T>;

        const SE3T t0 = Eigen::Map<const SE3T>(cp0);
        const SE3T t1 = Eigen::Map<const SE3T>(cp1);
        const SE3T t2 = Eigen::Map<const SE3T>(cp2);
        const SE3T t3 = Eigen::Map<const SE3T>(cp3);
        Eigen::Map<const Vec3T> bg0_vec(bgi);
        Eigen::Map<const Vec3T> bg1_vec(bgj);
        Eigen::Map<const Vec3T> ba0_vec(bai);
        Eigen::Map<const Vec3T> ba1_vec(baj);
        Eigen::Map<const Vec3T> lever_arm(lever_arm_ptr);

        const T td = td_ptr[0];
        const T u = (T(t_meas_) + td - T(t0_)) / T(dt_);
        const ResultT result = spline::BSplineEvaluator::Evaluate(u, T(dt_), t0, t1, t2, t3);

        const Vec3T bg = bg0_vec * (T(1) - u) + bg1_vec * u;
        const Vec3T ba = ba0_vec * (T(1) - u) + ba1_vec * u;

        const NavInfo<T> nav = EvaluateNavInfo(result.pose.translation(), result.v_world);
        const Vec3T omega_in_b = result.pose.so3().inverse() * (nav.omega_ie_n + nav.omega_en_n);
        const Vec3T gyro_pred = result.w_body + bg + omega_in_b;

        const Vec3T coriolis = (T(2) * nav.omega_ie_n + nav.omega_en_n).cross(result.v_world);
        const Vec3T acc_n = result.a_world - nav.gravity_n + coriolis;
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
        const Eigen::Vector3d& origin_blh,
        double dt,
        double t0,
        double sigma_a,
        double sigma_g) {
        return new ceres::AutoDiffCostFunction<ContinuousInertialFactor, 6,
            7, 7, 7, 7,
            3, 3,
            3, 3,
            3, 1>(
            new ContinuousInertialFactor(
                t_meas, accel_meas, gyro_meas, origin_blh, dt, t0, sigma_a, sigma_g));
    }

private:
    template <typename T>
    struct NavInfo {
        Eigen::Matrix<T, 3, 1> gravity_n = Eigen::Matrix<T, 3, 1>::Zero();
        Eigen::Matrix<T, 3, 1> omega_ie_n = Eigen::Matrix<T, 3, 1>::Zero();
        Eigen::Matrix<T, 3, 1> omega_en_n = Eigen::Matrix<T, 3, 1>::Zero();
    };

    template <typename T>
    NavInfo<T> EvaluateNavInfo(const Eigen::Matrix<T, 3, 1>& p_enu, const Eigen::Matrix<T, 3, 1>& v_world) const {
        constexpr double kRa = 6378137.0;
        constexpr double kE1 = 0.0066943799901413156;
        constexpr double kWie = 7.2921151467e-5;

        const T lat0 = T(origin_blh_.x());
        const T h0 = T(origin_blh_.z());
        const T sin_lat0 = ceres::sin(lat0);
        const T den0 = T(1.0) - T(kE1) * sin_lat0 * sin_lat0;
        const T sqrt_den0 = ceres::sqrt(den0);
        const T rn0 = T(kRa) / sqrt_den0;
        const T rm0 = T(kRa * (1.0 - kE1)) / (den0 * sqrt_den0);

        const T lat = lat0 + p_enu.y() / (rm0 + h0);
        const T h = h0 + p_enu.z();
        const T sin_lat = ceres::sin(lat);
        const T den = T(1.0) - T(kE1) * sin_lat * sin_lat;
        const T sqrt_den = ceres::sqrt(den);
        const T rn = T(kRa) / sqrt_den;
        const T rm = T(kRa * (1.0 - kE1)) / (den * sqrt_den);

        NavInfo<T> out;
        const T sin2 = sin_lat * sin_lat;
        const T gravity = T(9.7803267715) * (T(1.0) + T(0.0052790414) * sin2 + T(0.0000232718) * sin2 * sin2) +
                          h * (T(0.0000000043977311) * sin2 - T(0.0000030876910891)) +
                          T(0.0000000000007211) * h * h;
        out.gravity_n = Eigen::Matrix<T, 3, 1>(T(0), T(0), -gravity);
        out.omega_ie_n = Eigen::Matrix<T, 3, 1>(T(kWie) * ceres::cos(lat), T(0), -T(kWie) * sin_lat);
        out.omega_en_n = Eigen::Matrix<T, 3, 1>(
            -v_world.y() / (rm + h),
             v_world.x() / (rn + h),
             v_world.x() * ceres::tan(lat) / (rn + h));
        return out;
    }

    double t_meas_;
    Eigen::Vector3d accel_meas_;
    Eigen::Vector3d gyro_meas_;
    Eigen::Vector3d origin_blh_;
    double dt_;
    double t0_;
    double sigma_a_;
    double sigma_g_;
};

}  // namespace ct_fgo_sim::factors
