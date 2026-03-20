#pragma once

#include "ct_fgo_sim/navigation/earth.h"

#include <ceres/ceres.h>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Geometry>

#include <algorithm>

namespace ct_fgo_sim::factors {

struct ErrorStateProcessFactor {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using Matrix3d = Eigen::Matrix3d;
    using Matrix15d = Eigen::Matrix<double, 15, 15>;
    using Matrix15x12d = Eigen::Matrix<double, 15, 12>;
    using Vector15d = Eigen::Matrix<double, 15, 1>;

    ErrorStateProcessFactor(
        double dt,
        const Eigen::Vector3d& nominal_blh,
        const Eigen::Vector3d& nominal_vel_ned,
        const Eigen::Quaterniond& nominal_q_nb,
        const Eigen::Vector3d& nominal_specific_force_body,
        double sigma_gyro_rps,
        double sigma_accel_mps2,
        double sigma_bg_std,
        double sigma_ba_std,
        double bias_tau_s)
        : phi_(BuildPhi(
              dt,
              nominal_blh,
              nominal_vel_ned,
              nominal_q_nb,
              nominal_specific_force_body,
              sigma_gyro_rps,
              sigma_accel_mps2,
              sigma_bg_std,
              sigma_ba_std,
              bias_tau_s)),
          sqrt_info_(BuildSqrtInfo(
              dt,
              nominal_blh,
              nominal_vel_ned,
              nominal_q_nb,
              nominal_specific_force_body,
              sigma_gyro_rps,
              sigma_accel_mps2,
              sigma_bg_std,
              sigma_ba_std,
              bias_tau_s)) {}

    template <typename T>
    bool operator()(
        const T* const dtheta_i, const T* const dv_i, const T* const dp_i, const T* const dbg_i, const T* const dba_i,
        const T* const dtheta_j, const T* const dv_j, const T* const dp_j, const T* const dbg_j, const T* const dba_j,
        T* residuals) const {
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using Vec15T = Eigen::Matrix<T, 15, 1>;
        using Mat15T = Eigen::Matrix<T, 15, 15>;

        const Eigen::Map<const Vec3T> phi_i(dtheta_i);
        const Eigen::Map<const Vec3T> vel_i(dv_i);
        const Eigen::Map<const Vec3T> pos_i(dp_i);
        const Eigen::Map<const Vec3T> bg_i(dbg_i);
        const Eigen::Map<const Vec3T> ba_i(dba_i);
        const Eigen::Map<const Vec3T> phi_j(dtheta_j);
        const Eigen::Map<const Vec3T> vel_j(dv_j);
        const Eigen::Map<const Vec3T> pos_j(dp_j);
        const Eigen::Map<const Vec3T> bg_j(dbg_j);
        const Eigen::Map<const Vec3T> ba_j(dba_j);

        Vec15T xi = Vec15T::Zero();
        Vec15T xj = Vec15T::Zero();
        xi.template segment<3>(0) = pos_i;
        xi.template segment<3>(3) = vel_i;
        xi.template segment<3>(6) = phi_i;
        xi.template segment<3>(9) = bg_i;
        xi.template segment<3>(12) = ba_i;
        xj.template segment<3>(0) = pos_j;
        xj.template segment<3>(3) = vel_j;
        xj.template segment<3>(6) = phi_j;
        xj.template segment<3>(9) = bg_j;
        xj.template segment<3>(12) = ba_j;

        const Mat15T phi = phi_.cast<T>();
        const Mat15T sqrt_info = sqrt_info_.cast<T>();
        const Vec15T err = xj - phi * xi;

        Eigen::Map<Vec15T> res(residuals);
        res = sqrt_info * err;
        return true;
    }

    static ceres::CostFunction* Create(
        double dt,
        const Eigen::Vector3d& nominal_blh,
        const Eigen::Vector3d& nominal_vel_ned,
        const Eigen::Quaterniond& nominal_q_nb,
        const Eigen::Vector3d& nominal_specific_force_body,
        double sigma_gyro_rps,
        double sigma_accel_mps2,
        double sigma_bg_std,
        double sigma_ba_std,
        double bias_tau_s) {
        return new ceres::AutoDiffCostFunction<
            ErrorStateProcessFactor, 15,
            3, 3, 3, 3, 3,
            3, 3, 3, 3, 3>(
            new ErrorStateProcessFactor(
                dt,
                nominal_blh,
                nominal_vel_ned,
                nominal_q_nb,
                nominal_specific_force_body,
                sigma_gyro_rps,
                sigma_accel_mps2,
                sigma_bg_std,
                sigma_ba_std,
                bias_tau_s));
    }

private:
    static Eigen::Vector2d MeridianPrimeVerticalRadius(double lat_rad) {
        const double sin_lat = std::sin(lat_rad);
        const double den = 1.0 - kWgs84E1 * sin_lat * sin_lat;
        const double sqrt_den = std::sqrt(den);
        return {
            kWgs84Ra * (1.0 - kWgs84E1) / (sqrt_den * den),
            kWgs84Ra / sqrt_den,
        };
    }

    static Matrix3d SkewSymmetric(const Eigen::Vector3d& vector) {
        Matrix3d mat;
        mat << 0.0, -vector.z(), vector.y(),
               vector.z(), 0.0, -vector.x(),
              -vector.y(), vector.x(), 0.0;
        return mat;
    }

    static Matrix15d BuildF(
        const Eigen::Vector3d& nominal_blh,
        const Eigen::Vector3d& nominal_vel_ned,
        const Eigen::Quaterniond& nominal_q_nb,
        const Eigen::Vector3d& nominal_specific_force_body,
        double bias_tau_s) {
        Matrix15d F = Matrix15d::Zero();

        const Eigen::Vector2d rmrn = MeridianPrimeVerticalRadius(nominal_blh.x());
        const double gravity = Earth::Gravity(nominal_blh);
        const Eigen::Vector3d wie_n = Earth::Iewn(nominal_blh.x());
        const Eigen::Vector3d wen_n = Earth::Wnen(nominal_blh, nominal_vel_ned);
        const Matrix3d cbn = nominal_q_nb.toRotationMatrix();
        const Eigen::Vector3d f_n = cbn * nominal_specific_force_body;

        const double rmh = rmrn.x() + nominal_blh.z();
        const double rnh = rmrn.y() + nominal_blh.z();
        const double lat = nominal_blh.x();
        const double vn = nominal_vel_ned.x();
        const double ve = nominal_vel_ned.y();
        const double vd = nominal_vel_ned.z();

        Matrix3d temp = Matrix3d::Zero();
        temp(0, 0) = -vd / rmh;
        temp(0, 2) = vn / rmh;
        temp(1, 0) = ve * std::tan(lat) / rnh;
        temp(1, 1) = -(vd + vn * std::tan(lat)) / rnh;
        temp(1, 2) = ve / rnh;
        F.block<3, 3>(0, 0) = temp;
        F.block<3, 3>(0, 3) = Matrix3d::Identity();

        temp.setZero();
        temp(0, 0) = -2.0 * ve * kWgs84Wie * std::cos(lat) / rmh -
                     ve * ve / (rmh * rnh * std::cos(lat) * std::cos(lat));
        temp(0, 2) = vn * vd / (rmh * rmh) - ve * ve * std::tan(lat) / (rnh * rnh);
        temp(1, 0) = 2.0 * kWgs84Wie * (vn * std::cos(lat) - vd * std::sin(lat)) / rmh +
                     vn * ve / (rmh * rnh * std::cos(lat) * std::cos(lat));
        temp(1, 2) = (ve * vd + vn * ve * std::tan(lat)) / (rnh * rnh);
        temp(2, 0) = 2.0 * kWgs84Wie * ve * std::sin(lat) / rmh;
        temp(2, 2) = -ve * ve / (rnh * rnh) - vn * vn / (rmh * rmh) +
                     2.0 * gravity / (std::sqrt(rmrn.x() * rmrn.y()) + nominal_blh.z());
        F.block<3, 3>(3, 0) = temp;

        temp.setZero();
        temp(0, 0) = vd / rmh;
        temp(0, 1) = -2.0 * (kWgs84Wie * std::sin(lat) + ve * std::tan(lat) / rnh);
        temp(0, 2) = vn / rmh;
        temp(1, 0) = 2.0 * kWgs84Wie * std::sin(lat) + ve * std::tan(lat) / rnh;
        temp(1, 1) = (vd + vn * std::tan(lat)) / rnh;
        temp(1, 2) = 2.0 * kWgs84Wie * std::cos(lat) + ve / rnh;
        temp(2, 0) = -2.0 * vn / rmh;
        temp(2, 1) = -2.0 * (kWgs84Wie * std::cos(lat) + ve / rnh);
        F.block<3, 3>(3, 3) = temp;
        F.block<3, 3>(3, 6) = SkewSymmetric(f_n);
        F.block<3, 3>(3, 12) = cbn;

        temp.setZero();
        temp(0, 0) = -kWgs84Wie * std::sin(lat) / rmh;
        temp(0, 2) = ve / (rnh * rnh);
        temp(1, 2) = -vn / (rmh * rmh);
        temp(2, 0) = -kWgs84Wie * std::cos(lat) / rmh -
                     ve / (rmh * rnh * std::cos(lat) * std::cos(lat));
        temp(2, 2) = -ve * std::tan(lat) / (rnh * rnh);
        F.block<3, 3>(6, 0) = temp;

        temp.setZero();
        temp(0, 1) = 1.0 / rnh;
        temp(1, 0) = -1.0 / rmh;
        temp(2, 1) = -std::tan(lat) / rnh;
        F.block<3, 3>(6, 3) = temp;
        F.block<3, 3>(6, 6) = -SkewSymmetric(wie_n + wen_n);
        F.block<3, 3>(6, 9) = -cbn;

        const double tau = std::max(1.0, bias_tau_s);
        F.block<3, 3>(9, 9) = -Matrix3d::Identity() / tau;
        F.block<3, 3>(12, 12) = -Matrix3d::Identity() / tau;
        return F;
    }

    static Matrix15x12d BuildG(const Eigen::Quaterniond& nominal_q_nb) {
        Matrix15x12d G = Matrix15x12d::Zero();
        const Matrix3d cbn = nominal_q_nb.toRotationMatrix();
        G.block<3, 3>(3, 0) = cbn;
        G.block<3, 3>(6, 3) = cbn;
        G.block<3, 3>(9, 6) = Matrix3d::Identity();
        G.block<3, 3>(12, 9) = Matrix3d::Identity();
        return G;
    }

    static Eigen::Matrix<double, 12, 12> BuildQc(
        double sigma_gyro_rps,
        double sigma_accel_mps2,
        double sigma_bg_std,
        double sigma_ba_std,
        double bias_tau_s) {
        Eigen::Matrix<double, 12, 12> Qc = Eigen::Matrix<double, 12, 12>::Zero();
        Qc.block<3, 3>(0, 0) =
            Eigen::Vector3d::Constant(sigma_accel_mps2 * sigma_accel_mps2).asDiagonal();
        Qc.block<3, 3>(3, 3) =
            Eigen::Vector3d::Constant(sigma_gyro_rps * sigma_gyro_rps).asDiagonal();
        const double tau = std::max(1.0, bias_tau_s);
        Qc.block<3, 3>(6, 6) =
            Eigen::Vector3d::Constant(2.0 * sigma_bg_std * sigma_bg_std / tau).asDiagonal();
        Qc.block<3, 3>(9, 9) =
            Eigen::Vector3d::Constant(2.0 * sigma_ba_std * sigma_ba_std / tau).asDiagonal();
        return Qc;
    }

    static Matrix15d BuildPhi(
        double dt,
        const Eigen::Vector3d& nominal_blh,
        const Eigen::Vector3d& nominal_vel_ned,
        const Eigen::Quaterniond& nominal_q_nb,
        const Eigen::Vector3d& nominal_specific_force_body,
        double sigma_gyro_rps,
        double sigma_accel_mps2,
        double sigma_bg_std,
        double sigma_ba_std,
        double bias_tau_s) {
        (void)sigma_gyro_rps;
        (void)sigma_accel_mps2;
        (void)sigma_bg_std;
        (void)sigma_ba_std;
        const Matrix15d F = BuildF(
            nominal_blh, nominal_vel_ned, nominal_q_nb, nominal_specific_force_body, bias_tau_s);
        return Matrix15d::Identity() + F * dt;
    }

    static Matrix15d BuildQd(
        double dt,
        const Eigen::Vector3d& nominal_blh,
        const Eigen::Vector3d& nominal_vel_ned,
        const Eigen::Quaterniond& nominal_q_nb,
        const Eigen::Vector3d& nominal_specific_force_body,
        double sigma_gyro_rps,
        double sigma_accel_mps2,
        double sigma_bg_std,
        double sigma_ba_std,
        double bias_tau_s) {
        const Matrix15d phi = BuildPhi(
            dt,
            nominal_blh,
            nominal_vel_ned,
            nominal_q_nb,
            nominal_specific_force_body,
            sigma_gyro_rps,
            sigma_accel_mps2,
            sigma_bg_std,
            sigma_ba_std,
            bias_tau_s);
        const Matrix15x12d G = BuildG(nominal_q_nb);
        const Eigen::Matrix<double, 12, 12> Qc = BuildQc(
            sigma_gyro_rps, sigma_accel_mps2, sigma_bg_std, sigma_ba_std, bias_tau_s);

        Matrix15d Qd = G * Qc * G.transpose() * dt;
        Qd = (phi * Qd * phi.transpose() + Qd) * 0.5;
        return (Qd + Qd.transpose()) * 0.5;
    }

    static Matrix15d BuildSqrtInfo(
        double dt,
        const Eigen::Vector3d& nominal_blh,
        const Eigen::Vector3d& nominal_vel_ned,
        const Eigen::Quaterniond& nominal_q_nb,
        const Eigen::Vector3d& nominal_specific_force_body,
        double sigma_gyro_rps,
        double sigma_accel_mps2,
        double sigma_bg_std,
        double sigma_ba_std,
        double bias_tau_s) {
        Matrix15d Qd = BuildQd(
            dt,
            nominal_blh,
            nominal_vel_ned,
            nominal_q_nb,
            nominal_specific_force_body,
            sigma_gyro_rps,
            sigma_accel_mps2,
            sigma_bg_std,
            sigma_ba_std,
            bias_tau_s);
        Qd.diagonal().array() += 1.0e-12;
        const Matrix15d info = Qd.inverse();
        Eigen::LLT<Matrix15d> llt(info);
        Matrix15d sqrt_info = Matrix15d::Identity();
        if (llt.info() == Eigen::Success) {
            sqrt_info = llt.matrixL();
        } else {
            sqrt_info.diagonal() =
                Qd.diagonal().cwiseMax(1.0e-12).cwiseSqrt().cwiseInverse();
        }
        return sqrt_info;
    }

    Matrix15d phi_;
    Matrix15d sqrt_info_;
};

}  // namespace ct_fgo_sim::factors
