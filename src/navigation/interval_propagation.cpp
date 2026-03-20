#include "ct_fgo_sim/navigation/interval_propagation.h"

#include "ct_fgo_sim/navigation/earth.h"

#include <Eigen/Cholesky>
#include <unsupported/Eigen/MatrixFunctions>

#include <algorithm>
#include <cmath>

namespace ct_fgo_sim {

namespace {

using Matrix3d = Eigen::Matrix3d;
using Matrix12d = Eigen::Matrix<double, 12, 12>;
using Matrix15d = Eigen::Matrix<double, 15, 15>;
using Matrix15x12d = Eigen::Matrix<double, 15, 12>;
using Matrix30d = Eigen::Matrix<double, 30, 30>;

constexpr double kTimeTolerance = 1.0e-6;

Eigen::Vector2d MeridianPrimeVerticalRadius(double lat_rad) {
    const double sin_lat = std::sin(lat_rad);
    const double den = 1.0 - kWgs84E1 * sin_lat * sin_lat;
    const double sqrt_den = std::sqrt(den);
    return {
        kWgs84Ra * (1.0 - kWgs84E1) / (sqrt_den * den),
        kWgs84Ra / sqrt_den,
    };
}

Matrix3d SkewSymmetric(const Vector3d& vector) {
    Matrix3d mat;
    mat << 0.0, -vector.z(), vector.y(),
           vector.z(), 0.0, -vector.x(),
          -vector.y(), vector.x(), 0.0;
    return mat;
}

Matrix15d BuildF(
    const Vector3d& nominal_blh,
    const Vector3d& nominal_vel_ned,
    const Eigen::Quaterniond& nominal_q_nb,
    const Vector3d& nominal_specific_force_body,
    double bias_tau_s) {
    Matrix15d F = Matrix15d::Zero();

    const Eigen::Vector2d rmrn = MeridianPrimeVerticalRadius(nominal_blh.x());
    const double gravity = Earth::Gravity(nominal_blh);
    const Vector3d wie_n = Earth::Iewn(nominal_blh.x());
    const Vector3d wen_n = Earth::Wnen(nominal_blh, nominal_vel_ned);
    const Matrix3d cbn = nominal_q_nb.toRotationMatrix();
    const Vector3d f_n = cbn * nominal_specific_force_body;

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

Matrix15x12d BuildG(const Eigen::Quaterniond& nominal_q_nb) {
    Matrix15x12d G = Matrix15x12d::Zero();
    const Matrix3d cbn = nominal_q_nb.toRotationMatrix();
    G.block<3, 3>(3, 0) = cbn;
    G.block<3, 3>(6, 3) = cbn;
    G.block<3, 3>(9, 6) = Matrix3d::Identity();
    G.block<3, 3>(12, 9) = Matrix3d::Identity();
    return G;
}

Matrix12d BuildQc(
    double sigma_gyro_rps,
    double sigma_accel_mps2,
    double sigma_bg_std,
    double sigma_ba_std,
    double bias_tau_s) {
    Matrix12d Qc = Matrix12d::Zero();
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

void DiscretizeLinearSystem(
    const Matrix15d& F,
    const Matrix15x12d& G,
    const Matrix12d& Qc,
    double dt,
    Matrix15d& phi,
    Matrix15d& q) {
    const Matrix15d gcgt = G * Qc * G.transpose();

    Matrix30d van_loan = Matrix30d::Zero();
    van_loan.block<15, 15>(0, 0) = F;
    van_loan.block<15, 15>(0, 15) = gcgt;
    van_loan.block<15, 15>(15, 15) = -F.transpose();

    const Matrix30d expm = (van_loan * dt).exp();
    phi = expm.block<15, 15>(0, 0);
    q = expm.block<15, 15>(0, 15) * phi.transpose();
    q = (q + q.transpose()) * 0.5;
}

Matrix15d BuildSqrtInfo(const Matrix15d& q) {
    Matrix15d q_stable = q;
    q_stable.diagonal().array() += 1.0e-12;
    const Matrix15d info = q_stable.inverse();
    Eigen::LLT<Matrix15d> llt(info);
    Matrix15d sqrt_info = Matrix15d::Identity();
    if (llt.info() == Eigen::Success) {
        sqrt_info = llt.matrixL();
    } else {
        sqrt_info.diagonal() =
            q_stable.diagonal().cwiseMax(1.0e-12).cwiseSqrt().cwiseInverse();
    }
    return sqrt_info;
}

std::optional<NominalNavState> InterpolateNominalStateMid(
    const NominalNavState& start,
    const NominalNavState& end,
    double time) {
    const double dt = end.time - start.time;
    if (dt <= 1.0e-12) {
        return std::nullopt;
    }
    const double u = std::clamp((time - start.time) / dt, 0.0, 1.0);
    NominalNavState out;
    out.time = time;
    out.blh = start.blh * (1.0 - u) + end.blh * u;
    out.vel_ned = start.vel_ned * (1.0 - u) + end.vel_ned * u;
    out.q_nb = start.q_nb.slerp(u, end.q_nb).normalized();
    out.bg = start.bg * (1.0 - u) + end.bg * u;
    out.ba = start.ba * (1.0 - u) + end.ba * u;
    return out;
}

std::optional<size_t> FindImuIntervalIndex(
    const NominalImuIntervals& intervals,
    double time) {
    if (intervals.empty()) {
        return std::nullopt;
    }
    if (time <= intervals.front().start_time + kTimeTolerance) {
        return size_t{0};
    }
    if (time >= intervals.back().end_time - kTimeTolerance) {
        return intervals.size() - 1;
    }

    const auto upper = std::lower_bound(
        intervals.begin(),
        intervals.end(),
        time,
        [](const NominalImuInterval& interval, double t) { return interval.end_time < t; });
    if (upper == intervals.end()) {
        return std::nullopt;
    }
    return static_cast<size_t>(std::distance(intervals.begin(), upper));
}

}  // namespace

IntervalPropagationCache BuildIntervalPropagationCache(
    const ImuMeasurementArray& imu,
    const NominalNavStates& nominal_states,
    const spline::ControlPointArray& control_points,
    double sigma_gyro_rps,
    double sigma_accel_mps2,
    double sigma_bg_std,
    double sigma_ba_std,
    double bias_tau_s) {
    IntervalPropagationCache cache;
    if (imu.size() < 2 || nominal_states.size() < 2) {
        return cache;
    }

    cache.imu_intervals.reserve(imu.size() - 1);
    for (size_t i = 1; i < imu.size() && i < nominal_states.size(); ++i) {
        const ImuMeasurement& meas = imu[i];
        if (meas.dt <= 1.0e-9) {
            continue;
        }

        const NominalNavState& start_state = nominal_states[i - 1];
        const NominalNavState& end_state = nominal_states[i];
        const double mid_time = 0.5 * (start_state.time + end_state.time);
        const auto mid_state_opt = InterpolateNominalStateMid(start_state, end_state, mid_time);
        if (!mid_state_opt) {
            continue;
        }

        NominalImuInterval interval;
        interval.start_time = start_state.time;
        interval.end_time = end_state.time;
        interval.mid_time = mid_time;
        interval.dt = meas.dt;
        interval.imu_index = i;
        interval.start_state = start_state;
        interval.mid_state = *mid_state_opt;
        interval.end_state = end_state;

        const Vector3d bg_mid = interval.mid_state.bg;
        const Vector3d ba_mid = interval.mid_state.ba;
        interval.omega_ib_b_nom = (meas.dtheta - bg_mid * meas.dt) / meas.dt;
        interval.specific_force_b_nom = (meas.dvel - ba_mid * meas.dt) / meas.dt;

        const Vector3d omega_ie_n = Earth::Iewn(interval.mid_state.blh.x());
        const Vector3d omega_en_n = Earth::Wnen(interval.mid_state.blh, interval.mid_state.vel_ned);
        const Vector3d gravity_n(0.0, 0.0, Earth::Gravity(interval.mid_state.blh));
        interval.accel_n_mid =
            interval.mid_state.q_nb.toRotationMatrix() * interval.specific_force_b_nom +
            gravity_n - (2.0 * omega_ie_n + omega_en_n).cross(interval.mid_state.vel_ned);

        const Matrix15d F = BuildF(
            interval.mid_state.blh,
            interval.mid_state.vel_ned,
            interval.mid_state.q_nb,
            interval.specific_force_b_nom,
            bias_tau_s);
        const Matrix15x12d G = BuildG(interval.mid_state.q_nb);
        const Matrix12d Qc = BuildQc(
            sigma_gyro_rps,
            sigma_accel_mps2,
            sigma_bg_std,
            sigma_ba_std,
            bias_tau_s);
        DiscretizeLinearSystem(F, G, Qc, meas.dt, interval.phi, interval.q);

        cache.imu_intervals.push_back(std::move(interval));
    }

    if (control_points.size() < 2 || cache.imu_intervals.empty()) {
        return cache;
    }

    cache.knot_intervals.resize(control_points.size() - 1);
    size_t imu_cursor = 0;
    for (size_t i = 0; i + 1 < control_points.size(); ++i) {
        KnotIntervalPropagation knot_interval;
        knot_interval.start_time = control_points[i].Timestamp();
        knot_interval.end_time = control_points[i + 1].Timestamp();
        knot_interval.begin_imu_index = imu_cursor;

        Matrix15d phi_total = Matrix15d::Identity();
        Matrix15d q_total = Matrix15d::Zero();
        bool has_step = false;

        while (imu_cursor < cache.imu_intervals.size() &&
               cache.imu_intervals[imu_cursor].end_time <= knot_interval.start_time + kTimeTolerance) {
            ++imu_cursor;
        }

        size_t local_cursor = imu_cursor;
        while (local_cursor < cache.imu_intervals.size()) {
            const NominalImuInterval& imu_interval = cache.imu_intervals[local_cursor];
            if (imu_interval.start_time < knot_interval.start_time - kTimeTolerance) {
                knot_interval.valid = false;
                break;
            }
            if (imu_interval.end_time > knot_interval.end_time + kTimeTolerance) {
                break;
            }

            phi_total = imu_interval.phi * phi_total;
            q_total = imu_interval.phi * q_total * imu_interval.phi.transpose() + imu_interval.q;
            has_step = true;
            ++local_cursor;

            if (std::abs(imu_interval.end_time - knot_interval.end_time) <= kTimeTolerance) {
                knot_interval.valid = true;
                break;
            }
        }

        knot_interval.end_imu_index = local_cursor;
        if (has_step && knot_interval.valid) {
            knot_interval.phi = phi_total;
            knot_interval.q = (q_total + q_total.transpose()) * 0.5;
            knot_interval.sqrt_info = BuildSqrtInfo(knot_interval.q);
            imu_cursor = local_cursor;
        } else {
            knot_interval.valid = false;
        }
        cache.knot_intervals[i] = std::move(knot_interval);
    }

    return cache;
}

std::optional<Vector3d> EvaluateNominalGyroCenterAtTime(
    const IntervalPropagationCache& cache,
    double time) {
    const auto index = FindImuIntervalIndex(cache.imu_intervals, time);
    if (!index) {
        return std::nullopt;
    }
    return cache.imu_intervals[*index].omega_ib_b_nom;
}

std::optional<Vector3d> EvaluateNominalAccelAtTime(
    const IntervalPropagationCache& cache,
    double time) {
    const auto index = FindImuIntervalIndex(cache.imu_intervals, time);
    if (!index) {
        return std::nullopt;
    }
    return cache.imu_intervals[*index].accel_n_mid;
}

}  // namespace ct_fgo_sim
