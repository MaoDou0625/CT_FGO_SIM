#include "ct_fgo_sim/navigation/interval_propagation.h"

#include "ct_fgo_sim/navigation/earth.h"
#include "ct_fgo_sim/navigation/nav_math.h"

#include <Eigen/Cholesky>
#include <unsupported/Eigen/MatrixFunctions>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>

namespace ct_fgo_sim {

namespace {

using Matrix3d = Eigen::Matrix3d;
using Matrix18d = Eigen::Matrix<double, 18, 18>;
using Matrix21d = MatrixErrorState;
using Matrix21x18d = Eigen::Matrix<double, 21, 18>;
using Matrix42d = Eigen::Matrix<double, 42, 42>;

constexpr double kTimeTolerance = 1.0e-6;

Matrix21d BuildF(
    const Vector3d& nominal_blh,
    const Vector3d& nominal_vel_ned,
    const Eigen::Quaterniond& nominal_q_nb,
    const Vector3d& nominal_omega_ib_b,
    const Vector3d& nominal_specific_force_body,
    double bias_tau_s) {
    Matrix21d F = Matrix21d::Zero();

    const Eigen::Vector2d rmrn = Earth::MeridianPrimeVerticalRadii(nominal_blh.x());
    const double gravity = Earth::Gravity(nominal_blh);
    const Vector3d wie_n = Earth::Iewn(nominal_blh.x());
    const Vector3d wen_n = Earth::Wnen(nominal_blh, nominal_vel_ned);
    assert(std::isfinite(gravity));
    assert(wie_n.allFinite());
    assert(wen_n.allFinite());
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
    F.block<3, 3>(3, 6) = SkewSymmetric3(f_n);
    F.block<3, 3>(3, 12) = cbn;
    F.block<3, 3>(3, 18) = cbn * nominal_specific_force_body.asDiagonal();

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
    F.block<3, 3>(6, 6) = -SkewSymmetric3(wie_n + wen_n);
    F.block<3, 3>(6, 9) = -cbn;
    F.block<3, 3>(6, 15) = -cbn * nominal_omega_ib_b.asDiagonal();

    const double tau = std::max(1.0, bias_tau_s);
    F.block<3, 3>(9, 9) = -Matrix3d::Identity() / tau;
    F.block<3, 3>(12, 12) = -Matrix3d::Identity() / tau;
    F.block<3, 3>(15, 15) = -Matrix3d::Identity() / tau;
    F.block<3, 3>(18, 18) = -Matrix3d::Identity() / tau;
    return F;
}

Matrix21x18d BuildG(const Eigen::Quaterniond& nominal_q_nb) {
    Matrix21x18d G = Matrix21x18d::Zero();
    const Matrix3d cbn = nominal_q_nb.toRotationMatrix();
    G.block<3, 3>(3, 0) = cbn;
    G.block<3, 3>(6, 3) = cbn;
    G.block<3, 3>(9, 6) = Matrix3d::Identity();
    G.block<3, 3>(12, 9) = Matrix3d::Identity();
    G.block<3, 3>(15, 12) = Matrix3d::Identity();
    G.block<3, 3>(18, 15) = Matrix3d::Identity();
    return G;
}

Matrix18d BuildQc(
    double sigma_gyro_rps,
    double sigma_accel_mps2,
    double sigma_bg_std,
    double sigma_ba_std,
    double sigma_sg_std,
    double sigma_sa_std,
    double bias_tau_s) {
    Matrix18d Qc = Matrix18d::Zero();
    Qc.block<3, 3>(0, 0) =
        Eigen::Vector3d::Constant(sigma_accel_mps2 * sigma_accel_mps2).asDiagonal();
    Qc.block<3, 3>(3, 3) =
        Eigen::Vector3d::Constant(sigma_gyro_rps * sigma_gyro_rps).asDiagonal();
    const double tau = std::max(1.0, bias_tau_s);
    Qc.block<3, 3>(6, 6) =
        Eigen::Vector3d::Constant(2.0 * sigma_bg_std * sigma_bg_std / tau).asDiagonal();
    Qc.block<3, 3>(9, 9) =
        Eigen::Vector3d::Constant(2.0 * sigma_ba_std * sigma_ba_std / tau).asDiagonal();
    Qc.block<3, 3>(12, 12) =
        Eigen::Vector3d::Constant(2.0 * sigma_sg_std * sigma_sg_std / tau).asDiagonal();
    Qc.block<3, 3>(15, 15) =
        Eigen::Vector3d::Constant(2.0 * sigma_sa_std * sigma_sa_std / tau).asDiagonal();
    return Qc;
}

void DiscretizeLinearSystem(
    const Matrix21d& F,
    const Matrix21x18d& G,
    const Matrix18d& Qc,
    double dt,
    Matrix21d& phi,
    Matrix21d& q) {
    const Matrix21d gcgt = G * Qc * G.transpose();

    Matrix42d van_loan = Matrix42d::Zero();
    van_loan.block<21, 21>(0, 0) = F;
    van_loan.block<21, 21>(0, 21) = gcgt;
    van_loan.block<21, 21>(21, 21) = -F.transpose();

    const Matrix42d expm = (van_loan * dt).exp();
    phi = expm.block<21, 21>(0, 0);
    q = expm.block<21, 21>(0, 21) * phi.transpose();
    q = (q + q.transpose()) * 0.5;
}

Matrix21d BuildSqrtInfo(const Matrix21d& q) {
    Matrix21d q_stable = q;
    q_stable.diagonal().array() += 1.0e-12;
    const Matrix21d info = q_stable.inverse();
    Eigen::LLT<Matrix21d> llt(info);
    Matrix21d sqrt_info = Matrix21d::Identity();
    if (llt.info() == Eigen::Success) {
        sqrt_info = llt.matrixU();
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
    out.sg = start.sg * (1.0 - u) + end.sg * u;
    out.sa = start.sa * (1.0 - u) + end.sa * u;
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

bool BuildNominalImuIntervalAt(
    size_t imu_index,
    const ImuMeasurementArray& imu,
    const NominalNavStates& nominal_states,
    NominalImuInterval& interval) {
    if (imu_index == 0 || imu_index >= imu.size() || imu_index >= nominal_states.size()) {
        return false;
    }
    const ImuMeasurement& meas = imu[imu_index];
    if (meas.dt <= 1.0e-9) {
        return false;
    }

    const NominalNavState& start_state = nominal_states[imu_index - 1];
    const NominalNavState& end_state = nominal_states[imu_index];
    const double mid_time = 0.5 * (start_state.time + end_state.time);
    const auto mid_state_opt = InterpolateNominalStateMid(start_state, end_state, mid_time);
    if (!mid_state_opt) {
        return false;
    }

    interval = NominalImuInterval{};
    interval.start_time = start_state.time;
    interval.end_time = end_state.time;
    interval.mid_time = mid_time;
    interval.dt = meas.dt;
    interval.imu_index = imu_index;

    const NominalNavState& mid_state = *mid_state_opt;
    interval.omega_ib_b_nom =
        (meas.dtheta - mid_state.bg * meas.dt).cwiseQuotient((Vector3d::Ones() + mid_state.sg).cwiseMax(1.0e-8)) /
        meas.dt;
    const Vector3d specific_force_b_nom =
        (meas.dvel - mid_state.ba * meas.dt).cwiseQuotient((Vector3d::Ones() + mid_state.sa).cwiseMax(1.0e-8)) /
        meas.dt;
    const Vector3d omega_ie_n = Earth::Iewn(mid_state.blh.x());
    const Vector3d omega_en_n = Earth::Wnen(mid_state.blh, mid_state.vel_ned);
    const Vector3d gravity_n(0.0, 0.0, Earth::Gravity(mid_state.blh));
    interval.accel_n_mid =
        mid_state.q_nb.toRotationMatrix() * specific_force_b_nom +
        gravity_n - (2.0 * omega_ie_n + omega_en_n).cross(mid_state.vel_ned);
    return true;
}

bool BuildKnotIntervalFromCursor(
    size_t knot_index,
    size_t& imu_cursor,
    const ImuMeasurementArray& imu,
    const NominalNavStates& nominal_states,
    const spline::ControlPointArray& control_points,
    const Matrix18d& Qc,
    double bias_tau_s,
    const NominalImuIntervals& imu_intervals,
    KnotIntervalPropagation& knot_interval) {
    if (knot_index + 1 >= control_points.size()) {
        return false;
    }
    knot_interval = KnotIntervalPropagation{};
    knot_interval.start_time = control_points[knot_index].Timestamp();
    knot_interval.end_time = control_points[knot_index + 1].Timestamp();
    knot_interval.begin_imu_index = imu_cursor;

    Matrix21d phi_total = Matrix21d::Identity();
    Matrix21d q_total = Matrix21d::Zero();
    bool has_step = false;

    while (imu_cursor < imu_intervals.size() &&
           imu_intervals[imu_cursor].end_time <= knot_interval.start_time + kTimeTolerance) {
        ++imu_cursor;
    }

    size_t local_cursor = imu_cursor;
    while (local_cursor < imu_intervals.size()) {
        const NominalImuInterval& imu_interval = imu_intervals[local_cursor];
        if (imu_interval.start_time < knot_interval.start_time - kTimeTolerance) {
            knot_interval.valid = false;
            break;
        }
        if (imu_interval.end_time > knot_interval.end_time + kTimeTolerance) {
            break;
        }

        const size_t imu_index = imu_interval.imu_index;
        if (imu_index == 0 || imu_index >= imu.size() || imu_index >= nominal_states.size()) {
            break;
        }
        const ImuMeasurement& meas = imu[imu_index];
        const NominalNavState& start_state = nominal_states[imu_index - 1];
        const NominalNavState& end_state = nominal_states[imu_index];
        const auto mid_state_opt = InterpolateNominalStateMid(
            start_state,
            end_state,
            imu_interval.mid_time);
        if (!mid_state_opt) {
            break;
        }
        const NominalNavState& mid_state = *mid_state_opt;
        const Vector3d specific_force_b_nom =
            (meas.dvel - mid_state.ba * meas.dt)
                .cwiseQuotient((Vector3d::Ones() + mid_state.sa).cwiseMax(1.0e-8)) / meas.dt;
        const Matrix21d F = BuildF(
            mid_state.blh,
            mid_state.vel_ned,
            mid_state.q_nb,
            imu_interval.omega_ib_b_nom,
            specific_force_b_nom,
            bias_tau_s);
        const Matrix21x18d G = BuildG(mid_state.q_nb);
        Matrix21d phi_step = Matrix21d::Identity();
        Matrix21d q_step = Matrix21d::Zero();
        DiscretizeLinearSystem(F, G, Qc, meas.dt, phi_step, q_step);

        phi_total = phi_step * phi_total;
        q_total = phi_step * q_total * phi_step.transpose() + q_step;
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
    return true;
}

}  // namespace

void BuildIntervalPropagationCache(
    const ImuMeasurementArray& imu,
    const NominalNavStates& nominal_states,
    const spline::ControlPointArray& control_points,
    double sigma_gyro_rps,
    double sigma_accel_mps2,
    double sigma_bg_std,
    double sigma_ba_std,
    double sigma_sg_std,
    double sigma_sa_std,
    double bias_tau_s,
    IntervalPropagationCache& cache) {
    cache.imu_intervals.clear();
    cache.knot_intervals.clear();
    if (imu.size() < 2 || nominal_states.size() < 2) {
        return;
    }

    const Matrix18d Qc = BuildQc(
        sigma_gyro_rps,
        sigma_accel_mps2,
        sigma_bg_std,
        sigma_ba_std,
        sigma_sg_std,
        sigma_sa_std,
        bias_tau_s);
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

        const NominalNavState& mid_state = *mid_state_opt;
        const Vector3d bg_mid = mid_state.bg;
        const Vector3d ba_mid = mid_state.ba;
        interval.omega_ib_b_nom =
            (meas.dtheta - bg_mid * meas.dt)
                .cwiseQuotient((Vector3d::Ones() + mid_state.sg).cwiseMax(1.0e-8)) / meas.dt;
        const Vector3d specific_force_b_nom =
            (meas.dvel - ba_mid * meas.dt)
                .cwiseQuotient((Vector3d::Ones() + mid_state.sa).cwiseMax(1.0e-8)) / meas.dt;

        const Vector3d omega_ie_n = Earth::Iewn(mid_state.blh.x());
        const Vector3d omega_en_n = Earth::Wnen(mid_state.blh, mid_state.vel_ned);
        const double gravity = Earth::Gravity(mid_state.blh);
        assert(std::isfinite(gravity));
        assert(omega_ie_n.allFinite());
        assert(omega_en_n.allFinite());
        const Vector3d gravity_n(0.0, 0.0, gravity);
        interval.accel_n_mid =
            mid_state.q_nb.toRotationMatrix() * specific_force_b_nom +
            gravity_n - (2.0 * omega_ie_n + omega_en_n).cross(mid_state.vel_ned);

        cache.imu_intervals.push_back(std::move(interval));
    }

    if (control_points.size() < 2 || cache.imu_intervals.empty()) {
        return;
    }

    cache.knot_intervals.resize(control_points.size() - 1);
    size_t imu_cursor = 0;
    for (size_t i = 0; i + 1 < control_points.size(); ++i) {
        KnotIntervalPropagation knot_interval;
        knot_interval.start_time = control_points[i].Timestamp();
        knot_interval.end_time = control_points[i + 1].Timestamp();
        knot_interval.begin_imu_index = imu_cursor;

        Matrix21d phi_total = Matrix21d::Identity();
        Matrix21d q_total = Matrix21d::Zero();
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

            const size_t imu_index = imu_interval.imu_index;
            const ImuMeasurement& meas = imu[imu_index];
            const NominalNavState& start_state = nominal_states[imu_index - 1];
            const NominalNavState& end_state = nominal_states[imu_index];
            const auto mid_state_opt = InterpolateNominalStateMid(
                start_state,
                end_state,
                imu_interval.mid_time);
            if (!mid_state_opt) {
                break;
            }

            const NominalNavState& mid_state = *mid_state_opt;
            const Vector3d omega_ib_b_nom =
                (meas.dtheta - mid_state.bg * meas.dt)
                    .cwiseQuotient((Vector3d::Ones() + mid_state.sg).cwiseMax(1.0e-8)) / meas.dt;
            const Vector3d specific_force_b_nom =
                (meas.dvel - mid_state.ba * meas.dt)
                    .cwiseQuotient((Vector3d::Ones() + mid_state.sa).cwiseMax(1.0e-8)) / meas.dt;
            const Matrix21d F = BuildF(
                mid_state.blh,
                mid_state.vel_ned,
                mid_state.q_nb,
                omega_ib_b_nom,
                specific_force_b_nom,
                bias_tau_s);
            const Matrix21x18d G = BuildG(mid_state.q_nb);
            Matrix21d phi_step = Matrix21d::Identity();
            Matrix21d q_step = Matrix21d::Zero();
            DiscretizeLinearSystem(F, G, Qc, meas.dt, phi_step, q_step);

            phi_total = phi_step * phi_total;
            q_total = phi_step * q_total * phi_step.transpose() + q_step;
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
}

void AppendIntervalPropagationCache(
    const ImuMeasurementArray& imu,
    const NominalNavStates& nominal_states,
    const spline::ControlPointArray& control_points,
    double sigma_gyro_rps,
    double sigma_accel_mps2,
    double sigma_bg_std,
    double sigma_ba_std,
    double sigma_sg_std,
    double sigma_sa_std,
    double bias_tau_s,
    IntervalPropagationCache& cache) {
    if (imu.size() < 2 || nominal_states.size() < 2) {
        cache = IntervalPropagationCache{};
        return;
    }
    if (cache.imu_intervals.empty() && cache.knot_intervals.empty()) {
        BuildIntervalPropagationCache(
            imu,
            nominal_states,
            control_points,
            sigma_gyro_rps,
            sigma_accel_mps2,
            sigma_bg_std,
            sigma_ba_std,
            sigma_sg_std,
            sigma_sa_std,
            bias_tau_s,
            cache);
        return;
    }

    const bool invalid_prefix =
        (!cache.imu_intervals.empty() &&
         cache.imu_intervals.back().imu_index + 1 > nominal_states.size()) ||
        cache.knot_intervals.size() > (control_points.size() >= 2 ? control_points.size() - 1 : 0);
    if (invalid_prefix) {
        BuildIntervalPropagationCache(
            imu,
            nominal_states,
            control_points,
            sigma_gyro_rps,
            sigma_accel_mps2,
            sigma_bg_std,
            sigma_ba_std,
            sigma_sg_std,
            sigma_sa_std,
            bias_tau_s,
            cache);
        return;
    }

    const size_t imu_last = std::min(imu.size(), nominal_states.size()) - 1;
    size_t next_imu_index = 1;
    if (!cache.imu_intervals.empty()) {
        next_imu_index = cache.imu_intervals.back().imu_index + 1;
    }
    cache.imu_intervals.reserve(imu_last);
    for (size_t i = next_imu_index; i <= imu_last; ++i) {
        NominalImuInterval interval;
        if (BuildNominalImuIntervalAt(i, imu, nominal_states, interval)) {
            cache.imu_intervals.push_back(std::move(interval));
        }
    }

    if (control_points.size() < 2 || cache.imu_intervals.empty()) {
        cache.knot_intervals.clear();
        return;
    }
    const Matrix18d Qc = BuildQc(
        sigma_gyro_rps,
        sigma_accel_mps2,
        sigma_bg_std,
        sigma_ba_std,
        sigma_sg_std,
        sigma_sa_std,
        bias_tau_s);

    const size_t required_knot_intervals = control_points.size() - 1;
    size_t start_knot = cache.knot_intervals.size();
    size_t imu_cursor = 0;
    if (start_knot > 0) {
        imu_cursor = cache.knot_intervals[start_knot - 1].end_imu_index;
    }

    cache.knot_intervals.resize(required_knot_intervals);
    for (size_t i = start_knot; i < required_knot_intervals; ++i) {
        KnotIntervalPropagation knot_interval;
        if (!BuildKnotIntervalFromCursor(
                i,
                imu_cursor,
                imu,
                nominal_states,
                control_points,
                Qc,
                bias_tau_s,
                cache.imu_intervals,
                knot_interval)) {
            BuildIntervalPropagationCache(
                imu,
                nominal_states,
                control_points,
                sigma_gyro_rps,
                sigma_accel_mps2,
                sigma_bg_std,
                sigma_ba_std,
                sigma_sg_std,
                sigma_sa_std,
                bias_tau_s,
                cache);
            return;
        }
        cache.knot_intervals[i] = std::move(knot_interval);
    }
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
