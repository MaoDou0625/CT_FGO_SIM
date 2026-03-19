#include "ct_fgo_sim/navigation/mechanization.h"

#include "ct_fgo_sim/navigation/earth.h"

#include <algorithm>
#include <cmath>

namespace ct_fgo_sim {

namespace {

Eigen::Matrix3d BuildTriadFrame(const Vector3d& primary, const Vector3d& secondary) {
    const Vector3d t1 = primary.normalized();
    Vector3d t2 = t1.cross(secondary);
    if (t2.norm() < 1.0e-12) {
        t2 = t1.unitOrthogonal();
    } else {
        t2.normalize();
    }
    const Vector3d t3 = t1.cross(t2);
    Eigen::Matrix3d frame;
    frame.col(0) = t1;
    frame.col(1) = t2;
    frame.col(2) = t3;
    return frame;
}

Eigen::Matrix3d ExpRot(const Vector3d& rotvec) {
    const double angle = rotvec.norm();
    if (angle < 1.0e-12) {
        return Eigen::Matrix3d::Identity();
    }
    return Eigen::AngleAxisd(angle, rotvec / angle).toRotationMatrix();
}

Vector3d InterpolateBias(
    double time,
    const std::vector<double>& bias_times,
    const std::vector<Vector3d>& biases,
    const Vector3d& fallback) {
    if (bias_times.empty() || biases.empty() || bias_times.size() != biases.size()) {
        return fallback;
    }
    if (time <= bias_times.front()) {
        return biases.front();
    }
    if (time >= bias_times.back()) {
        return biases.back();
    }

    const auto upper = std::upper_bound(bias_times.begin(), bias_times.end(), time);
    const size_t j = static_cast<size_t>(std::distance(bias_times.begin(), upper));
    const size_t i = j - 1;
    const double dt = bias_times[j] - bias_times[i];
    if (dt <= 1.0e-9) {
        return biases[i];
    }
    const double u = (time - bias_times[i]) / dt;
    return biases[i] * (1.0 - u) + biases[j] * u;
}

Quaterniond InterpolateQuaternion(
    double time,
    const NominalNavStates& states,
    size_t i,
    size_t j) {
    const double dt = states[j].time - states[i].time;
    if (dt <= 1.0e-9) {
        return states[i].q_nb;
    }
    const double u = std::clamp((time - states[i].time) / dt, 0.0, 1.0);
    return states[i].q_nb.slerp(u, states[j].q_nb).normalized();
}

}  // namespace

StaticAlignmentResult EstimateInitialAlignment(
    const std::vector<ImuMeasurement>& imu,
    const Vector3d& origin_blh,
    double align_time_s) {
    StaticAlignmentResult result;
    if (imu.empty()) {
        return result;
    }

    result.window_start_time = imu.front().time;
    result.window_end_time = imu.front().time + align_time_s;
    result.reference_time = result.window_end_time;
    Vector3d gyro_mean = Vector3d::Zero();
    Vector3d accel_mean = Vector3d::Zero();
    int count = 0;
    for (const auto& meas : imu) {
        if (meas.time > result.window_end_time) {
            break;
        }
        gyro_mean += meas.dtheta;
        accel_mean += meas.dvel;
        result.reference_time = meas.time;
        ++count;
    }

    if (count < 10 || accel_mean.norm() < 1.0e-6 || gyro_mean.norm() < 1.0e-9) {
        return result;
    }

    gyro_mean /= static_cast<double>(count);
    accel_mean /= static_cast<double>(count);

    const Vector3d up_b = accel_mean.normalized();
    const Vector3d up_n = Vector3d::UnitZ();
    const Vector3d wie_n = Earth::Iewn(origin_blh.x());
    const Eigen::Matrix3d triad_b = BuildTriadFrame(up_b, gyro_mean.normalized());
    const Eigen::Matrix3d triad_n = BuildTriadFrame(up_n, wie_n.normalized());
    result.q_nb = Quaterniond(triad_n * triad_b.transpose()).normalized();

    const Eigen::Matrix3d c_nb = result.q_nb.toRotationMatrix();
    const Vector3d expected_gyro_b = c_nb.transpose() * wie_n;
    const Vector3d expected_accel_b = c_nb.transpose() * Vector3d(0.0, 0.0, Earth::Gravity(origin_blh));
    result.bg0 = gyro_mean - expected_gyro_b;
    result.ba0 = accel_mean - expected_accel_b;
    return result;
}

NominalNavStates PropagateNominalTrajectory(
    const std::vector<ImuMeasurement>& imu,
    const Vector3d& initial_blh,
    const StaticAlignmentResult& alignment,
    const std::vector<double>& bias_times,
    const std::vector<Vector3d>& gyro_biases,
    const std::vector<Vector3d>& accel_biases) {
    NominalNavStates nav;
    if (imu.empty()) {
        return nav;
    }

    nav.reserve(imu.size());
    Vector3d blh = initial_blh;
    Vector3d vel_enu = Vector3d::Zero();
    Quaterniond q_nb = alignment.q_nb;
    size_t anchor_index = 0;
    while (anchor_index + 1 < imu.size() && imu[anchor_index].time < alignment.reference_time) {
        ++anchor_index;
    }

    for (size_t i = 0; i <= anchor_index; ++i) {
        const Vector3d bg = InterpolateBias(imu[i].time, bias_times, gyro_biases, alignment.bg0);
        const Vector3d ba = InterpolateBias(imu[i].time, bias_times, accel_biases, alignment.ba0);
        nav.push_back({imu[i].time, blh, vel_enu, q_nb, bg, ba});
    }

    for (size_t i = anchor_index + 1; i < imu.size(); ++i) {
        const auto& meas = imu[i];
        if (meas.dt <= 0.0) {
            continue;
        }

        const Vector3d bg = InterpolateBias(meas.time, bias_times, gyro_biases, alignment.bg0);
        const Vector3d ba = InterpolateBias(meas.time, bias_times, accel_biases, alignment.ba0);
        const Vector3d omega_corr = meas.dtheta - bg;
        const Vector3d f_corr = meas.dvel - ba;

        const Vector3d wie_n = Earth::Iewn(blh.x());
        const Vector3d wen_n = Earth::Wnen(blh, vel_enu);
        const Eigen::Matrix3d c_nb_old = q_nb.toRotationMatrix();
        const Eigen::Matrix3d c_nn = ExpRot(-(wie_n + wen_n) * meas.dt);
        const Eigen::Matrix3d c_bb = ExpRot(omega_corr * meas.dt);
        const Eigen::Matrix3d c_nb_new = c_nn * c_nb_old * c_bb;
        q_nb = Quaterniond(c_nb_new).normalized();

        const Vector3d gravity_n(0.0, 0.0, -Earth::Gravity(blh));
        const Vector3d f_n = q_nb.toRotationMatrix() * f_corr;
        const Vector3d coriolis = (2.0 * wie_n + wen_n).cross(vel_enu);
        const Vector3d acc_n = f_n + gravity_n - coriolis;
        vel_enu += acc_n * meas.dt;

        const Vector3d pos_delta_enu = vel_enu * meas.dt;
        blh = Earth::LocalToGlobal(initial_blh, Earth::GlobalToLocal(initial_blh, blh) + pos_delta_enu);

        nav.push_back({meas.time, blh, vel_enu, q_nb, bg, ba});
    }

    return nav;
}

std::optional<NominalNavState> EvaluateNominalState(
    const NominalNavStates& states,
    double time) {
    if (states.empty()) {
        return std::nullopt;
    }
    if (states.size() == 1 || time <= states.front().time) {
        NominalNavState out = states.front();
        out.time = time;
        return out;
    }
    if (time >= states.back().time) {
        NominalNavState out = states.back();
        out.time = time;
        return out;
    }

    const auto upper = std::lower_bound(
        states.begin(),
        states.end(),
        time,
        [](const NominalNavState& state, double t) { return state.time < t; });
    if (upper == states.begin()) {
        NominalNavState out = states.front();
        out.time = time;
        return out;
    }

    const size_t j = static_cast<size_t>(std::distance(states.begin(), upper));
    const size_t i = j - 1;
    const double dt = states[j].time - states[i].time;
    if (dt <= 1.0e-9) {
        NominalNavState out = states[i];
        out.time = time;
        return out;
    }

    const double u = std::clamp((time - states[i].time) / dt, 0.0, 1.0);
    NominalNavState out;
    out.time = time;
    out.blh = states[i].blh * (1.0 - u) + states[j].blh * u;
    out.vel_enu = states[i].vel_enu * (1.0 - u) + states[j].vel_enu * u;
    out.q_nb = InterpolateQuaternion(time, states, i, j);
    out.bg = states[i].bg * (1.0 - u) + states[j].bg * u;
    out.ba = states[i].ba * (1.0 - u) + states[j].ba * u;
    return out;
}

}  // namespace ct_fgo_sim
