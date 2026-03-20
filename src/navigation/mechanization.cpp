#include "ct_fgo_sim/navigation/mechanization.h"

#include "ct_fgo_sim/navigation/earth.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace ct_fgo_sim {

namespace {

struct MechImuSample {
    double dt = 0.0;
    Vector3d dtheta = Vector3d::Zero();
    Vector3d dvel = Vector3d::Zero();
};

struct PvaState {
    Vector3d blh = Vector3d::Zero();
    Vector3d vel_ned = Vector3d::Zero();
    Quaterniond q_nb = Quaterniond::Identity();
};

Eigen::Matrix3d BuildNedTriad(const Vector3d& down_axis, const Vector3d& earth_rate_axis) {
    Vector3d down = down_axis.normalized();
    Vector3d east = down.cross(earth_rate_axis);
    if (east.norm() < 1.0e-12) {
        east = down.unitOrthogonal();
    } else {
        east.normalize();
    }
    Vector3d north = east.cross(down);
    if (north.norm() < 1.0e-12) {
        north = east.unitOrthogonal();
    } else {
        north.normalize();
    }

    Eigen::Matrix3d frame;
    frame.col(0) = north;
    frame.col(1) = east;
    frame.col(2) = down;
    return frame;
}

Eigen::Vector2d MeridianPrimeVerticalRadius(double lat_rad) {
    const double sin_lat = std::sin(lat_rad);
    const double den = 1.0 - kWgs84E1 * sin_lat * sin_lat;
    const double sqrt_den = std::sqrt(den);
    return {
        kWgs84Ra * (1.0 - kWgs84E1) / (sqrt_den * den),
        kWgs84Ra / sqrt_den,
    };
}

Quaterniond RotvecToQuaternion(const Vector3d& rotvec) {
    const double angle = rotvec.norm();
    if (angle < 1.0e-12) {
        return Quaterniond::Identity();
    }
    return Quaterniond(Eigen::AngleAxisd(angle, rotvec / angle)).normalized();
}

Vector3d QuaternionToRotvec(const Quaterniond& quaternion) {
    const Eigen::AngleAxisd axis_angle(quaternion);
    return axis_angle.angle() * axis_angle.axis();
}

Eigen::Matrix3d ExpRot(const Vector3d& rotvec) {
    const double angle = rotvec.norm();
    if (angle < 1.0e-12) {
        return Eigen::Matrix3d::Identity();
    }
    return Eigen::AngleAxisd(angle, rotvec / angle).toRotationMatrix();
}

Matrix3d SkewSymmetric(const Vector3d& vector) {
    Matrix3d mat;
    mat << 0.0, -vector.z(), vector.y(),
           vector.z(), 0.0, -vector.x(),
          -vector.y(), vector.x(), 0.0;
    return mat;
}

Quaterniond Qne(const Vector3d& blh) {
    Quaterniond quat;
    const double cos_lon = std::cos(blh.y() * 0.5);
    const double sin_lon = std::sin(blh.y() * 0.5);
    const double cos_lat = std::cos(-M_PI * 0.25 - blh.x() * 0.5);
    const double sin_lat = std::sin(-M_PI * 0.25 - blh.x() * 0.5);
    quat.w() = cos_lat * cos_lon;
    quat.x() = -sin_lat * sin_lon;
    quat.y() = sin_lat * cos_lon;
    quat.z() = cos_lat * sin_lon;
    return quat.normalized();
}

Vector3d BlhFromQne(const Quaterniond& qne, double height) {
    return {
        -2.0 * std::atan(qne.y() / qne.w()) - M_PI * 0.5,
        2.0 * std::atan2(qne.z(), qne.w()),
        height,
    };
}

Matrix3d DRi(const Vector3d& blh) {
    Matrix3d dri = Matrix3d::Zero();
    const Eigen::Vector2d rmn = MeridianPrimeVerticalRadius(blh.x());
    dri(0, 0) = 1.0 / (rmn.x() + blh.z());
    dri(1, 1) = 1.0 / ((rmn.y() + blh.z()) * std::cos(blh.x()));
    dri(2, 2) = -1.0;
    return dri;
}

Vector3d InterpolateBias(
    double time,
    const std::vector<double>& bias_times,
    const AlignedVec3Array& biases,
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

MechImuSample BiasCompensate(
    const ImuMeasurement& meas,
    const Vector3d& gyro_bias_rps,
    const Vector3d& accel_bias_mps2) {
    MechImuSample corrected;
    corrected.dt = meas.dt;
    corrected.dtheta = meas.dtheta - gyro_bias_rps * meas.dt;
    corrected.dvel = meas.dvel - accel_bias_mps2 * meas.dt;
    return corrected;
}

void KfVelUpdate(
    const PvaState& pvapre,
    PvaState& pvacur,
    const MechImuSample& imupre,
    const MechImuSample& imucur) {
    static int debug_step = 0;
    const Matrix3d I33 = Matrix3d::Identity();
    const Eigen::Vector2d rmrn = MeridianPrimeVerticalRadius(pvapre.blh.x());
    Vector3d wie_n;
    Vector3d wen_n;
    wie_n << kWgs84Wie * std::cos(pvapre.blh.x()), 0.0, -kWgs84Wie * std::sin(pvapre.blh.x());
    wen_n << pvapre.vel_ned.y() / (rmrn.y() + pvapre.blh.z()),
             -pvapre.vel_ned.x() / (rmrn.x() + pvapre.blh.z()),
             -pvapre.vel_ned.y() * std::tan(pvapre.blh.x()) / (rmrn.y() + pvapre.blh.z());
    const double gravity = Earth::Gravity(pvapre.blh);

    const Vector3d temp1 = imucur.dtheta.cross(imucur.dvel) / 2.0;
    const Vector3d temp2 = imupre.dtheta.cross(imucur.dvel) / 12.0;
    const Vector3d temp3 = imupre.dvel.cross(imucur.dtheta) / 12.0;
    const Vector3d d_vfb = imucur.dvel + temp1 + temp2 + temp3;

    const Vector3d rot_half = (wie_n + wen_n) * imucur.dt / 2.0;
    Matrix3d cnn = I33 - SkewSymmetric(rot_half);
    Vector3d d_vfn = cnn * pvapre.q_nb.toRotationMatrix() * d_vfb;

    Vector3d gl(0.0, 0.0, gravity);
    Vector3d d_vgn = (gl - (2.0 * wie_n + wen_n).cross(pvapre.vel_ned)) * imucur.dt;
    const Vector3d midvel = pvapre.vel_ned + (d_vfn + d_vgn) / 2.0;

    Quaterniond qnn = RotvecToQuaternion(rot_half);
    Quaterniond qee = RotvecToQuaternion(Vector3d(0.0, 0.0, -kWgs84Wie * imucur.dt / 2.0));
    Quaterniond qne = Qne(pvapre.blh);
    qne = (qee * qne * qnn).normalized();
    Vector3d midpos;
    midpos.z() = pvapre.blh.z() - midvel.z() * imucur.dt / 2.0;
    midpos = BlhFromQne(qne, midpos.z());

    const Eigen::Vector2d rmrn_mid = MeridianPrimeVerticalRadius(midpos.x());
    wie_n << kWgs84Wie * std::cos(midpos.x()), 0.0, -kWgs84Wie * std::sin(midpos.x());
    wen_n << midvel.y() / (rmrn_mid.y() + midpos.z()),
             -midvel.x() / (rmrn_mid.x() + midpos.z()),
             -midvel.y() * std::tan(midpos.x()) / (rmrn_mid.y() + midpos.z());

    cnn = I33 - SkewSymmetric((wie_n + wen_n) * imucur.dt / 2.0);
    d_vfn = cnn * pvapre.q_nb.toRotationMatrix() * d_vfb;
    gl << 0.0, 0.0, Earth::Gravity(midpos);
    d_vgn = (gl - (2.0 * wie_n + wen_n).cross(midvel)) * imucur.dt;

    pvacur.vel_ned = pvapre.vel_ned + d_vfn + d_vgn;
    if (debug_step < 5) {
        std::fprintf(
            stderr,
            "[mech] step=%d dt=%.9f dtheta=(%.9e %.9e %.9e) dvel=(%.9e %.9e %.9e) "
            "d_vfb=(%.9e %.9e %.9e) d_vfn=(%.9e %.9e %.9e) d_vgn=(%.9e %.9e %.9e) vel=(%.9e %.9e %.9e)\n",
            debug_step,
            imucur.dt,
            imucur.dtheta.x(), imucur.dtheta.y(), imucur.dtheta.z(),
            imucur.dvel.x(), imucur.dvel.y(), imucur.dvel.z(),
            d_vfb.x(), d_vfb.y(), d_vfb.z(),
            d_vfn.x(), d_vfn.y(), d_vfn.z(),
            d_vgn.x(), d_vgn.y(), d_vgn.z(),
            pvacur.vel_ned.x(), pvacur.vel_ned.y(), pvacur.vel_ned.z());
    }
    ++debug_step;
}

void KfPosUpdate(
    const PvaState& pvapre,
    PvaState& pvacur,
    const MechImuSample& imucur) {
    const Vector3d midvel = (pvacur.vel_ned + pvapre.vel_ned) / 2.0;
    const Vector3d midpos = pvapre.blh + DRi(pvapre.blh) * midvel * imucur.dt / 2.0;

    const Eigen::Vector2d rmrn = MeridianPrimeVerticalRadius(midpos.x());
    Vector3d wie_n;
    Vector3d wen_n;
    wie_n << kWgs84Wie * std::cos(midpos.x()), 0.0, -kWgs84Wie * std::sin(midpos.x());
    wen_n << midvel.y() / (rmrn.y() + midpos.z()),
             -midvel.x() / (rmrn.x() + midpos.z()),
             -midvel.y() * std::tan(midpos.x()) / (rmrn.y() + midpos.z());

    const Quaterniond qnn = RotvecToQuaternion((wie_n + wen_n) * imucur.dt);
    const Quaterniond qee = RotvecToQuaternion(Vector3d(0.0, 0.0, -kWgs84Wie * imucur.dt));
    Quaterniond qne = Qne(pvapre.blh);
    qne = (qee * qne * qnn).normalized();

    pvacur.blh.z() = pvapre.blh.z() - midvel.z() * imucur.dt;
    pvacur.blh = BlhFromQne(qne, pvacur.blh.z());
}

void KfAttUpdate(
    const PvaState& pvapre,
    PvaState& pvacur,
    const MechImuSample& imupre,
    const MechImuSample& imucur) {
    const Vector3d midvel = (pvapre.vel_ned + pvacur.vel_ned) / 2.0;
    const Quaterniond qne_pre = Qne(pvapre.blh);
    const Quaterniond qne_cur = Qne(pvacur.blh);
    Quaterniond qee = RotvecToQuaternion(Vector3d(0.0, 0.0, -kWgs84Wie * imucur.dt));
    const Vector3d qne_delta = QuaternionToRotvec((qne_pre.inverse() * qee.inverse() * qne_cur).normalized());
    qee = RotvecToQuaternion(Vector3d(0.0, 0.0, -kWgs84Wie * imucur.dt / 2.0));
    const Quaterniond qne_mid = (qee * qne_pre * RotvecToQuaternion(qne_delta / 2.0)).normalized();

    Vector3d midpos;
    midpos.z() = (pvapre.blh.z() + pvacur.blh.z()) / 2.0;
    midpos = BlhFromQne(qne_mid, midpos.z());

    const Eigen::Vector2d rmrn = MeridianPrimeVerticalRadius(midpos.x());
    Vector3d wie_n;
    Vector3d wen_n;
    wie_n << kWgs84Wie * std::cos(midpos.x()), 0.0, -kWgs84Wie * std::sin(midpos.x());
    wen_n << midvel.y() / (rmrn.y() + midpos.z()),
             -midvel.x() / (rmrn.x() + midpos.z()),
             -midvel.y() * std::tan(midpos.x()) / (rmrn.y() + midpos.z());

    const Quaterniond qnn = RotvecToQuaternion(-(wie_n + wen_n) * imucur.dt);
    const Quaterniond qbb = RotvecToQuaternion(imucur.dtheta + imupre.dtheta.cross(imucur.dtheta) / 12.0);
    pvacur.q_nb = (qnn * pvapre.q_nb * qbb).normalized();
}

}  // namespace

StaticAlignmentResult EstimateInitialAlignment(
    const ImuMeasurementArray& imu,
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
        if (meas.dt <= 1.0e-6) {
            continue;
        }
        gyro_mean += meas.dtheta / meas.dt;
        accel_mean += meas.dvel / meas.dt;
        result.reference_time = meas.time;
        ++count;
    }

    if (count < 10 || accel_mean.norm() < 1.0e-6 || gyro_mean.norm() < 1.0e-9) {
        return result;
    }

    gyro_mean /= static_cast<double>(count);
    accel_mean /= static_cast<double>(count);

    const Vector3d down_b = -accel_mean.normalized();
    const Vector3d down_n = Vector3d::UnitZ();
    const Vector3d wie_n = Earth::Iewn(origin_blh.x());
    const Eigen::Matrix3d triad_b = BuildNedTriad(down_b, gyro_mean);
    const Eigen::Matrix3d triad_n = BuildNedTriad(down_n, wie_n);
    result.q_nb = Quaterniond(triad_n * triad_b.transpose()).normalized();

    const Eigen::Matrix3d c_nb = result.q_nb.toRotationMatrix();
    const Vector3d expected_gyro_b = c_nb.transpose() * wie_n;
    const Vector3d expected_accel_b = c_nb.transpose() * Vector3d(0.0, 0.0, -Earth::Gravity(origin_blh));
    result.bg0 = gyro_mean - expected_gyro_b;
    result.ba0 = accel_mean - expected_accel_b;
    return result;
}

NominalNavStates PropagateNominalTrajectory(
    const ImuMeasurementArray& imu,
    const Vector3d& initial_blh,
    const StaticAlignmentResult& alignment,
    const std::vector<double>& bias_times,
    const AlignedVec3Array& gyro_biases,
    const AlignedVec3Array& accel_biases) {
    NominalNavStates nav;
    if (imu.empty()) {
        return nav;
    }

    nav.reserve(imu.size());
    Vector3d blh = initial_blh;
    PvaState pvacur;
    pvacur.blh = blh;
    pvacur.vel_ned = alignment.vel0_ned;
    pvacur.q_nb = alignment.q_nb;
    size_t anchor_index = 0;
    while (anchor_index + 1 < imu.size() && imu[anchor_index].time < alignment.reference_time) {
        ++anchor_index;
    }

    for (size_t i = 0; i <= anchor_index; ++i) {
        const Vector3d bg = InterpolateBias(imu[i].time, bias_times, gyro_biases, alignment.bg0);
        const Vector3d ba = InterpolateBias(imu[i].time, bias_times, accel_biases, alignment.ba0);
        nav.push_back({imu[i].time, pvacur.blh, pvacur.vel_ned, pvacur.q_nb, bg, ba});
    }

    for (size_t i = anchor_index + 1; i < imu.size(); ++i) {
        const auto& meas = imu[i];
        if (meas.dt <= 0.0) {
            continue;
        }

        const Vector3d bg_pre = InterpolateBias(imu[i - 1].time, bias_times, gyro_biases, alignment.bg0);
        const Vector3d ba_pre = InterpolateBias(imu[i - 1].time, bias_times, accel_biases, alignment.ba0);
        const Vector3d bg_cur = InterpolateBias(meas.time, bias_times, gyro_biases, alignment.bg0);
        const Vector3d ba_cur = InterpolateBias(meas.time, bias_times, accel_biases, alignment.ba0);
        const MechImuSample imupre = BiasCompensate(imu[i - 1], bg_pre, ba_pre);
        const MechImuSample imucur = BiasCompensate(meas, bg_cur, ba_cur);

        PvaState pvapre = pvacur;
        KfVelUpdate(pvapre, pvacur, imupre, imucur);
        KfPosUpdate(pvapre, pvacur, imucur);
        KfAttUpdate(pvapre, pvacur, imupre, imucur);

        nav.push_back({meas.time, pvacur.blh, pvacur.vel_ned, pvacur.q_nb, bg_cur, ba_cur});
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
    out.vel_ned = states[i].vel_ned * (1.0 - u) + states[j].vel_ned * u;
    out.q_nb = InterpolateQuaternion(time, states, i, j);
    out.bg = states[i].bg * (1.0 - u) + states[j].bg * u;
    out.ba = states[i].ba * (1.0 - u) + states[j].ba * u;
    return out;
}

}  // namespace ct_fgo_sim
