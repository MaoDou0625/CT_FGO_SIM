#pragma once

#include "ct_fgo_sim/spline/control_point.h"

#include <Eigen/Core>
#include <sophus/se3.hpp>

namespace ct_fgo_sim::spline {

class BSplineEvaluator {
public:
    template <typename T>
    struct Result {
        Sophus::SE3<T> pose;
        Eigen::Matrix<T, 3, 1> v_world = Eigen::Matrix<T, 3, 1>::Zero();
        Eigen::Matrix<T, 3, 1> v_body = Eigen::Matrix<T, 3, 1>::Zero();
        Eigen::Matrix<T, 3, 1> w_body = Eigen::Matrix<T, 3, 1>::Zero();
        Eigen::Matrix<T, 3, 1> a_world = Eigen::Matrix<T, 3, 1>::Zero();
        Eigen::Matrix<T, 3, 1> alpha_body = Eigen::Matrix<T, 3, 1>::Zero();
    };

    static Eigen::Matrix4d BaseMatrix() {
        Eigen::Matrix4d m;
        m << 1, 4, 1, 0,
            -3, 0, 3, 0,
             3, -6, 3, 0,
            -1, 3, -3, 1;
        return m / 6.0;
    }

    template <typename T>
    static Result<T> Evaluate(
        T u,
        T dt,
        const Sophus::SE3<T>& t0,
        const Sophus::SE3<T>& t1,
        const Sophus::SE3<T>& t2,
        const Sophus::SE3<T>& t3) {
        Result<T> out;

        Eigen::Matrix<T, 4, 1> u_vec;
        u_vec << T(1), u, u * u, u * u * u;
        const Eigen::Matrix<T, 4, 4> m = BaseMatrix().cast<T>();
        const Eigen::Matrix<T, 4, 1> b = m.transpose() * u_vec;

        const T b1 = b(1) + b(2) + b(3);
        const T b2 = b(2) + b(3);
        const T b3 = b(3);

        const Sophus::Vector6<T> omega1 = (t0.inverse() * t1).log();
        const Sophus::Vector6<T> omega2 = (t1.inverse() * t2).log();
        const Sophus::Vector6<T> omega3 = (t2.inverse() * t3).log();

        const Sophus::SE3<T> e1 = Sophus::SE3<T>::exp(omega1 * b1);
        const Sophus::SE3<T> e2 = Sophus::SE3<T>::exp(omega2 * b2);
        const Sophus::SE3<T> e3 = Sophus::SE3<T>::exp(omega3 * b3);
        out.pose = t0 * e1 * e2 * e3;

        const T eps = T(1e-3);
        if (u - T(2.0) * eps >= T(0.0) && u + T(2.0) * eps <= T(1.0)) {
            const Kinematics<T> kin0 = EvaluateFirstOrder(u, dt, t0, t1, t2, t3, eps);
            const Kinematics<T> kin_plus = EvaluateFirstOrder(u + eps, dt, t0, t1, t2, t3, eps);
            const Kinematics<T> kin_minus = EvaluateFirstOrder(u - eps, dt, t0, t1, t2, t3, eps);

            out.v_body = kin0.v_body;
            out.w_body = kin0.w_body;
            out.v_world = kin0.v_world;
            out.a_world = (kin_plus.v_world - kin_minus.v_world) / (T(2.0) * eps * dt);
            out.alpha_body = (kin_plus.w_body - kin_minus.w_body) / (T(2.0) * eps * dt);
        } else if (u - eps >= T(0.0) && u + eps <= T(1.0)) {
            const Kinematics<T> kin0 = EvaluateFirstOrder(u, dt, t0, t1, t2, t3, eps);
            out.v_body = kin0.v_body;
            out.w_body = kin0.w_body;
            out.v_world = kin0.v_world;
        }

        return out;
    }

private:
    template <typename T>
    struct Kinematics {
        Eigen::Matrix<T, 3, 1> v_body = Eigen::Matrix<T, 3, 1>::Zero();
        Eigen::Matrix<T, 3, 1> w_body = Eigen::Matrix<T, 3, 1>::Zero();
        Eigen::Matrix<T, 3, 1> v_world = Eigen::Matrix<T, 3, 1>::Zero();
    };

    template <typename T>
    static Sophus::SE3<T> EvaluatePose(
        T u,
        const Sophus::SE3<T>& t0,
        const Sophus::SE3<T>& t1,
        const Sophus::SE3<T>& t2,
        const Sophus::SE3<T>& t3) {
        Eigen::Matrix<T, 4, 1> u_vec;
        u_vec << T(1), u, u * u, u * u * u;
        const Eigen::Matrix<T, 4, 4> m = BaseMatrix().cast<T>();
        const Eigen::Matrix<T, 4, 1> b = m.transpose() * u_vec;
        const T b1 = b(1) + b(2) + b(3);
        const T b2 = b(2) + b(3);
        const T b3 = b(3);
        return t0 *
               Sophus::SE3<T>::exp((t0.inverse() * t1).log() * b1) *
               Sophus::SE3<T>::exp((t1.inverse() * t2).log() * b2) *
               Sophus::SE3<T>::exp((t2.inverse() * t3).log() * b3);
    }

    template <typename T>
    static Kinematics<T> EvaluateFirstOrder(
        T u,
        T dt,
        const Sophus::SE3<T>& t0,
        const Sophus::SE3<T>& t1,
        const Sophus::SE3<T>& t2,
        const Sophus::SE3<T>& t3,
        T eps) {
        Kinematics<T> out;
        const Sophus::SE3<T> pose = EvaluatePose(u, t0, t1, t2, t3);
        const Sophus::SE3<T> pose_plus = EvaluatePose(u + eps, t0, t1, t2, t3);
        const Sophus::SE3<T> pose_minus = EvaluatePose(u - eps, t0, t1, t2, t3);
        const Sophus::Vector6<T> delta = (pose_minus.inverse() * pose_plus).log() / (T(2.0) * eps * dt);
        out.v_body = delta.template head<3>();
        out.w_body = delta.template tail<3>();
        out.v_world = pose.so3() * out.v_body;
        return out;
    }
};

}  // namespace ct_fgo_sim::spline
