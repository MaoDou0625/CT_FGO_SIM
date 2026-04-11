#pragma once

#include "ct_fgo_sim/navigation/interval_propagation.h"

#include <ceres/ceres.h>

namespace ct_fgo_sim::factors {

struct ErrorStateBridgeFactor {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using Matrix21d = ErrorStateMatrix;

    ErrorStateBridgeFactor(double u_i, double u_j, const Matrix21d& phi, const Matrix21d& sqrt_info)
        : u_i_(u_i), u_j_(u_j), phi_(phi), sqrt_info_(sqrt_info) {}

    template <typename T>
    bool operator()(
        const T* const dtheta_i0, const T* const dv_i0, const T* const dp_i0,
        const T* const dbg_i0, const T* const dba_i0, const T* const dsg_i0, const T* const dsa_i0,
        const T* const dtheta_i1, const T* const dv_i1, const T* const dp_i1,
        const T* const dbg_i1, const T* const dba_i1, const T* const dsg_i1, const T* const dsa_i1,
        const T* const dtheta_j0, const T* const dv_j0, const T* const dp_j0,
        const T* const dbg_j0, const T* const dba_j0, const T* const dsg_j0, const T* const dsa_j0,
        const T* const dtheta_j1, const T* const dv_j1, const T* const dp_j1,
        const T* const dbg_j1, const T* const dba_j1, const T* const dsg_j1, const T* const dsa_j1,
        T* residuals) const {
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using Vec21T = Eigen::Matrix<T, 21, 1>;
        using Mat21T = Eigen::Matrix<T, 21, 21>;

        auto interpolate = [](const T* const a, const T* const b, const T u) {
            const Eigen::Map<const Vec3T> va(a);
            const Eigen::Map<const Vec3T> vb(b);
            return (T(1.0) - u) * va + u * vb;
        };
        auto pack = [&](double u,
                        const T* const dtheta0, const T* const dv0, const T* const dp0,
                        const T* const dbg0, const T* const dba0, const T* const dsg0, const T* const dsa0,
                        const T* const dtheta1, const T* const dv1, const T* const dp1,
                        const T* const dbg1, const T* const dba1, const T* const dsg1, const T* const dsa1) {
            Vec21T x = Vec21T::Zero();
            const T ut = T(u);
            x.template segment<3>(0) = interpolate(dp0, dp1, ut);
            x.template segment<3>(3) = interpolate(dv0, dv1, ut);
            x.template segment<3>(6) = interpolate(dtheta0, dtheta1, ut);
            x.template segment<3>(9) = interpolate(dbg0, dbg1, ut);
            x.template segment<3>(12) = interpolate(dba0, dba1, ut);
            x.template segment<3>(15) = interpolate(dsg0, dsg1, ut);
            x.template segment<3>(18) = interpolate(dsa0, dsa1, ut);
            return x;
        };

        const Vec21T xi = pack(
            u_i_,
            dtheta_i0, dv_i0, dp_i0, dbg_i0, dba_i0, dsg_i0, dsa_i0,
            dtheta_i1, dv_i1, dp_i1, dbg_i1, dba_i1, dsg_i1, dsa_i1);
        const Vec21T xj = pack(
            u_j_,
            dtheta_j0, dv_j0, dp_j0, dbg_j0, dba_j0, dsg_j0, dsa_j0,
            dtheta_j1, dv_j1, dp_j1, dbg_j1, dba_j1, dsg_j1, dsa_j1);

        const Mat21T phi = phi_.cast<T>();
        const Mat21T sqrt_info = sqrt_info_.cast<T>();
        Eigen::Map<Vec21T> res(residuals);
        res = sqrt_info * (xj - phi * xi);
        return true;
    }

    static ceres::CostFunction* Create(double u_i, double u_j, const Matrix21d& phi, const Matrix21d& sqrt_info) {
        return new ceres::AutoDiffCostFunction<
            ErrorStateBridgeFactor, 21,
            3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3, 3>(
            new ErrorStateBridgeFactor(u_i, u_j, phi, sqrt_info));
    }

private:
    double u_i_ = 0.0;
    double u_j_ = 0.0;
    Matrix21d phi_;
    Matrix21d sqrt_info_;
};

}  // namespace ct_fgo_sim::factors
