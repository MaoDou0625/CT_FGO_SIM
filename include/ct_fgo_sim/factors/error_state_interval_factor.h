#pragma once

#include "ct_fgo_sim/types.h"

#include <ceres/ceres.h>
#include <Eigen/Core>

namespace ct_fgo_sim::factors {

struct ErrorStateIntervalFactor {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using Matrix21d = MatrixErrorState;

    ErrorStateIntervalFactor(const Matrix21d& phi, const Matrix21d& sqrt_info)
        : phi_(phi), sqrt_info_(sqrt_info) {}

    template <typename T>
    bool operator()(
        const T* const dtheta_i, const T* const dv_i, const T* const dp_i, const T* const dbg_i, const T* const dba_i,
        const T* const dsg_i, const T* const dsa_i,
        const T* const dtheta_j, const T* const dv_j, const T* const dp_j, const T* const dbg_j, const T* const dba_j,
        const T* const dsg_j, const T* const dsa_j,
        T* residuals) const {
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using Vec21T = Eigen::Matrix<T, 21, 1>;
        using Mat21T = Eigen::Matrix<T, 21, 21>;

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
        const Eigen::Map<const Vec3T> sg_i(dsg_i);
        const Eigen::Map<const Vec3T> sa_i(dsa_i);
        const Eigen::Map<const Vec3T> sg_j(dsg_j);
        const Eigen::Map<const Vec3T> sa_j(dsa_j);

        Vec21T xi = Vec21T::Zero();
        Vec21T xj = Vec21T::Zero();
        xi.template segment<3>(0) = pos_i;
        xi.template segment<3>(3) = vel_i;
        xi.template segment<3>(6) = phi_i;
        xi.template segment<3>(9) = bg_i;
        xi.template segment<3>(12) = ba_i;
        xi.template segment<3>(15) = sg_i;
        xi.template segment<3>(18) = sa_i;
        xj.template segment<3>(0) = pos_j;
        xj.template segment<3>(3) = vel_j;
        xj.template segment<3>(6) = phi_j;
        xj.template segment<3>(9) = bg_j;
        xj.template segment<3>(12) = ba_j;
        xj.template segment<3>(15) = sg_j;
        xj.template segment<3>(18) = sa_j;

        const Mat21T phi = phi_.cast<T>();
        const Mat21T sqrt_info = sqrt_info_.cast<T>();
        const Vec21T err = xj - phi * xi;

        Eigen::Map<Vec21T> res(residuals);
        res = sqrt_info * err;
        return true;
    }

    static ceres::CostFunction* Create(const Matrix21d& phi, const Matrix21d& sqrt_info) {
        return new ceres::AutoDiffCostFunction<
            ErrorStateIntervalFactor, 21,
            3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3,
            3, 3>(
            new ErrorStateIntervalFactor(phi, sqrt_info));
    }

private:
    Matrix21d phi_;
    Matrix21d sqrt_info_;
};

}  // namespace ct_fgo_sim::factors
