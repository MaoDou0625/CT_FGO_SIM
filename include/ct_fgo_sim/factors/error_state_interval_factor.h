#pragma once

#include <ceres/ceres.h>
#include <Eigen/Core>

namespace ct_fgo_sim::factors {

struct ErrorStateIntervalFactor {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using Matrix15d = Eigen::Matrix<double, 15, 15>;

    ErrorStateIntervalFactor(const Matrix15d& phi, const Matrix15d& sqrt_info)
        : phi_(phi), sqrt_info_(sqrt_info) {}

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

    static ceres::CostFunction* Create(const Matrix15d& phi, const Matrix15d& sqrt_info) {
        return new ceres::AutoDiffCostFunction<
            ErrorStateIntervalFactor, 15,
            3, 3, 3, 3, 3,
            3, 3, 3, 3, 3>(
            new ErrorStateIntervalFactor(phi, sqrt_info));
    }

private:
    Matrix15d phi_;
    Matrix15d sqrt_info_;
};

}  // namespace ct_fgo_sim::factors
