#pragma once

#include <ceres/ceres.h>

namespace ct_fgo_sim::factors {

struct RoadProfileGnssFactor {
    RoadProfileGnssFactor(double u, double meas_h_ned, double sigma_m)
        : u_(u),
          meas_h_ned_(meas_h_ned),
          inv_sigma_(1.0 / std::max(1.0e-6, sigma_m)) {}

    template <typename T>
    bool operator()(const T* const hi, const T* const hj, T* residuals) const {
        const T h_pred = (T(1.0) - T(u_)) * hi[0] + T(u_) * hj[0];
        residuals[0] = T(inv_sigma_) * (h_pred - T(meas_h_ned_));
        return true;
    }

    static ceres::CostFunction* Create(double u, double meas_h_ned, double sigma_m) {
        return new ceres::AutoDiffCostFunction<RoadProfileGnssFactor, 1, 1, 1>(
            new RoadProfileGnssFactor(u, meas_h_ned, sigma_m));
    }

private:
    double u_;
    double meas_h_ned_;
    double inv_sigma_;
};

struct DualRoadProfileGnssFactor {
    DualRoadProfileGnssFactor(
        double base_u,
        double residual_u,
        double meas_h_ned,
        double sigma_m)
        : base_u_(base_u),
          residual_u_(residual_u),
          meas_h_ned_(meas_h_ned),
          inv_sigma_(1.0 / std::max(1.0e-6, sigma_m)) {}

    template <typename T>
    bool operator()(
        const T* const base_i,
        const T* const base_j,
        const T* const residual_i,
        const T* const residual_j,
        T* residuals) const {
        const T base_pred = (T(1.0) - T(base_u_)) * base_i[0] + T(base_u_) * base_j[0];
        const T residual_pred =
            (T(1.0) - T(residual_u_)) * residual_i[0] + T(residual_u_) * residual_j[0];
        residuals[0] = T(inv_sigma_) * (base_pred + residual_pred - T(meas_h_ned_));
        return true;
    }

    static ceres::CostFunction* Create(
        double base_u,
        double residual_u,
        double meas_h_ned,
        double sigma_m) {
        return new ceres::AutoDiffCostFunction<DualRoadProfileGnssFactor, 1, 1, 1, 1, 1>(
            new DualRoadProfileGnssFactor(base_u, residual_u, meas_h_ned, sigma_m));
    }

private:
    double base_u_;
    double residual_u_;
    double meas_h_ned_;
    double inv_sigma_;
};

struct RoadProfileAnchorFactor {
    RoadProfileAnchorFactor(double ref_h_ned, double sigma_m)
        : ref_h_ned_(ref_h_ned),
          inv_sigma_(1.0 / std::max(1.0e-6, sigma_m)) {}

    template <typename T>
    bool operator()(const T* const h, T* residuals) const {
        residuals[0] = T(inv_sigma_) * (h[0] - T(ref_h_ned_));
        return true;
    }

    static ceres::CostFunction* Create(double ref_h_ned, double sigma_m) {
        return new ceres::AutoDiffCostFunction<RoadProfileAnchorFactor, 1, 1>(
            new RoadProfileAnchorFactor(ref_h_ned, sigma_m));
    }

private:
    double ref_h_ned_;
    double inv_sigma_;
};

struct RoadProfileSmoothnessFactor {
    explicit RoadProfileSmoothnessFactor(double sigma_m)
        : inv_sigma_(1.0 / std::max(1.0e-6, sigma_m)) {}

    template <typename T>
    bool operator()(const T* const hi, const T* const hj, T* residuals) const {
        residuals[0] = T(inv_sigma_) * (hj[0] - hi[0]);
        return true;
    }

    static ceres::CostFunction* Create(double sigma_m) {
        return new ceres::AutoDiffCostFunction<RoadProfileSmoothnessFactor, 1, 1, 1>(
            new RoadProfileSmoothnessFactor(sigma_m));
    }

private:
    double inv_sigma_;
};

struct RoadProfileCurvatureFactor {
    explicit RoadProfileCurvatureFactor(double sigma_m)
        : inv_sigma_(1.0 / std::max(1.0e-6, sigma_m)) {}

    template <typename T>
    bool operator()(const T* const him1, const T* const hi, const T* const hip1, T* residuals) const {
        residuals[0] = T(inv_sigma_) * (hip1[0] - T(2.0) * hi[0] + him1[0]);
        return true;
    }

    static ceres::CostFunction* Create(double sigma_m) {
        return new ceres::AutoDiffCostFunction<RoadProfileCurvatureFactor, 1, 1, 1, 1>(
            new RoadProfileCurvatureFactor(sigma_m));
    }

private:
    double inv_sigma_;
};

struct RoadProfileZeroFactor {
    explicit RoadProfileZeroFactor(double sigma_m)
        : inv_sigma_(1.0 / std::max(1.0e-6, sigma_m)) {}

    template <typename T>
    bool operator()(const T* const h, T* residuals) const {
        residuals[0] = T(inv_sigma_) * h[0];
        return true;
    }

    static ceres::CostFunction* Create(double sigma_m) {
        return new ceres::AutoDiffCostFunction<RoadProfileZeroFactor, 1, 1>(
            new RoadProfileZeroFactor(sigma_m));
    }

private:
    double inv_sigma_;
};

}  // namespace ct_fgo_sim::factors
