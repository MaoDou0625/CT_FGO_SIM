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

}  // namespace ct_fgo_sim::factors
