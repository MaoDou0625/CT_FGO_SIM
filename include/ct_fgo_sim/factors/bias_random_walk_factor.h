#pragma once

#include <ceres/ceres.h>
#include <Eigen/Core>

namespace ct_fgo_sim::factors {

class BiasRandomWalkFactor : public ceres::SizedCostFunction<3, 3, 3> {
public:
    BiasRandomWalkFactor(double dt, double sigma_rw, double tau) : dt_(dt), sigma_rw_(sigma_rw), tau_(tau) {}

    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override {
        const Eigen::Map<const Eigen::Vector3d> b0(parameters[0]);
        const Eigen::Map<const Eigen::Vector3d> b1(parameters[1]);
        Eigen::Map<Eigen::Vector3d> res(residuals);
        const double decay = std::exp(-dt_ / tau_);
        const double sigma = sigma_rw_ * std::sqrt(std::max(1.0e-12, 1.0 - decay * decay));
        res = (b1 - decay * b0) / sigma;

        if (jacobians != nullptr) {
            if (jacobians[0] != nullptr) {
                Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> j0(jacobians[0]);
                j0.setIdentity();
                j0 *= (-decay / sigma);
            }
            if (jacobians[1] != nullptr) {
                Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> j1(jacobians[1]);
                j1.setIdentity();
                j1 *= (1.0 / sigma);
            }
        }
        return true;
    }

    static ceres::CostFunction* Create(double dt, double sigma_rw, double tau) {
        return new BiasRandomWalkFactor(dt, sigma_rw, tau);
    }

private:
    double dt_;
    double sigma_rw_;
    double tau_;
};

}  // namespace ct_fgo_sim::factors
