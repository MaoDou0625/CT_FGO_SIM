#pragma once

#include "ct_fgo_sim/types.h"

#include <ceres/ceres.h>
#include <Eigen/Core>

namespace ct_fgo_sim {

struct FactorGraphSession;

/// Information-form linear prior on one knot's 21D state in **Ceres block order**
/// `[dtheta; dvel; dpos; dbg; dba; dsg; dsa]` (matches interval-factor Jacobians).
struct MarginalizationFrontier {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    bool valid = false;
    int anchor_knot_index = -1;
    VectorErrorState x0 = VectorErrorState::Zero();
    MatrixErrorState H = MatrixErrorState::Zero();
    VectorErrorState g = VectorErrorState::Zero();
    void reset() {
        valid = false;
        anchor_knot_index = -1;
    }
};

/// Linearize interval + GNSS (+ optional prior on k_drop), Schur-complement out knot k_drop,
/// store quadratic on knot k_drop+1 in `out` (anchor_knot_index = k_drop + 1).
bool MarginalizeOldestKnotTwoKnotWindow(
    int k_drop,
    const FactorGraphSession& session,
    const MarginalizationFrontier* prior_on_k_drop,
    MarginalizationFrontier& out_on_k_drop_plus_1);

ceres::CostFunction* CreateMarginalizationPriorCost(const MarginalizationFrontier& frontier);

}  // namespace ct_fgo_sim
