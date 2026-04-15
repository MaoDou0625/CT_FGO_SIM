#pragma once

#include "ct_fgo_sim/types.h"

namespace ct_fgo_sim {

inline Matrix3d SkewSymmetric3(const Vector3d& vector) {
    Matrix3d mat;
    mat << 0.0, -vector.z(), vector.y(),
           vector.z(), 0.0, -vector.x(),
          -vector.y(), vector.x(), 0.0;
    return mat;
}

}  // namespace ct_fgo_sim
