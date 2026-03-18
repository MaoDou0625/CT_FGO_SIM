#pragma once

#include "ct_fgo_sim/types.h"

#include <cmath>

namespace ct_fgo_sim {

inline constexpr double kWgs84Wie = 7.2921151467e-5;
inline constexpr double kWgs84Ra = 6378137.0;
inline constexpr double kWgs84E1 = 0.0066943799901413156;

class Earth {
public:
    static double Gravity(const Vector3d& blh) {
        double sin2 = std::sin(blh.x());
        sin2 *= sin2;
        return 9.7803267715 * (1 + 0.0052790414 * sin2 + 0.0000232718 * sin2 * sin2) +
               blh.z() * (0.0000000043977311 * sin2 - 0.0000030876910891) +
               0.0000000000007211 * blh.z() * blh.z();
    }

    static Vector3d BlhToEcef(const Vector3d& blh) {
        const double cos_lat = std::cos(blh.x());
        const double sin_lat = std::sin(blh.x());
        const double cos_lon = std::cos(blh.y());
        const double sin_lon = std::sin(blh.y());
        const double rn = Rn(blh.x());
        const double rnh = rn + blh.z();
        return {
            rnh * cos_lat * cos_lon,
            rnh * cos_lat * sin_lon,
            (rnh - rn * kWgs84E1) * sin_lat,
        };
    }

    static Vector3d EcefToBlh(const Vector3d& ecef) {
        const double p = std::sqrt(ecef.x() * ecef.x() + ecef.y() * ecef.y());
        double lat = std::atan(ecef.z() / (p * (1.0 - kWgs84E1)));
        const double lon = 2.0 * std::atan2(ecef.y(), ecef.x() + p);
        double h = 0.0;
        double h_prev = 0.0;
        do {
            h_prev = h;
            const double rn = Rn(lat);
            h = p / std::cos(lat) - rn;
            lat = std::atan(ecef.z() / (p * (1.0 - kWgs84E1 * rn / (rn + h))));
        } while (std::fabs(h - h_prev) > 1.0e-4);
        return {lat, lon, h};
    }

    static Matrix3d Cne(const Vector3d& blh) {
        const double sin_lat = std::sin(blh.x());
        const double sin_lon = std::sin(blh.y());
        const double cos_lat = std::cos(blh.x());
        const double cos_lon = std::cos(blh.y());

        Matrix3d dcm;
        dcm << -sin_lat * cos_lon, -sin_lon, -cos_lat * cos_lon,
               -sin_lat * sin_lon,  cos_lon, -cos_lat * sin_lon,
                cos_lat,            0.0,     -sin_lat;
        return dcm;
    }

    static Vector3d GlobalToLocal(const Vector3d& origin_blh, const Vector3d& global_blh) {
        const Vector3d ecef0 = BlhToEcef(origin_blh);
        const Vector3d ecef1 = BlhToEcef(global_blh);
        return Cne(origin_blh).transpose() * (ecef1 - ecef0);
    }

    static Vector3d LocalToGlobal(const Vector3d& origin_blh, const Vector3d& local_enu) {
        const Vector3d ecef0 = BlhToEcef(origin_blh);
        const Vector3d ecef1 = ecef0 + Cne(origin_blh) * local_enu;
        return EcefToBlh(ecef1);
    }

    static Vector3d Iewn(double lat_rad) {
        return {kWgs84Wie * std::cos(lat_rad), 0.0, -kWgs84Wie * std::sin(lat_rad)};
    }

private:
    static double Rn(double lat_rad) {
        const double sin_lat = std::sin(lat_rad);
        return kWgs84Ra / std::sqrt(1.0 - kWgs84E1 * sin_lat * sin_lat);
    }
};

}  // namespace ct_fgo_sim
