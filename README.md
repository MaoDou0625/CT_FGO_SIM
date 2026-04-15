# CT_FGO_SIM zAxisPro-error-state-mature

This branch is a **mature error-state FGO** line: nominal trajectory plus interpolated `delta_pos`, `delta_vel`, `delta_theta` (and bias nodes), with **IMU interval propagation** factors and optional NHC. The earlier **direct spline-state (Zevesilov SE(3) control-point) path has been removed** from the codebase to reduce dual-mode confusion and maintenance cost.

Road-profile / IRI experiments can still use dense trajectory output and vertical GNSS tuning; interpret results in the error-state formulation.

## Current scope

- Error-state continuous-time FGO on a knot grid aligned with `nominal_nav` / IMU boundaries
- IMU and GNSS/RTK fusion (`ErrorStateIntervalFactor`, `ErrorStateGnss*` factors)
- Optional body-frame NHC (`ErrorStateBodyVelocityNhcFactor`)
- Dense trajectory querying (`output_query_dt_s`, `dense_trajectory_enu.txt`)

## YAML highlights

- `kf_interval_sec` — knot spacing (also used as `spline_dt_s` in config)
- `use_gnss_factors`, `use_imu_factors` — disable both for pure inertial replay
- `output_query_dt_s` — dense export step; `0` uses IMU rate fallback in code
- `body_frame` / `nhc_file` — optional NHC

There is **no** `use_direct_spline_state` switch; only one optimization backend remains.

## Historical note

The repository previously carried a **direct spline-state** experiment (`ContinuousGnssFactor` / `ContinuousInertialFactor` on `SE3` control points). That path was removed on this branch. Historical discussion of vertical GNSS coupling into `delta_pos` and IRI metrics still applies to **error-state** tuning.

## Build

```powershell
cmake -S D:\Code\CT_FGO_SIM_zAxisPro -B D:\Code\CT_FGO_SIM_zAxisPro\build
cmake --build D:\Code\CT_FGO_SIM_zAxisPro\build --config Release
```

## Run

```powershell
D:\Code\CT_FGO_SIM_zAxisPro\build\Release\ct_fgo_sim_main.exe D:\Code\CT_FGO_SIM_zAxisPro\config\minimal.yaml
```

## Primary implementation files

- `src/core/system.cpp` — orchestration, outputs
- `src/core/factor_graph_session.cpp` — Ceres error-state graph
- `src/navigation/interval_propagation.cpp` — knot interval Φ/Q for IMU factors
- `include/ct_fgo_sim/factors/error_state_interval_factor.h`
- `include/ct_fgo_sim/factors/error_state_gnss_factor.h`
- `include/ct_fgo_sim/factors/error_state_nhc_factor.h`

## Status

**`zAxisPro-error-state-mature`** — error-state FGO only; suitable as the maintained variant for thesis-oriented road reconstruction without the removed direct-spline code path.
