# CT_FGO_SIM · sliding window + marginalization

This branch extends the **mature error-state FGO** baseline (`zAxisPro-error-state-mature`): nominal trajectory plus interpolated `delta_pos`, `delta_vel`, `delta_theta` (and bias nodes), with **IMU interval propagation** factors and optional NHC. The focus here is **sliding-window optimization** and **marginalization** (linearized priors on dropped states) for bounded-memory or streaming-style estimation on the same error-state graph.

Road-profile / IRI experiments can still use dense trajectory output and vertical GNSS tuning; interpret results in the error-state formulation.

## Inherited scope (unchanged formulation)

- Error-state continuous-time FGO on a knot grid aligned with `nominal_nav` / IMU boundaries
- IMU and GNSS/RTK fusion (`ErrorStateIntervalFactor`, `ErrorStateGnss*` factors)
- Optional body-frame NHC (`ErrorStateBodyVelocityNhcFactor`)
- Dense trajectory querying (`output_query_dt_s`, `dense_trajectory_enu.txt`)

## Target scope on this branch

- **Sliding window** over knot / interval states (fixed lag or controlled window growth)
- **Marginalization** of states that leave the window (Schur complement / information-form prior carried forward)
- Config surface for window length, marginalization policy, and reuse of existing YAML knobs where possible

## YAML highlights

- `kf_interval_sec` — knot spacing (also used as `spline_dt_s` in config)
- `use_gnss_factors`, `use_imu_factors` — disable both for pure inertial replay
- `output_query_dt_s` — dense export step; `0` uses IMU rate fallback in code
- `body_frame` / `nhc_file` — optional NHC

There is **no** `use_direct_spline_state` switch. **v0.2.0+** uses **GTSAM only** for navigation solves; Ceres remains linked for marginalization Jacobian evaluation (`Evaluate` on existing cost functors) only.

## Historical note

The repository previously carried a **direct spline-state** experiment (`ContinuousGnssFactor` / `ContinuousInertialFactor` on `SE3` control points). That path was removed on the error-state baseline. Historical discussion of vertical GNSS coupling into `delta_pos` and IRI metrics still applies to **error-state** tuning.

## Build

```powershell
cmake -S D:\Code\CT_FGO_SIM_sliding-window -B D:\Code\CT_FGO_SIM_sliding-window\build
cmake --build D:\Code\CT_FGO_SIM_sliding-window\build --config Release
```

GTSAM is **required** (`find_package(GTSAM REQUIRED)`).

**GTSAM-only navigation limits (v0.2.0):** the batch solver rejects non-zero IMU lever arm, active NHC body-velocity axes, `estimate_q_body_imu` with NHC, and `use_imu_factors=false`. Those cases used to be Ceres-only; use zero lever arm and NHC axes disabled (or match the shipped outage YAMLs) until GTSAM factors cover them.

## Run

```powershell
D:\Code\CT_FGO_SIM_sliding-window\build\Release\ct_fgo_sim_main.exe D:\Code\CT_FGO_SIM_sliding-window\config\minimal.yaml
```

## Primary implementation files

- `src/core/system.cpp` — orchestration, outputs
- `src/core/factor_graph_backend.cpp` — GTSAM window/batch solve (+ marginal prior)
- `src/navigation/interval_propagation.cpp` — knot interval Φ/Q for IMU factors
- `include/ct_fgo_sim/factors/error_state_interval_factor.h`
- `include/ct_fgo_sim/factors/error_state_gnss_factor.h`
- `include/ct_fgo_sim/factors/error_state_nhc_factor.h`

## Status

**`zAxisPro-sliding-window-marginalization`** — development line for **sliding-window** error-state FGO with **marginalization**; builds on `zAxisPro-error-state-mature` without restoring the removed direct-spline path. Local checkout folder: **`CT_FGO_SIM_sliding-window`**.
