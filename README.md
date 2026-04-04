# CT_FGO_SIM zAxisPro

`CT_FGO_SIM_zAxisPro` is the working branch used for controlled road-profile reconstruction experiments with IMU, RTK, and optional NHC constraints.

This branch is no longer just the original minimal IMU+GNSS demo. It now contains the direct spline-state implementation used to study distance-domain road reconstruction, IRI preservation, and the effect of wrong constraints.

## Current Scope

- Continuous-time spline trajectory optimization
- IMU and GNSS/RTK fusion
- Optional body-frame NHC measurements
- Dense trajectory querying for distance-domain reconstruction
- Controlled simulation support for good-road and poor-road IRI cases

## Major Modifications In This Branch

### 1. Added direct spline-state mode

The original error-state pipeline used a nominal trajectory plus interpolated `delta_pos`, `delta_vel`, and `delta_theta` nodes. In the road-profile experiments, that structure allowed vertical GNSS observations to be written into `delta_pos` nodes too directly, which produced artificial mid-band waviness in the reconstructed road profile.

To address that, this branch adds:

- `use_direct_spline_state`
- direct GNSS factors on spline control points
- direct continuous inertial factors on spline control points
- dense state querying through the spline state itself

In this mode, position is no longer carried mainly by `delta_pos` nodes.

### 2. Added NHC measurement support

This branch adds:

- `nhc_file` loading
- `NhcMeasurement` data type
- `ErrorStateBodyVelocityNhcFactor`
- optional NHC factor insertion during optimization

This was introduced to support controlled experiments with:

- correct NHC
- wrong vertical NHC
- wrong lateral NHC
- NHC on/off comparison

### 3. Added dense trajectory output

This branch adds:

- `output_query_dt_s`
- `dense_trajectory_enu.txt`

The dense output is used by the MATLAB evaluation pipeline to compute distance-domain road curves and IRI without relying only on sparse GNSS-timestamp exports.

## Important Bug Fix

### Coordinate-frame bug in `continuous_inertial_factor`

One major error in an earlier version of this branch was a coordinate-frame inconsistency inside `continuous_inertial_factor`.

The spline control-point translation was stored in local `NED`, but the inertial factor interpreted it with inconsistent axis and sign usage. The most important mistakes were:

- wrong interpretation of horizontal position increments
- wrong height sign
- wrong gravity sign
- inconsistent transport-rate expression

This caused the direct spline-state graph to behave incorrectly and produced unrealistic trajectory errors.

The current branch fixes that by making the factor consistent with `NED`:

- `lat = lat0 + north / (rm + h)`
- `h = h0 - down`
- `gravity_n = [0, 0, +g]`
- transport-rate terms computed from `v_ned`

This fix is critical. Without it, the direct spline-state branch does not produce reliable geometry.

## Parameter Tuning Attempts

This branch includes a number of parameter investigations carried out for the controlled IRI experiments.

### Initialization

- added long static alignment before motion in the simulation pipeline
- used the same alignment result for both KF and CT
- avoided starting directly with a velocity step

### GNSS vertical weighting

Tested by increasing `gnss_sigma_vertical_m` so that CT and KF use comparable vertical RTK confidence.

Observed effect:

- too small: vertical GNSS writes profile waviness into the solution
- too large: the trajectory becomes over-smoothed and IRI is underestimated

### Control-point spacing

Tested by changing `kf_interval_sec`.

Observed effect:

- too dense: easier to reproduce GNSS-induced mid-band waviness
- too sparse: over-smoothing and loss of road-profile detail

### IMU residual sigmas

Tested by gradually reducing:

- `imu_sigma_accel_mps2`
- `imu_sigma_gyro_rps`

Observed effect:

- modest reduction can improve geometry slightly
- too small causes instability or drives the solution into an over-constrained regime

### NHC on/off comparison

The branch now supports explicit comparison between:

- `nhc_off`
- `nhc_on`

This is intended for later controlled studies of correct and wrong NHC behavior. The NHC implementation should always be interpreted together with the direct spline-state structure and the current simulation setup.

## What Was Learned From The Debugging

The most important lessons from this branch are:

1. A low global RMSE does not guarantee correct road-profile reconstruction.
2. If GNSS vertical observations are allowed to directly drive free position-error nodes, the reconstructed road profile can develop false mid-band waviness.
3. IRI can therefore be badly biased even when absolute position RMSE is small.
4. The coordinate frame inside the continuous inertial factor must be strictly consistent with the spline state definition.
5. For road-profile applications, preserving the structure of the distance-domain curve matters as much as minimizing navigation RMSE.

## Repository Cleanliness

This repository should only retain source code, configs, and useful tools.

The following should not be committed:

- build folders
- local debug outputs
- temporary stdout and stderr logs
- generated trajectory folders

The `.gitignore` has been updated accordingly.

## Build

```powershell
cmake -S D:\Code\CT_FGO_SIM_zAxisPro -B D:\Code\CT_FGO_SIM_zAxisPro\build_zAxisPro
cmake --build D:\Code\CT_FGO_SIM_zAxisPro\build_zAxisPro --config Release
```

## Run

```powershell
D:\Code\CT_FGO_SIM_zAxisPro\build_zAxisPro\Release\ct_fgo_sim_main.exe D:\Code\CT_FGO_SIM_zAxisPro\config\minimal.yaml
```

## Files Most Relevant To The Current zAxisPro Work

- `src/core/system.cpp`
- `include/ct_fgo_sim/core/system.h`
- `include/ct_fgo_sim/factors/continuous_inertial_factor.h`
- `include/ct_fgo_sim/factors/error_state_nhc_factor.h`
- `include/ct_fgo_sim/io/text_measurement_io.h`
- `include/ct_fgo_sim/types.h`

## Status

This branch is the important experimental branch used for the current thesis-oriented road reconstruction work.

It should be treated as:

- the reference branch for direct spline-state CT-FGO experiments
- the branch where the coordinate-frame error was fixed
- the branch where NHC support and dense output were introduced

## 2026-04 Road-Profile Findings

Recent road-profile experiments compared three directions:

- direct spline-state baseline without road-profile state
- single-layer distance-domain road-profile state
- dual-layer road-profile state with base/residual splitting

### Branches

- `paper/single-layer-param-laws`: current working branch for parameter-law mapping
- `paper/dual-road-profile-unusable`: archived dual-layer branch kept only for reference; it is marked as unusable because it did not outperform the single-layer results in the controlled IRI experiments

### Current conclusion

The single-layer road-profile state is currently the most useful structure.

The dual-layer branch was reasonable in principle, but in practice it failed to produce a robust parameter region that improved both:

- low-IRI cases around `IRI ~= 2`
- high-IRI cases around `IRI ~= 10`

The dual-layer versions repeatedly showed one of the following failure modes:

- suppressing `IRI ~= 10` too strongly
- over-amplifying `IRI ~= 2`
- improving one side only by damaging the other

For this reason, the current work has been rolled back to the single-layer structure.

### Single-layer parameter-law summary

The most important single-layer parameters are:

- `road_profile_ds_m`: distance-domain node spacing of the road-profile state
- `road_profile_anchor_sigma_m`: low-frequency anchoring strength
- `gnss_sigma_vertical_m`: vertical GNSS weighting

Current observations from the parameter map are:

1. `road_profile_ds_m` is the dominant parameter controlling IRI scaling.
2. Smaller `road_profile_ds_m` gives higher profile bandwidth and tends to enlarge IRI.
3. Larger `road_profile_ds_m` smooths the profile more strongly and tends to shrink IRI.
4. `gnss_sigma_vertical_m` has a scenario-dependent optimum; it is not true that smaller is always better.
5. `road_profile_anchor_sigma_m` mainly acts as a secondary tuning parameter and has much weaker influence than `road_profile_ds_m`.

### Best currently observed single-layer settings in the controlled map

For the tested controlled cases:

- `IRI ~= 2`: best match was near `gnss_sigma_vertical_m = 0.003`, `road_profile_ds_m = 0.25`, `road_profile_anchor_sigma_m = 0.02`
- `IRI ~= 10`: best match was near `gnss_sigma_vertical_m = 0.0008`, `road_profile_ds_m = 0.25`, `road_profile_anchor_sigma_m = 0.02`

These values should be treated as empirical findings for the current simulation chain, not as universal defaults.
