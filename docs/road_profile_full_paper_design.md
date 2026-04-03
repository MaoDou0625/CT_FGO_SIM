# Road Profile Full-Paper Design

## Goal

Improve `CT_FGO_SIM` so the continuous-time solver can preserve the IRI-sensitive vertical wavelength band without directly writing RTK noise into the recovered height profile.

The target is not to mimic a discrete KF implementation mechanically.
The target is to give the CT system the same useful information flow that the KF currently has in the vertical channel:

1. local vertical degrees of freedom
2. direct vertical measurement injection
3. weak and controllable smoothness prior instead of a hard global low-pass shape

## Current problem

The current direct spline-state mode uses one smooth `SE3` spline to carry:

1. horizontal motion
2. attitude
3. vertical trajectory

GNSS position factors are applied directly on the spline pose.
IMU continuous factors also constrain the same spline.

This structure is good for geometric smoothness, but it makes the vertical channel too dependent on:

1. cubic spline basis smoothness
2. control-point spacing
3. inertial consistency on the same trajectory

As a result, the solver tends to:

1. suppress the `1.25-4 m` IRI-sensitive wavelength band
2. redistribute energy into longer wavelengths
3. keep good RMSE while still underestimating IRI

Lowering `gnss_sigma_vertical_m` helps because it forces stronger vertical data injection, but this is only compensation, not a structural fix.

## Design principles

The full-paper version should satisfy four principles.

### 1. Vertical decoupling

Vertical road-sensitive content must not be forced to live entirely inside the same smooth `SE3` spline that carries the vehicle body trajectory.

### 2. Controlled local update

Vertical GNSS information should update a local vertical state directly, similar to how a KF can correct `z` at each measurement epoch.

### 3. Noise rejection by explicit priors

The system must reject RTK noise through explicit vertical priors and robust costs, not by relying on the whole trajectory parameterization to be globally smooth.

### 4. Road-profile-oriented output

The final quantity used for IRI should move toward a road-profile estimate, not just a vehicle-center trajectory height.

## Proposed architecture

The design is split into three stages so the repository can evolve without breaking the current direct spline-state pipeline.

## Stage A: Vertical residual field on top of the current CT spline

This is the first implementation step.
It is the lowest-risk path and should be built before any full road-profile state.

### State

Keep the existing direct spline-state variables:

1. spline control-point `SE3`
2. gyro bias nodes
3. accel bias nodes

Add a new vertical-only node array:

1. `delta_z_nodes_`

Optional second step:

1. `delta_vz_nodes_`

### Meaning

The total vertical output becomes:

`z_total(t) = z_spline(t) + delta_z(t)`

where:

1. `z_spline(t)` is the current smooth body trajectory height
2. `delta_z(t)` is a local vertical correction field that can preserve the sensitive wavelength band

### Factors

Add three new vertical factors.

1. `VerticalGnssFactor`
   Directly constrains `z_total(t)` against GNSS height.

2. `VerticalSmoothnessFactor`
   Regularizes adjacent `delta_z` nodes with a weak prior.
   This replaces structural over-smoothing with tunable smoothing.

3. `VerticalInertialFactor`
   Uses the vertical component of the accelerometer prediction to constrain the second derivative of `delta_z`.
   This should be added after the GNSS-only vertical path is stable.

### Expected effect

This stage should:

1. reduce suppression of `1.25-4 m`
2. keep the current CT horizontal and attitude behavior unchanged
3. avoid raw RTK write-through because `delta_z` is still regularized

## Stage B: Vertical dynamic sub-system

After Stage A is stable, the vertical correction field should become a small dynamical model instead of only a residual field.

### State

Upgrade vertical nodes to:

1. `delta_z`
2. `delta_vz`

Optional later:

1. `delta_az`

### Prior

Use a weak process model between adjacent nodes:

1. constant velocity
2. or constant acceleration

This makes the vertical channel behave more like a KF while staying in the graph framework.

### Benefit

Compared with a pure spline correction, this allows:

1. more local vertical adaptation
2. better interpretation of IMU vertical evidence
3. clearer tuning of bandwidth versus noise leakage

## Stage C: Road-profile state in distance domain

This is the full-paper target structure.

The key idea is to separate:

1. vehicle motion state
2. road elevation profile state

### State split

Vehicle state:

1. body `SE3` continuous trajectory
2. body velocity and biases

Road state:

1. `h_road(s_k)` on a distance-domain knot grid
2. optionally road slope or curvature state

### Coupling

Introduce a mapping from time to distance:

1. use current `s(t)` from the vehicle solution
2. query road profile at the corresponding wheel-track distance

The vertical observation model then becomes:

`z_vehicle(t) = h_road(s(t)) + suspension/lever/body terms`

In the current simplified simulation, suspension terms can remain omitted at first.

### Benefit

This is the first architecture that matches the actual IRI object:

1. IRI is a road-profile metric
2. not a generic vehicle trajectory smoothness metric

## Preventing RTK-driven false IRI inflation

A vertical-decoupled model can fail if it simply tracks RTK noise.
The full-paper design must explicitly defend against that.

### Required protections

1. robust vertical GNSS loss
2. adjacent-node smoothness prior
3. optional innovation gating on vertical GNSS residual
4. vertical bandwidth tuning through node spacing and smoothness, not through raw GNSS sigma alone

### Tuning rule

The tuning target should not be minimum RMSE alone.
The target should jointly consider:

1. `delta IRI`
2. `50 m` window IRI RMSE
3. `rho_iri`
4. `rho_long`

The desired outcome is:

1. recover `rho_iri` toward `1`
2. avoid pushing `rho_long` too high
3. avoid large positive IRI bias caused by RTK noise leakage

## Recommended implementation order

### Step 1

Synchronize `CT_FGO_SIM` source to the current `zAxisPro` direct spline-state baseline.

### Step 2

Implement `delta_z_nodes_` and a `VerticalGnssFactor`.

### Step 3

Implement `VerticalSmoothnessFactor`.

### Step 4

Expose new config parameters:

1. `enable_vertical_profile_field`
2. `vertical_node_dt_s`
3. `vertical_gnss_sigma_m`
4. `vertical_smooth_sigma_m`
5. `vertical_gnss_cauchy_scale_m`

### Step 5

Update solver output so height export includes:

1. base spline height
2. vertical correction
3. total height

### Step 6

Run the existing IRI≈2 and IRI≈10 simulation suite and compare:

1. height overlay
2. wavelength spectrum
3. `50 m` window IRI
4. `rho_iri / rho_long`

### Step 7

Only after Stage A is validated, add vertical inertial coupling and then move toward distance-domain road-profile states.

## Immediate repository changes implied by this design

The next code iteration should touch:

1. `include/ct_fgo_sim/core/system.h`
2. `src/core/system.cpp`
3. new factor headers under `include/ct_fgo_sim/factors/`
4. config loading and output summary

The first implementation should avoid touching the horizontal solver path as much as possible.
That keeps the experiment interpretable and reduces the risk of accidentally regressing the current direct spline-state improvements.
