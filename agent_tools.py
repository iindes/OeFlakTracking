"""
agent_tools.py  —  OeFlakTrack ADK Tool Definitions
====================================================
Five pure-Python tool functions registered with the Google ADK agent.

ADK auto-generates the JSON tool schema from:
  • Function name       → tool identifier
  • Type-annotated args → parameter schema (type + required/optional)
  • Docstring           → tool + parameter descriptions sent to the LLM

Design rules
------------
  • Every function returns a plain dict with a fixed top-level schema so
    the LLM sees consistent structure regardless of which tool ran.
  • No side-effects: no ZeroMQ, no sleep, no file I/O — computation only.
  • Inputs are validated and clamped so the LLM cannot pass illegal ranges.
  • All float outputs are rounded to avoid noise in the LLM context window.
"""

import math
import time
from typing import Any

import numpy as np

# ── Shared response builders ──────────────────────────────────────────────

def _ok(tool: str, result: Any, summary: str, t0: float) -> dict:
    """Build a successful tool response envelope."""
    return {
        "status": "success",
        "tool": tool,
        "result": result,
        "summary": summary,
        "computation_ms": round((time.perf_counter() - t0) * 1_000, 3),
    }


def _err(tool: str, message: str) -> dict:
    """Build an error tool response envelope."""
    return {
        "status": "error",
        "tool": tool,
        "result": {},
        "summary": f"Tool {tool!r} failed: {message}",
        "computation_ms": 0.0,
    }


# ── Tool 1 ── Radar Simulation ────────────────────────────────────────────

def simulate_radar_scan(
    steps: int = 10,
    dt: float = 1.0,
    noise_range_m: float = 50.0,
    noise_az_rad: float = 0.02,
    noise_el_rad: float = 0.01,
) -> dict:
    """
    Simulate a 3-D radar scan and return synthetic spherical-coordinate measurements.

    Uses a constant-velocity aircraft model: initial position (10000, 5000, 3000) m,
    velocity (-200, -50, -10) m/s. Adds independent Gaussian noise to range,
    azimuth, and elevation at every scan.

    Args:
        steps: Number of radar scan intervals to generate. Range: 1–200. Default: 10.
        dt: Scan interval in seconds. Range: 0.1–10.0. Default: 1.0.
        noise_range_m: 1-sigma range measurement noise in metres. Range: 0–500. Default: 50.
        noise_az_rad: 1-sigma azimuth noise in radians. Range: 0–0.5. Default: 0.02.
        noise_el_rad: 1-sigma elevation noise in radians. Range: 0–0.5. Default: 0.01.

    Returns:
        dict with 'measurements' (list of scan dicts each containing step, timestamp_s,
        range_m, azimuth_rad, elevation_rad, true_x_m, true_y_m, true_z_m), trajectory
        start/end positions, and noise parameters used.
    """
    t0 = time.perf_counter()
    steps         = max(1, min(200, int(steps)))
    dt            = max(0.1, min(10.0, float(dt)))
    noise_range_m = max(0.0, min(500.0, float(noise_range_m)))
    noise_az_rad  = max(0.0, min(0.5,   float(noise_az_rad)))
    noise_el_rad  = max(0.0, min(0.5,   float(noise_el_rad)))

    rng   = np.random.default_rng(seed=42)
    state = np.array([10_000.0, 5_000.0, 3_000.0, -200.0, -50.0, -10.0])
    F     = np.eye(6, dtype=float)
    F[:3, 3:] = np.eye(3) * dt

    measurements = []
    for i in range(steps):
        state      = F @ state
        px, py, pz = state[0], state[1], state[2]
        rho        = math.sqrt(px**2 + py**2)
        r          = math.sqrt(rho**2 + pz**2)
        az         = math.atan2(py, px)
        el         = math.atan2(pz, rho)

        measurements.append({
            "step":           i,
            "timestamp_s":    round(i * dt, 3),
            "range_m":        round(r  + float(rng.normal(0, noise_range_m)), 2),
            "azimuth_rad":    round(az + float(rng.normal(0, noise_az_rad)),  6),
            "elevation_rad":  round(el + float(rng.normal(0, noise_el_rad)),  6),
            "true_x_m":       round(float(px), 2),
            "true_y_m":       round(float(py), 2),
            "true_z_m":       round(float(pz), 2),
        })

    first, last = measurements[0], measurements[-1]
    summary = (
        f"Simulated {steps} scans over {steps*dt:.1f}s. "
        f"Target: ({first['true_x_m']:.0f}, {first['true_y_m']:.0f}, "
        f"{first['true_z_m']:.0f}) → ({last['true_x_m']:.0f}, "
        f"{last['true_y_m']:.0f}, {last['true_z_m']:.0f}) m."
    )

    return _ok("simulate_radar_scan", {
        "n_steps":         steps,
        "dt_s":            dt,
        "noise_range_m":   noise_range_m,
        "noise_az_rad":    noise_az_rad,
        "noise_el_rad":    noise_el_rad,
        "measurements":    measurements,
        "start_pos_m":     [first["true_x_m"], first["true_y_m"], first["true_z_m"]],
        "end_pos_m":       [last["true_x_m"],  last["true_y_m"],  last["true_z_m"]],
    }, summary, t0)


# ── Tool 2 ── EKF3D Filtering ─────────────────────────────────────────────

def run_ekf3d_filter(
    steps: int = 30,
    dt: float = 1.0,
    noise_range_m: float = 50.0,
    noise_az_rad: float = 0.02,
    noise_el_rad: float = 0.01,
) -> dict:
    """
    Run the 3-D EKF on a simulated radar trajectory and return the filtered state history.

    Generates the same synthetic trajectory as simulate_radar_scan, then applies the
    non-linear EKF (spherical measurement → Jacobian linearisation → Cartesian estimate).
    Returns per-step filtered positions, velocity estimates, and running RMSE.

    Args:
        steps: Number of steps to simulate and filter. Range: 5–200. Default: 30.
        dt: Scan interval in seconds. Range: 0.1–10.0. Default: 1.0.
        noise_range_m: 1-sigma range noise (metres). Default: 50.
        noise_az_rad: 1-sigma azimuth noise (radians). Default: 0.02.
        noise_el_rad: 1-sigma elevation noise (radians). Default: 0.01.

    Returns:
        dict with 'filtered_states' (list of per-step dicts containing step, est_x_m,
        est_y_m, est_z_m, est_vx, est_vy, est_vz, ekf_error_m, raw_error_m),
        plus final_rmse_ekf_m and final_rmse_raw_m scalars.
    """
    t0 = time.perf_counter()
    from EKF3DTracker import EKF3D, _generate_3d_telemetry

    steps = max(5, min(200, int(steps)))
    data  = _generate_3d_telemetry(
        n_steps=steps, dt=dt,
        noise_r=noise_range_m, noise_az=noise_az_rad, noise_el=noise_el_rad,
        rng_seed=42,
    )

    ekf = EKF3D(dt=dt, noise_range=noise_range_m, noise_az=noise_az_rad,
                noise_el=noise_el_rad, process_noise_scale=0.5)
    ekf.set_initial_state([9_000.0, 4_200.0, 2_500.0, 0.0, 0.0, 0.0])

    filtered_states = []
    raw_sq, ekf_sq  = [], []

    for rec in data:
        z = np.array([[rec["r"]], [rec["az"]], [rec["el"]]])
        try:
            ekf.step(z)
        except RuntimeError:
            continue

        tx, ty, tz = rec["true_x"], rec["true_y"], rec["true_z"]
        ex, ey, ez = ekf.position
        cos_el     = math.cos(rec["el"])
        rx = rec["r"] * cos_el * math.cos(rec["az"])
        ry = rec["r"] * cos_el * math.sin(rec["az"])
        rz = rec["r"] * math.sin(rec["el"])

        ekf_err = math.sqrt((ex-tx)**2 + (ey-ty)**2 + (ez-tz)**2)
        raw_err = math.sqrt((rx-tx)**2 + (ry-ty)**2 + (rz-tz)**2)
        raw_sq.append(raw_err**2)
        ekf_sq.append(ekf_err**2)

        filtered_states.append({
            "step":        int(rec["t"]),
            "est_x_m":     round(float(ex), 2),
            "est_y_m":     round(float(ey), 2),
            "est_z_m":     round(float(ez), 2),
            "est_vx_mps":  round(float(ekf.velocity[0]), 3),
            "est_vy_mps":  round(float(ekf.velocity[1]), 3),
            "est_vz_mps":  round(float(ekf.velocity[2]), 3),
            "ekf_error_m": round(ekf_err, 2),
            "raw_error_m": round(raw_err, 2),
        })

    n            = len(raw_sq)
    rmse_raw     = round(math.sqrt(sum(raw_sq) / n), 2) if n else 0.0
    rmse_ekf     = round(math.sqrt(sum(ekf_sq) / n), 2) if n else 0.0
    noise_red    = round((rmse_raw - rmse_ekf) / rmse_raw * 100, 1) if rmse_raw else 0.0
    final_sigma  = [round(float(s), 2) for s in ekf.position_uncertainty]

    summary = (
        f"EKF filtered {n} steps. Final RMSE: {rmse_ekf}m (vs raw {rmse_raw}m). "
        f"Noise reduction: {noise_red}%. Final σ[X,Y,Z]: {final_sigma} m."
    )

    return _ok("run_ekf3d_filter", {
        "n_steps":          steps,
        "filtered_states":  filtered_states,
        "final_rmse_ekf_m": rmse_ekf,
        "final_rmse_raw_m": rmse_raw,
        "noise_reduction_pct": noise_red,
        "final_sigma_xyz_m":   final_sigma,
    }, summary, t0)


# ── Tool 3 ── Filter Performance Comparison ───────────────────────────────

def compare_filter_performance(
    n_steps: int = 80,
    dt: float = 1.0,
) -> dict:
    """
    Benchmark EKF3D against the standard linear Kalman Filter and raw sensor RMSE.

    Runs both filters on the same synthetic 80-step trajectory and reports full-run
    and steady-state (second half) RMSE, noise reduction percentages, and the
    improvement of EKF3D over the linear KF.

    Args:
        n_steps: Total simulation steps. Range: 20–200. Default: 80.
        dt: Scan interval in seconds. Range: 0.1–10.0. Default: 1.0.

    Returns:
        dict with 'full_run' and 'steady_state' RMSE tables for Raw, LinearKF,
        and EKF3D, plus ekf3d_improvement_over_linear_pct.
    """
    t0 = time.perf_counter()
    from benchmark_ekf3d import run_accuracy_benchmark

    n_steps = max(20, min(200, int(n_steps)))
    acc     = run_accuracy_benchmark(n_steps=n_steps, dt=dt, rng_seed=42)
    steady  = acc["steady"]
    full    = acc["full"]

    summary = (
        f"Over {n_steps} steps: EKF3D steady-state RMSE {steady['ekf']:.1f}m vs "
        f"LinearKF {steady['lkf']:.1f}m vs Raw {steady['raw']:.1f}m. "
        f"EKF3D noise reduction {steady['ekf_reduction']:.1f}%, "
        f"improvement over LinearKF: {steady['ekf_vs_lkf']:.1f}%."
    )

    return _ok("compare_filter_performance", {
        "n_steps": n_steps,
        "steady_state_window": f"steps {acc['steady_start']}–{n_steps-1}",
        "full_run": {
            "raw_rmse_m":    round(full["raw"], 2),
            "linear_kf_rmse_m": round(full["lkf"], 2),
            "ekf3d_rmse_m":  round(full["ekf"], 2),
        },
        "steady_state": {
            "raw_rmse_m":    round(steady["raw"], 2),
            "linear_kf_rmse_m": round(steady["lkf"], 2),
            "ekf3d_rmse_m":  round(steady["ekf"], 2),
            "linear_kf_noise_reduction_pct": round(steady["lkf_reduction"], 1),
            "ekf3d_noise_reduction_pct":     round(steady["ekf_reduction"], 1),
            "ekf3d_improvement_over_linear_pct": round(steady["ekf_vs_lkf"], 1),
        },
    }, summary, t0)


# ── Tool 4 ── Vectorised Trajectory Prediction ────────────────────────────

def predict_future_positions(
    x_m: float,
    y_m: float,
    z_m: float,
    vx_mps: float,
    vy_mps: float,
    vz_mps: float,
    n_steps: int = 5,
    dt: float = 1.0,
) -> dict:
    """
    Dead-reckon future Cartesian positions from a given state using constant-velocity model.

    Uses the closed-form vectorised solution: pos(k) = pos₀ + vel₀·k·dt (no Python loop).

    Args:
        x_m:    Current X position in metres.
        y_m:    Current Y position in metres.
        z_m:    Current Z position in metres.
        vx_mps: Current X velocity in m/s.
        vy_mps: Current Y velocity in m/s.
        vz_mps: Current Z velocity in m/s.
        n_steps: Number of future steps to predict. Range: 1–100. Default: 5.
        dt:     Time step in seconds. Range: 0.1–60.0. Default: 1.0.

    Returns:
        dict with 'predictions' list (step, timestamp_s, x_m, y_m, z_m, range_m),
        and initial + final positions.
    """
    t0 = time.perf_counter()
    n_steps = max(1, min(100, int(n_steps)))
    dt      = max(0.1, min(60.0, float(dt)))

    pos0 = np.array([[x_m], [y_m], [z_m]])
    vel0 = np.array([[vx_mps], [vy_mps], [vz_mps]])

    # Vectorised: (3,1) + (3,1) * (1,n) → (3,n) — no Python loop
    k_vec     = np.arange(1, n_steps + 1, dtype=float)
    positions = pos0 + vel0 * (k_vec * dt)   # broadcast, shape (3, n)

    predictions = []
    for k in range(n_steps):
        px, py, pz = positions[:, k]
        r = math.sqrt(float(px)**2 + float(py)**2 + float(pz)**2)
        predictions.append({
            "step":        k + 1,
            "timestamp_s": round((k + 1) * dt, 3),
            "x_m":         round(float(px), 2),
            "y_m":         round(float(py), 2),
            "z_m":         round(float(pz), 2),
            "range_m":     round(r, 2),
        })

    summary = (
        f"Predicted {n_steps} steps from ({x_m:.0f}, {y_m:.0f}, {z_m:.0f}) m "
        f"at ({vx_mps:.1f}, {vy_mps:.1f}, {vz_mps:.1f}) m/s. "
        f"Final position: ({predictions[-1]['x_m']:.0f}, "
        f"{predictions[-1]['y_m']:.0f}, {predictions[-1]['z_m']:.0f}) m "
        f"at range {predictions[-1]['range_m']:.0f} m."
    )

    return _ok("predict_future_positions", {
        "n_steps":        n_steps,
        "dt_s":           dt,
        "initial_pos_m":  [round(x_m, 2), round(y_m, 2), round(z_m, 2)],
        "velocity_mps":   [round(vx_mps, 3), round(vy_mps, 3), round(vz_mps, 3)],
        "predictions":    predictions,
    }, summary, t0)


# ── Tool 5 ── Spherical → Cartesian Conversion ────────────────────────────

def convert_spherical_to_cartesian(
    range_m: float,
    azimuth_deg: float,
    elevation_deg: float,
) -> dict:
    """
    Convert a single radar measurement from spherical to Cartesian coordinates.

    Applies the standard aeronautical spherical-to-Cartesian transform:
        X = r · cos(φ) · cos(θ)
        Y = r · cos(φ) · sin(θ)
        Z = r · sin(φ)
    Also returns the 3×3 Jacobian J_T evaluated at this geometry, which is
    used by LinearKF3D for the first-order noise covariance propagation.

    Args:
        range_m:       Slant range in metres. Must be > 0.
        azimuth_deg:   Azimuth angle in degrees. Range: -180 to 180.
        elevation_deg: Elevation angle in degrees. Range: -90 to 90.

    Returns:
        dict with x_m, y_m, z_m (Cartesian position), horizontal_range_m
        (ground range), back-converted range/az/el to verify round-trip,
        and the 3×3 Jacobian rows.
    """
    t0 = time.perf_counter()
    if range_m <= 0:
        return _err("convert_spherical_to_cartesian",
                    f"range_m must be positive, got {range_m}")

    az  = math.radians(azimuth_deg)
    el  = math.radians(elevation_deg)
    c_e = math.cos(el)
    s_e = math.sin(el)
    c_a = math.cos(az)
    s_a = math.sin(az)

    x = range_m * c_e * c_a
    y = range_m * c_e * s_a
    z = range_m * s_e

    rho  = math.sqrt(x**2 + y**2)   # horizontal / ground range
    r_bt = math.sqrt(x**2 + y**2 + z**2)  # back-computed range (should == range_m)

    # Jacobian J_T = ∂[X,Y,Z]/∂[r, θ, φ]  (3×3) — vectorised np.array
    J_T = np.array([
        [ c_e * c_a,  -range_m * c_e * s_a,  -range_m * s_e * c_a],
        [ c_e * s_a,   range_m * c_e * c_a,  -range_m * s_e * s_a],
        [ s_e,         0.0,                    range_m * c_e       ],
    ], dtype=float)

    summary = (
        f"Range={range_m:.1f}m, Az={azimuth_deg:.2f}°, El={elevation_deg:.2f}° → "
        f"X={x:.1f}m, Y={y:.1f}m, Z={z:.1f}m (ground range={rho:.1f}m)."
    )

    return _ok("convert_spherical_to_cartesian", {
        "input": {
            "range_m": round(range_m, 2),
            "azimuth_deg": round(azimuth_deg, 4),
            "elevation_deg": round(elevation_deg, 4),
        },
        "cartesian": {
            "x_m": round(x, 2),
            "y_m": round(y, 2),
            "z_m": round(z, 2),
        },
        "horizontal_range_m":     round(rho, 2),
        "back_computed_range_m":  round(r_bt, 4),
        "jacobian_J_T_rows": {
            "dX_d_r_az_el": [round(float(v), 6) for v in J_T[0]],
            "dY_d_r_az_el": [round(float(v), 6) for v in J_T[1]],
            "dZ_d_r_az_el": [round(float(v), 6) for v in J_T[2]],
        },
    }, summary, t0)


# ── Registry — imported by agent_runner.py ────────────────────────────────

# ADK registers plain callables directly; this list is the single source of truth.
ALL_TOOLS = [
    simulate_radar_scan,
    run_ekf3d_filter,
    compare_filter_performance,
    predict_future_positions,
    convert_spherical_to_cartesian,
]
