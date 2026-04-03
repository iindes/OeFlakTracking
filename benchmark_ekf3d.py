"""
benchmark_ekf3d.py  —  EKF3D Latency & Accuracy Benchmark
==========================================================
Two independent benchmarks:

  1. LATENCY — predict + update loop  (N = 10 000 cycles, 200 warm-up)
     ┌─────────────────────────────────────────────────────────┐
     │ Target: mean cycle time < 0.200 ms (200 µs)            │
     │ Reports: mean, std, min, p50, p95, p99, max            │
     └─────────────────────────────────────────────────────────┘

  2. ACCURACY — RMSE on an 80-step 3-D constant-velocity trajectory
     Filters compared:
       • Raw        — spherical→Cartesian conversion only (no filtering)
       • LinearKF3D — standard converted-measurement linear Kalman Filter
                      Uses a static, first-order propagated R_cart and a
                      linear measurement matrix H = [I₃ | 0₃]
       • EKF3D      — non-linear spherical EKF with analytic Jacobian
     Reports:
       • Full-run RMSE (includes transient / convergence period)
       • Steady-state RMSE  (second half of steps, post-convergence)
       • Noise reduction %  vs raw sensor
       • EKF3D improvement % vs LinearKF3D (steady-state)

LinearKF3D vs EKF3D — key difference
--------------------------------------
LinearKF3D converts z_sph → z_cart (trig) then applies the linear KF with a
*fixed* Cartesian noise covariance R_cart computed once via first-order error
propagation at the initial range/angle geometry:

    R_cart ≈ J_T · R_sph · J_T^T
    where J_T = ∂[Cartesian] / ∂[spherical]  at (r₀, θ₀, φ₀)

Because R_cart is frozen at initialization, it stops being accurate as the
target moves (range and angles change), causing residual modeling error.
EKF3D avoids this by recomputing the Jacobian Hj at *every* step, so the
linearization tracks the current geometry continuously.

Usage
-----
  python benchmark_ekf3d.py
"""

from __future__ import annotations

import math
import time
import platform

import numpy as np

from EKF3DTracker import EKF3D, _generate_3d_telemetry


# ---------------------------------------------------------------------------
# LinearKF3D  —  Converted-Measurement Linear Kalman Filter
# ---------------------------------------------------------------------------

class LinearKF3D:
    """
    Standard (linear) Kalman Filter for 3-D radar tracking.

    Reception model
    ---------------
    Radar measurements arrive as spherical coords [r, θ, φ].  Before each
    update, they are converted to Cartesian [X, Y, Z] via:

        X = r · cos(φ) · cos(θ)
        Y = r · cos(φ) · sin(θ)
        Z = r · sin(φ)

    The measurement equation is then *linear*:
        z_cart = H · x + v,   H = [I₃ | 0₃]  (observe position only)

    Noise model
    -----------
    The Cartesian noise covariance R_cart is approximated once at construction
    by first-order error propagation from the spherical noise parameters:

        R_cart ≈ J_T · diag(σ_r², σ_θ², σ_φ²) · J_T^T

    where J_T = ∂(Cartesian) / ∂(spherical) evaluated at the reference
    geometry (ref_r, ref_az, ref_el).  This approximation becomes inaccurate
    as the target moves — exactly the weakness exploited by the EKF.

    Parameters
    ----------
    dt, noise_range, noise_az, noise_el, process_noise_scale
        Same semantics as EKF3D.
    ref_r, ref_az, ref_el
        Reference geometry used to compute R_cart once at init.
        Should be set to the expected first-measurement geometry.
    """

    def __init__(
        self,
        dt: float = 1.0,
        noise_range: float = 50.0,
        noise_az: float   = 0.02,
        noise_el: float   = 0.01,
        process_noise_scale: float = 0.5,
        ref_r:  float = 10_000.0,
        ref_az: float = 0.464,    # atan2(5000, 10000) ≈ 26.6°
        ref_el: float = 0.291,    # atan2(3000, ~11180) ≈ 15°
    ) -> None:
        self.dt = dt
        n, m = 6, 3

        # State and covariance
        self.x = np.zeros((n, 1))
        self.P = np.eye(n) * 1000.0

        # State transition (constant velocity)
        self.F = np.eye(n, dtype=float)
        self.F[:3, 3:] = np.eye(3) * dt

        # Linear measurement matrix  H = [I₃ | 0₃]  (3×6)
        self.H = np.zeros((m, n), dtype=float)
        self.H[:3, :3] = np.eye(3)

        # Process noise
        self.Q = np.eye(n) * process_noise_scale
        self._I = np.eye(n)

        # Cartesian measurement noise — first-order propagation (computed once)
        self.R = self._propagate_R(
            ref_r, ref_az, ref_el, noise_range, noise_az, noise_el
        )

    # ------------------------------------------------------------------

    def set_initial_state(self, x0, P0=None):
        self.x = np.asarray(x0, dtype=float).reshape(6, 1)
        if P0 is not None:
            self.P = np.asarray(P0, dtype=float)

    # ------------------------------------------------------------------
    # Core steps
    # ------------------------------------------------------------------

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z_spherical: np.ndarray) -> np.ndarray:
        """
        Convert spherical measurement to Cartesian, then apply linear update.

        Parameters
        ----------
        z_spherical : array-like, shape (3,1)  [range, azimuth, elevation]
        """
        r, az, el = np.asarray(z_spherical, dtype=float).ravel()

        # Spherical → Cartesian conversion (non-linear, applied to raw measurement)
        cos_el = math.cos(el)
        z_cart = np.array([
            [r * cos_el * math.cos(az)],
            [r * cos_el * math.sin(az)],
            [r * math.sin(el)],
        ])

        # Linear KF update with H = [I₃ | 0₃]
        y = z_cart - self.H @ self.x                   # innovation  (3,1)
        S = self.H @ self.P @ self.H.T + self.R        # innov. cov  (3,3)
        K = self.P @ self.H.T @ np.linalg.inv(S)       # gain        (6,3)
        self.x = self.x + K @ y                        # state update
        self.P = (self._I - K @ self.H) @ self.P       # cov update
        return self.x

    def step(self, z_spherical: np.ndarray) -> np.ndarray:
        self.predict()
        return self.update(z_spherical)

    @property
    def position(self) -> np.ndarray:
        return self.x[:3, 0]

    # ------------------------------------------------------------------
    # R propagation
    # ------------------------------------------------------------------

    @staticmethod
    def _propagate_R(
        r: float, az: float, el: float,
        sig_r: float, sig_az: float, sig_el: float,
    ) -> np.ndarray:
        """
        First-order error propagation: R_sph → R_cart.

        J_T = ∂[X,Y,Z] / ∂[r, θ, φ]  (3×3)

            ∂X/∂r  = cos(φ)·cos(θ)       ∂X/∂θ = -r·cos(φ)·sin(θ)   ∂X/∂φ = -r·sin(φ)·cos(θ)
            ∂Y/∂r  = cos(φ)·sin(θ)       ∂Y/∂θ =  r·cos(φ)·cos(θ)   ∂Y/∂φ = -r·sin(φ)·sin(θ)
            ∂Z/∂r  = sin(φ)              ∂Z/∂θ =  0                  ∂Z/∂φ =  r·cos(φ)

        R_cart ≈ J_T · diag(σ_r², σ_θ², σ_φ²) · J_T^T
        """
        c_el, s_el = math.cos(el), math.sin(el)
        c_az, s_az = math.cos(az), math.sin(az)

        # 3×3 Jacobian of the spherical→Cartesian transform
        J_T = np.array([
            [ c_el * c_az,   -r * c_el * s_az,   -r * s_el * c_az],
            [ c_el * s_az,    r * c_el * c_az,   -r * s_el * s_az],
            [ s_el,           0.0,                r * c_el        ],
        ], dtype=float)

        R_sph = np.diag([sig_r**2, sig_az**2, sig_el**2])
        return J_T @ R_sph @ J_T.T   # (3,3)


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def _percentiles(arr: np.ndarray, qs) -> np.ndarray:
    """Return percentiles without scipy dependency."""
    return np.percentile(arr, qs)


def _fmt_us(v: float) -> str:
    return f"{v:9.3f} µs"


def _pass_fail(mean_us: float, target_us: float = 200.0) -> str:
    return "PASS ✓" if mean_us < target_us else "FAIL ✗"


# ---------------------------------------------------------------------------
# Benchmark 1: Latency
# ---------------------------------------------------------------------------

def run_latency_benchmark(
    n_warmup: int = 200,
    n_measure: int = 10_000,
    dt: float = 1.0,
    target_us: float = 200.0,
) -> dict:
    """
    Time EKF3D and LinearKF3D predict+update cycles.

    Methodology
    -----------
    - Filters are initialized once, then a fixed batch of n_measure random
      spherical measurements is pre-generated (off the timed path).
    - n_warmup cycles are run first (discarded) to warm CPU caches and
      trigger JIT compilation of NumPy internals.
    - Each of the n_measure cycles is individually timed with
      time.perf_counter() to capture tail latencies.
    - Statistics: mean, std, min, p50, p95, p99, max (all µs).
    """
    rng = np.random.default_rng(0)

    # Pre-generate all measurements — outside the timed loop
    total = n_warmup + n_measure
    r_vals  = rng.uniform(2_000, 12_000, total)
    az_vals = rng.uniform(-math.pi, math.pi, total)
    el_vals = rng.uniform(-math.pi / 4, math.pi / 4, total)
    meas = [
        np.array([[r_vals[i]], [az_vals[i]], [el_vals[i]]])
        for i in range(total)
    ]

    results = {}

    for name, FilterClass, kwargs in [
        ("EKF3D",      EKF3D,       {}),
        ("LinearKF3D", LinearKF3D,  {}),
    ]:
        filt = FilterClass(dt=dt, **kwargs)
        filt.set_initial_state([9_000, 4_200, 2_500, 0, 0, 0])

        # Warm-up (discarded)
        for i in range(n_warmup):
            try:
                filt.step(meas[i])
            except RuntimeError:
                pass

        # Measured runs
        latencies = np.empty(n_measure)
        for i in range(n_measure):
            z = meas[n_warmup + i]
            t0 = time.perf_counter()
            try:
                filt.step(z)
            except RuntimeError:
                pass
            t1 = time.perf_counter()
            latencies[i] = (t1 - t0) * 1e6  # → µs

        p50, p95, p99 = _percentiles(latencies, [50, 95, 99])
        results[name] = {
            "mean":  latencies.mean(),
            "std":   latencies.std(),
            "min":   latencies.min(),
            "p50":   p50,
            "p95":   p95,
            "p99":   p99,
            "max":   latencies.max(),
            "pass":  latencies.mean() < target_us,
        }

    return results


# ---------------------------------------------------------------------------
# Benchmark 2: RMSE Accuracy
# ---------------------------------------------------------------------------

def run_accuracy_benchmark(
    n_steps: int = 80,
    dt: float = 1.0,
    rng_seed: int = 42,
) -> dict:
    """
    Compare full-run and steady-state RMSE for:
      Raw (no filter), LinearKF3D, EKF3D
    on an identical 80-step 3-D constant-velocity trajectory.
    """
    data = _generate_3d_telemetry(n_steps=n_steps, dt=dt, rng_seed=rng_seed)

    # Reference geometry for LinearKF3D R_cart computation
    # Use the first true position to get reference angles
    first = data[0]
    ref_r  = math.sqrt(first["true_x"]**2 + first["true_y"]**2 + first["true_z"]**2)
    ref_az = math.atan2(first["true_y"], first["true_x"])
    ref_el = math.atan2(first["true_z"],
                        math.sqrt(first["true_x"]**2 + first["true_y"]**2))

    # Instantiate both filters with identical parameters
    ekf = EKF3D(dt=dt, noise_range=50.0, noise_az=0.02, noise_el=0.01,
                process_noise_scale=0.5)
    ekf.set_initial_state([9_000, 4_200, 2_500, 0, 0, 0])

    lkf = LinearKF3D(dt=dt, noise_range=50.0, noise_az=0.02, noise_el=0.01,
                     process_noise_scale=0.5,
                     ref_r=ref_r, ref_az=ref_az, ref_el=ref_el)
    lkf.set_initial_state([9_000, 4_200, 2_500, 0, 0, 0])

    raw_sq, lkf_sq, ekf_sq = [], [], []
    steady_start = n_steps // 2   # steps 40–79 = steady-state window

    for rec in data:
        z = np.array([[rec["r"]], [rec["az"]], [rec["el"]]])

        # Run both filters
        try:
            ekf.step(z)
        except RuntimeError:
            continue
        lkf.step(z)

        tx, ty, tz = rec["true_x"], rec["true_y"], rec["true_z"]

        # Raw error: spherical → Cartesian, no filtering
        cos_el = math.cos(rec["el"])
        rx = rec["r"] * cos_el * math.cos(rec["az"])
        ry = rec["r"] * cos_el * math.sin(rec["az"])
        rz = rec["r"] * math.sin(rec["el"])
        raw_sq.append((rx-tx)**2 + (ry-ty)**2 + (rz-tz)**2)

        # Linear KF error
        ex, ey, ez = lkf.position
        lkf_sq.append((ex-tx)**2 + (ey-ty)**2 + (ez-tz)**2)

        # EKF error
        ex, ey, ez = ekf.position
        ekf_sq.append((ex-tx)**2 + (ey-ty)**2 + (ez-tz)**2)

    def rmse(sq):
        return math.sqrt(sum(sq) / len(sq))

    def noise_reduction(signal_sq, ref_sq):
        r_ref = math.sqrt(sum(ref_sq) / len(ref_sq))
        r_sig = math.sqrt(sum(signal_sq) / len(signal_sq))
        return (r_ref - r_sig) / r_ref * 100.0

    ss_raw = raw_sq[steady_start:]
    ss_lkf = lkf_sq[steady_start:]
    ss_ekf = ekf_sq[steady_start:]

    return {
        "n_steps":      n_steps,
        "steady_start": steady_start,
        # Full-run RMSE
        "full": {
            "raw":  rmse(raw_sq),
            "lkf":  rmse(lkf_sq),
            "ekf":  rmse(ekf_sq),
        },
        # Steady-state RMSE
        "steady": {
            "raw":           rmse(ss_raw),
            "lkf":           rmse(ss_lkf),
            "ekf":           rmse(ss_ekf),
            "lkf_reduction": noise_reduction(ss_lkf, ss_raw),
            "ekf_reduction": noise_reduction(ss_ekf, ss_raw),
            "ekf_vs_lkf":    (rmse(ss_lkf) - rmse(ss_ekf)) / rmse(ss_lkf) * 100.0,
        },
        # Per-step detail for the table
        "steps": [
            {
                "t":        rec["t"],
                "true_x":   rec["true_x"],
                "true_y":   rec["true_y"],
                "true_z":   rec["true_z"],
                "raw_err":  math.sqrt(raw_sq[i]),
                "lkf_err":  math.sqrt(lkf_sq[i]),
                "ekf_err":  math.sqrt(ekf_sq[i]),
            }
            for i, rec in enumerate(data)
            if int(rec["t"]) % 10 == 0 or int(rec["t"]) < 3
        ],
    }


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def print_latency_report(results: dict, target_us: float = 200.0) -> None:
    W = 66
    print("=" * W)
    print(" BENCHMARK 1 — Predict + Update Cycle Latency")
    print(f" Target: mean < {target_us:.0f} µs   N = 10 000 cycles  (200 warm-up)")
    print("=" * W)
    print(f"{'Metric':<12}", end="")
    for name in results:
        print(f"  {name:>18}", end="")
    print()
    print("-" * W)

    metrics = [
        ("mean",  "Mean"),
        ("std",   "Std dev"),
        ("min",   "Min"),
        ("p50",   "p50 (median)"),
        ("p95",   "p95"),
        ("p99",   "p99"),
        ("max",   "Max"),
    ]
    for key, label in metrics:
        print(f"{label:<14}", end="")
        for name, r in results.items():
            print(f"  {_fmt_us(r[key]):>18}", end="")
        print()

    print("-" * W)
    print(f"{'vs target':<14}", end="")
    for name, r in results.items():
        status = _pass_fail(r["mean"], target_us)
        pct    = r["mean"] / target_us * 100
        print(f"  {f'{pct:.1f}% of budget  {status}':>18}", end="")
    print()
    print("=" * W)


def print_accuracy_report(acc: dict) -> None:
    W = 72
    print()
    print("=" * W)
    print(" BENCHMARK 2 — RMSE Accuracy vs Standard Linear Kalman Filter")
    print(f" Trajectory: {acc['n_steps']} steps, dt=1s, σ_r=50m, σ_θ=0.02rad, σ_φ=0.01rad")
    print(f" Initial guess offset: ~1 000 m from truth (rough init)")
    print(f" Steady-state window : steps {acc['steady_start']}–{acc['n_steps']-1}")
    print("=" * W)

    # Per-step table
    hdr = f"{'Step':>4}  {'True (X,Y,Z)':>28}  {'Raw':>8}  {'LinKF':>8}  {'EKF3D':>8}"
    print(hdr)
    print("-" * W)
    for s in acc["steps"]:
        t  = int(s["t"])
        tx = int(s["true_x"]); ty = int(s["true_y"]); tz = int(s["true_z"])
        print(
            f"{t:4d}  ({tx:8d},{ty:7d},{tz:7d})"
            f"  {s['raw_err']:8.1f}m  {s['lkf_err']:8.1f}m  {s['ekf_err']:8.1f}m"
        )

    # Summary tables
    print()
    full   = acc["full"]
    steady = acc["steady"]
    ss0    = acc["steady_start"]
    n      = acc["n_steps"]

    print(f"{'─'*W}")
    print(f"{'Filter':<14}  {'Full-run RMSE':>15}  {'Steady-state RMSE':>18}  "
          f"{'SS noise red.':>13}  {'SS vs LinKF':>11}")
    print(f"{'─'*W}")

    rows = [
        ("Raw",      full["raw"], steady["raw"],  None,                      None),
        ("LinearKF", full["lkf"], steady["lkf"],  steady["lkf_reduction"],   None),
        ("EKF3D",    full["ekf"], steady["ekf"],  steady["ekf_reduction"],   steady["ekf_vs_lkf"]),
    ]
    for label, fr, ss, red, vs in rows:
        red_s = f"{red:+.1f}%" if red is not None else "      —"
        vs_s  = f"{vs:+.1f}%" if vs  is not None else "         —"
        print(
            f"{label:<14}  {fr:>12.2f} m  {ss:>15.2f} m"
            f"  {red_s:>13}  {vs_s:>11}"
        )

    print(f"{'─'*W}")
    print()
    print(f"  EKF3D vs LinearKF steady-state improvement : "
          f"{steady['ekf_vs_lkf']:+.1f}% lower RMSE")
    print(
        "  Explanation: LinearKF computes R_cart once at init from first-order\n"
        "  error propagation. As range decreases the approximation degrades.\n"
        "  EKF3D re-linearises the Jacobian Hj every step, tracking the current\n"
        "  geometry — no frozen-R modeling error."
    )
    print("=" * W)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    W = 66
    print("=" * W)
    print(" EKF3DTracker — Benchmark Suite")
    print("=" * W)
    print(f" Platform  : {platform.system()} {platform.release()}")
    print(f" CPU       : {platform.processor()}")
    print(f" NumPy     : {np.__version__}")
    print("=" * W)
    print()

    print("Running latency benchmark …  (10 000 cycles each, ~2 s)")
    lat = run_latency_benchmark(n_warmup=200, n_measure=10_000, target_us=200.0)
    print_latency_report(lat, target_us=200.0)

    print()
    print("Running accuracy benchmark …  (80 steps, seed=42)")
    acc = run_accuracy_benchmark(n_steps=80, dt=1.0, rng_seed=42)
    print_accuracy_report(acc)


if __name__ == "__main__":
    main()
