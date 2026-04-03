"""
EKF3DTracker.py  —  3-D Extended Kalman Filter with Vectorized NumPy
=====================================================================
Extends the 2-D EKF in ExtfKFTracker.py to full 3-D spherical coordinates.

State vector (6 × 1)
--------------------
  x = [X, Y, Z, Vx, Vy, Vz]^T          (Cartesian position + velocity)

Non-linear measurement function  h(x) : R^6 → R^3
---------------------------------------------------
  Radar measures in *spherical* coordinates:

         ┌  r  ┐   ┌  sqrt(X² + Y² + Z²)       ┐   3-D slant range
  h(x) = │  θ  │ = │  atan2(Y, X)               │   azimuth   [-π,  π]
         └  φ  ┘   └  atan2(Z, sqrt(X² + Y²))   ┘   elevation [-π/2, π/2]

  Let  ρ = sqrt(X² + Y²)  (ground-range / horizontal range)
       r = sqrt(ρ² + Z²)  (slant range / 3-D range)

Jacobian  Hj = ∂h/∂x  (3 × 6)  — derived analytically
------------------------------------------------------
           X     Y     Z     Vx   Vy   Vz
  ∂r :  [ X/r,  Y/r,  Z/r,   0,   0,   0  ]
  ∂θ :  [-Y/ρ², X/ρ²,  0,    0,   0,   0  ]
  ∂φ :  [-XZ/(r²ρ), -YZ/(r²ρ), ρ/r², 0, 0, 0]

All Jacobian terms are evaluated as vectorized NumPy expressions in a
single np.array() call — no Python loops.

Dead-reckoning  batch_predict(n)
---------------------------------
For a constant-velocity model the closed-form solution is:
  X(k) = X₀ + Vx·k·dt   (same for Y, Z)
  V(k) = V₀             (unchanged)

This allows vectorized prediction over n future steps without iteration:
  k_vec = np.arange(1, n+1)                    shape (n,)
  positions = pos0 + vel0 * (k_vec * dt)        shape (3, n)  — pure broadcast
  states    = np.vstack([positions, velocities]) shape (6, n)

Usage
-----
  ekf = EKF3D(dt=1.0)
  z = np.array([[r], [theta], [phi]])   # spherical measurement (3×1)
  state = ekf.step(z)                   # predict + update → (6,1)
  future = ekf.batch_predict(10)        # 10-step dead-reckoning → (6,10)

Run self-test
-------------
  python EKF3DTracker.py
"""

from __future__ import annotations

import math
import time
import platform

import numpy as np


# ---------------------------------------------------------------------------
# Helper: vectorized angle wrapping
# ---------------------------------------------------------------------------

def _wrap(angle: np.ndarray, limit: float) -> np.ndarray:
    """Wrap *angle* into (−limit, +limit] using modular arithmetic (no loop)."""
    return (angle + limit) % (2.0 * limit) - limit


# ---------------------------------------------------------------------------
# EKF3D
# ---------------------------------------------------------------------------

class EKF3D:
    """
    Extended Kalman Filter for 3-D radar target tracking.

    Coordinate frames
    -----------------
    Cartesian state  : [X, Y, Z, Vx, Vy, Vz]  metres / metres·s⁻¹
    Spherical meas.  : [range r, azimuth θ, elevation φ]  m / rad / rad

    Parameters
    ----------
    dt : float
        Filter time step (seconds).  Must match the radar's scan interval.
    noise_range : float
        1-σ measurement noise on slant range (metres).
    noise_az : float
        1-σ measurement noise on azimuth (radians).
    noise_el : float
        1-σ measurement noise on elevation (radians).
    process_noise_scale : float
        Diagonal scale for the process noise matrix Q.  Increase for more
        agile / manoeuvring targets.
    """

    def __init__(
        self,
        dt: float = 1.0,
        noise_range: float = 50.0,
        noise_az: float   = 0.02,
        noise_el: float   = 0.01,
        process_noise_scale: float = 0.5,
    ) -> None:
        self.dt = dt
        self._n = 6   # state dimension
        self._m = 3   # measurement dimension

        # ── 1. State vector [X, Y, Z, Vx, Vy, Vz]^T ─────────────────────
        # Initialised to zero; caller should call set_initial_state() before
        # the first update, or rely on the filter to converge from high P.
        self.x: np.ndarray = np.zeros((self._n, 1))

        # ── 2. State covariance P (high initial uncertainty) ──────────────
        self.P: np.ndarray = np.eye(self._n) * 1000.0

        # ── 3. State transition matrix F  (constant-velocity model) ───────
        self.F: np.ndarray = self._build_F(dt)

        # ── 4. Measurement noise covariance R  (3×3 diagonal) ─────────────
        self.R: np.ndarray = np.diag([noise_range**2, noise_az**2, noise_el**2])

        # ── 5. Process noise covariance Q ─────────────────────────────────
        self.Q: np.ndarray = np.eye(self._n) * process_noise_scale

        # Identity cached for covariance update
        self._I: np.ndarray = np.eye(self._n)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def set_initial_state(self, x0: np.ndarray, P0: np.ndarray | None = None) -> None:
        """
        Set the initial state estimate and (optionally) its covariance.

        Parameters
        ----------
        x0 : array-like, shape (6,) or (6,1)
            [X, Y, Z, Vx, Vy, Vz] initial estimate.
        P0 : array-like, shape (6,6), optional
            Initial covariance.  Defaults to eye(6) × 1000 if omitted.
        """
        self.x = np.asarray(x0, dtype=float).reshape(self._n, 1)
        if P0 is not None:
            self.P = np.asarray(P0, dtype=float)

    # ------------------------------------------------------------------
    # Core EKF steps
    # ------------------------------------------------------------------

    def predict(self) -> np.ndarray:
        """
        EKF predict step: propagate state and covariance by one time step.

        Returns
        -------
        x_pred : np.ndarray, shape (6,1)
            A priori state estimate  x⁻ = F·x
        """
        self.x = self.F @ self.x                        # (6,1)
        self.P = self.F @ self.P @ self.F.T + self.Q   # (6,6)
        return self.x

    def update(self, z: np.ndarray) -> np.ndarray:
        """
        EKF update step with spherical-coordinate measurement.

        Parameters
        ----------
        z : np.ndarray, shape (3,1)
            Measurement vector [range, azimuth, elevation].

        Returns
        -------
        x_upd : np.ndarray, shape (6,1)
            A posteriori state estimate.

        Raises
        ------
        RuntimeError
            If the target is within 1 m of the radar origin (singularity).
        """
        z = np.asarray(z, dtype=float).reshape(self._m, 1)

        # Predicted measurement from current state estimate
        h_x = self._h(self.x)            # (3,1)
        Hj  = self._jacobian_H(self.x)   # (3,6)  — vectorized, no loops

        # Innovation (residual)
        y = z - h_x                      # (3,1)

        # Wrap angle innovations to prevent discontinuity errors at ±π / ±π/2
        y[1, 0] = _wrap(y[1, 0], math.pi)         # azimuth   ∈ (−π,  π]
        y[2, 0] = _wrap(y[2, 0], math.pi / 2.0)   # elevation ∈ (−π/2, π/2]

        # Innovation covariance S and Kalman gain K
        S = Hj @ self.P @ Hj.T + self.R           # (3,3)
        K = self.P @ Hj.T @ np.linalg.inv(S)      # (6,3)

        # State and covariance update
        self.x = self.x + K @ y                   # (6,1)
        self.P = (self._I - K @ Hj) @ self.P      # (6,6)

        return self.x

    def step(self, z: np.ndarray) -> np.ndarray:
        """Convenience: predict then update in one call."""
        self.predict()
        return self.update(z)

    # ------------------------------------------------------------------
    # Vectorized batch dead-reckoning
    # ------------------------------------------------------------------

    def batch_predict(self, n_steps: int) -> np.ndarray:
        """
        Propagate the *current* state forward n_steps using the closed-form
        constant-velocity solution — no Python loop, pure NumPy broadcast.

        For constant velocity:  pos(k) = pos₀ + vel₀ · k · dt
                                vel(k) = vel₀   (constant)

        Parameters
        ----------
        n_steps : int
            Number of future steps to predict.

        Returns
        -------
        states : np.ndarray, shape (6, n_steps)
            Each column is the predicted state at step k = 1 … n_steps.
            Row order: [X, Y, Z, Vx, Vy, Vz].
        """
        k_vec = np.arange(1, n_steps + 1, dtype=float)  # (n,)

        pos0 = self.x[:3]   # (3,1)  current position
        vel0 = self.x[3:]   # (3,1)  current velocity

        # Vectorized: (3,1) + (3,1)·(1,n)  →  (3,n)
        positions  = pos0 + vel0 * (k_vec * self.dt)     # broadcast
        velocities = np.repeat(vel0, n_steps, axis=1)    # (3,n)

        return np.vstack([positions, velocities])         # (6,n)

    # ------------------------------------------------------------------
    # Non-linear measurement function h(x) and its Jacobian
    # ------------------------------------------------------------------

    def _h(self, x: np.ndarray) -> np.ndarray:
        """
        Non-linear measurement function:  Cartesian state → spherical coords.

          h(x) = [ sqrt(X²+Y²+Z²),  atan2(Y,X),  atan2(Z, sqrt(X²+Y²)) ]^T

        Parameters
        ----------
        x : np.ndarray, shape (6,1)

        Returns
        -------
        z_pred : np.ndarray, shape (3,1)
        """
        px, py, pz = x[0, 0], x[1, 0], x[2, 0]

        rho = math.sqrt(px**2 + py**2)          # ground / horizontal range
        r   = math.sqrt(rho**2 + pz**2)         # 3-D slant range

        if r < 1.0:
            raise RuntimeError(
                f"Target at ({px:.1f},{py:.1f},{pz:.1f}) is within 1 m of "
                "radar origin — Jacobian singularity."
            )

        return np.array([
            [r],
            [math.atan2(py, px)],       # azimuth
            [math.atan2(pz, rho)],      # elevation
        ])

    def _jacobian_H(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the 3×6 Jacobian of h evaluated at state *x*.

        All partial derivatives are assembled in a single vectorized
        np.array() call — no element-wise assignment loops.

        Derivation
        ----------
        Let  ρ  = sqrt(X²+Y²)          (horizontal range)
             r  = sqrt(X²+Y²+Z²)       (slant range)
             ρ² = X²+Y²
             r² = X²+Y²+Z²

          ∂r/∂[X,Y,Z]   = [X/r,  Y/r,  Z/r]

          ∂θ/∂[X,Y,Z]   = [-Y/ρ²,  X/ρ²,  0]

          ∂φ/∂X  = ∂/∂X [atan2(Z,ρ)]
                 = (∂atan2/∂ρ)·(∂ρ/∂X)
                 = (−Z/r²)·(X/ρ)  =  −XZ/(r²ρ)

          ∂φ/∂Y  = −YZ/(r²ρ)   (by symmetry)

          ∂φ/∂Z  = ∂/∂Z [atan2(Z,ρ)]
                 = ρ/(Z²+ρ²)  =  ρ/r²

        Velocity columns are always zero (measurement is position-only).

        Parameters
        ----------
        x : np.ndarray, shape (6,1)

        Returns
        -------
        Hj : np.ndarray, shape (3,6)
        """
        px, py, pz = x[0, 0], x[1, 0], x[2, 0]

        rho_sq = px**2 + py**2
        rho    = math.sqrt(rho_sq)
        r_sq   = rho_sq + pz**2
        r      = math.sqrt(r_sq)

        # Guard against degenerate geometry
        if r < 1e-6:
            return np.zeros((self._m, self._n))
        if rho < 1e-6:
            # Target directly above/below radar — azimuth Jacobian undefined
            # Return zero rows for θ and φ (filter skips the angular update)
            Hj = np.zeros((self._m, self._n))
            Hj[0, :3] = [px / r, py / r, pz / r]
            return Hj

        # ── Single vectorized np.array() call ────────────────────────────
        #
        #        X              Y              Z         Vx  Vy  Vz
        Hj = np.array([
            [ px / r,         py / r,         pz / r,     0., 0., 0.],  # ∂range
            [-py / rho_sq,    px / rho_sq,    0.,          0., 0., 0.],  # ∂azimuth
            [-px*pz/(r_sq*rho), -py*pz/(r_sq*rho), rho/r_sq, 0., 0., 0.],  # ∂elevation
        ], dtype=float)

        return Hj   # shape (3, 6)

    # ------------------------------------------------------------------
    # Internal constructor helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_F(dt: float) -> np.ndarray:
        """
        Build the 6×6 state transition matrix for a constant-velocity model.

          ┌ I₃  dt·I₃ ┐
          └ 0₃    I₃  ┘

        where I₃ is the 3×3 identity and 0₃ is the 3×3 zero matrix.
        Constructed with a single vectorized block-assignment.
        """
        F = np.eye(6, dtype=float)
        F[:3, 3:] = np.eye(3) * dt   # position += velocity × dt
        return F

    # ------------------------------------------------------------------
    # Properties: read-only views of current estimate
    # ------------------------------------------------------------------

    @property
    def position(self) -> np.ndarray:
        """Current position estimate [X, Y, Z] in metres, shape (3,)."""
        return self.x[:3, 0]

    @property
    def velocity(self) -> np.ndarray:
        """Current velocity estimate [Vx, Vy, Vz] in m/s, shape (3,)."""
        return self.x[3:, 0]

    @property
    def position_uncertainty(self) -> np.ndarray:
        """1-σ position uncertainty [σ_X, σ_Y, σ_Z] (sqrt of diag of P[:3,:3])."""
        return np.sqrt(np.diag(self.P[:3, :3]))


# ---------------------------------------------------------------------------
# Inline self-test  (python EKF3DTracker.py)
# ---------------------------------------------------------------------------

def _generate_3d_telemetry(
    n_steps: int = 80,
    dt: float = 1.0,
    noise_r: float = 50.0,
    noise_az: float = 0.02,
    noise_el: float = 0.01,
    rng_seed: int = 42,
) -> list[dict]:
    """
    Produce synthetic 3-D radar telemetry without external dependencies.

    True trajectory: constant-velocity straight-line approach.
      Initial position  : (10 000, 5 000, 3 000) m
      Initial velocity  : (−200, −50, −10) m/s
    """
    rng = np.random.default_rng(rng_seed)

    state = np.array([10_000.0, 5_000.0, 3_000.0, -200.0, -50.0, -10.0])
    F = np.eye(6)
    F[:3, 3:] = np.eye(3) * dt

    records = []
    for t in range(n_steps):
        state = F @ state
        px, py, pz = state[:3]

        rho  = math.sqrt(px**2 + py**2)
        r    = math.sqrt(rho**2 + pz**2)
        az   = math.atan2(py, px)
        el   = math.atan2(pz, rho)

        records.append({
            "t":       t * dt,
            "r":       r   + rng.normal(0, noise_r),
            "az":      az  + rng.normal(0, noise_az),
            "el":      el  + rng.normal(0, noise_el),
            "true_x":  px,
            "true_y":  py,
            "true_z":  pz,
        })
    return records


def _run_self_test() -> None:
    print("=" * 60)
    print(" EKF3DTracker — Self-Test")
    print("=" * 60)
    print(f" Platform : {platform.system()} {platform.release()}")
    print(f" CPU      : {platform.processor()}")
    print("-" * 60)

    dt       = 1.0
    n_steps  = 80
    data     = _generate_3d_telemetry(n_steps=n_steps, dt=dt)

    # ── Instantiate filter ────────────────────────────────────────────
    ekf = EKF3D(
        dt=dt,
        noise_range=50.0,
        noise_az=0.02,
        noise_el=0.01,
        process_noise_scale=0.5,
    )

    # Intentionally rough initial guess  (≈ 1 km off in each axis)
    ekf.set_initial_state(
        x0=[9_000.0, 4_200.0, 2_500.0, 0.0, 0.0, 0.0],
    )

    raw_sq_errors = []
    ekf_sq_errors = []
    latencies_us  = []
    convergence_step: int | None = None    # first step EKF error < 100 m

    # Steady-state window = second half of the run, used for RMSE after convergence
    steady_start = n_steps // 2

    print(f"{'Step':>4}  {'True (X,Y,Z)':>28}  {'EKF (X,Y,Z)':>28}  "
          f"{'EKF_err':>8}  {'Raw_err':>8}")
    print("-" * 90)

    for idx, rec in enumerate(data):
        z = np.array([[rec["r"]], [rec["az"]], [rec["el"]]])

        t0 = time.perf_counter()
        try:
            ekf.step(z)
        except RuntimeError as exc:
            print(f"[WARN] {exc}")
            continue
        t1 = time.perf_counter()

        latencies_us.append((t1 - t0) * 1e6)

        # Errors
        tx, ty, tz = rec["true_x"], rec["true_y"], rec["true_z"]
        ex, ey, ez = ekf.position

        # Spherical → Cartesian conversion for raw measurement error
        cos_el = math.cos(rec["el"])
        raw_x  = rec["r"] * cos_el * math.cos(rec["az"])
        raw_y  = rec["r"] * cos_el * math.sin(rec["az"])
        raw_z  = rec["r"] * math.sin(rec["el"])

        raw_sq = (raw_x - tx)**2 + (raw_y - ty)**2 + (raw_z - tz)**2
        ekf_sq = (ex    - tx)**2 + (ey    - ty)**2 + (ez    - tz)**2

        raw_sq_errors.append(raw_sq)
        ekf_sq_errors.append(ekf_sq)

        # Detect first convergence (EKF 3-D error drops below 100 m)
        if convergence_step is None and math.sqrt(ekf_sq) < 100.0:
            convergence_step = idx

        step = int(rec["t"])
        if step % 10 == 0 or step < 3:
            print(
                f"{step:4.0f}  "
                f"({tx:8.0f},{ty:7.0f},{tz:7.0f})  "
                f"({ex:8.1f},{ey:7.1f},{ez:7.1f})  "
                f"{math.sqrt(ekf_sq):8.1f}m  "
                f"{math.sqrt(raw_sq):8.1f}m"
            )

    # ── Batch dead-reckoning demo ─────────────────────────────────────
    print()
    print("── Batch dead-reckoning (next 5 steps from final state, no loop) ──")
    future = ekf.batch_predict(5)   # (6,5) — closed-form broadcast, no Python loop
    for k in range(5):
        pos = future[:3, k]
        print(f"  +{k+1}s  X={pos[0]:9.1f}  Y={pos[1]:9.1f}  Z={pos[2]:9.1f}")

    # ── Summary ───────────────────────────────────────────────────────
    n          = len(raw_sq_errors)
    raw_rmse   = math.sqrt(sum(raw_sq_errors) / n)
    ekf_rmse   = math.sqrt(sum(ekf_sq_errors) / n)

    # Steady-state RMSE: only steps after the convergence window
    ss_raw  = raw_sq_errors[steady_start:]
    ss_ekf  = ekf_sq_errors[steady_start:]
    ss_raw_rmse = math.sqrt(sum(ss_raw) / len(ss_raw))
    ss_ekf_rmse = math.sqrt(sum(ss_ekf) / len(ss_ekf))
    ss_reduction = (ss_raw_rmse - ss_ekf_rmse) / ss_raw_rmse * 100

    avg_us  = sum(latencies_us) / len(latencies_us)
    max_hz  = 1e6 / avg_us

    print()
    print("=" * 60)
    print("=== PERFORMANCE SUMMARY ===")
    print("=" * 60)
    print(f"  Steps processed    : {n}")
    print(f"  Convergence at     : step {convergence_step if convergence_step is not None else 'N/A'}")
    print()
    print(f"  ── Full-run RMSE (includes transient) ──")
    print(f"  Raw sensor RMSE    : {raw_rmse:>10.2f} m")
    print(f"  EKF filter RMSE    : {ekf_rmse:>10.2f} m")
    print()
    print(f"  ── Steady-state RMSE (steps {steady_start}–{n}) ──")
    print(f"  Raw sensor RMSE    : {ss_raw_rmse:>10.2f} m")
    print(f"  EKF filter RMSE    : {ss_ekf_rmse:>10.2f} m")
    print(f"  Noise reduction    : {ss_reduction:>10.1f} %")
    print()
    print(f"  Avg step latency   : {avg_us:>10.3f} µs")
    print(f"  Max throughput     : {max_hz:>10,.0f} Hz")
    print("=" * 60)

    # ── Position uncertainty at end ───────────────────────────────────
    sig = ekf.position_uncertainty
    print(f"  Final σ [X,Y,Z]    : [{sig[0]:.2f}, {sig[1]:.2f}, {sig[2]:.2f}] m")
    print("=" * 60)


if __name__ == "__main__":
    _run_self_test()
