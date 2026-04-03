"""
ExtfKFTracker_v2.py  —  EKF State-Estimation Engine (v2)
=========================================================
Copied from ExtfKFTracker.py; changes:
  - Connects to the *ingestion pipeline* via a PULL socket (port 5556)
    instead of subscribing directly to the simulator's PUB socket.
    This achieves full decoupling: the EKF knows nothing about how or
    where telemetry originates.
  - Parses enriched JSON produced by ingestion_pipeline.py (no topic-prefix
    stripping needed; JSON is sent directly via push.send_json).
  - Reports *pipeline latency* (time from ingestion to EKF processing) in
    addition to EKF processing latency.
  - Detects sequence gaps in 'pipeline_seq' to surface dropped messages.
"""

import numpy as np
import csv
import math


class RadarEKF:
    def __init__(self, dt=1.0):
        """
        Initialize the Extended Kalman Filter for radar tracking.
        """
        self.dt = dt

        # 1. State Vector [X, Y, Vx, Vy]^T
        # We start with a rough guess. Let's say we assume it's at X=9000, Y=4000 moving 0m/s.
        self.x = np.array([[9000.0], [4000.0], [0.0], [0.0]])

        # 2. State Covariance Matrix (P)
        # We don't trust our initial guess at all, so we set uncertainty very high (1000).
        self.P = np.eye(4) * 1000.0

        # 3. State Transition Matrix (F) - Linear Physics (Constant Velocity)
        self.F = np.array([
            [1.0, 0.0, self.dt, 0.0],
            [0.0, 1.0, 0.0, self.dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        # 4. Measurement Noise Matrix (R)
        # These values should roughly match the sensor's hardware specs (the noise we injected).
        self.R = np.array([
            [50.0**2, 0.0],       # Range variance
            [0.0, 0.02**2],       # Angle variance
        ])

        # 5. Process Noise Matrix (Q)
        # Represents unpredictable maneuvers (wind, pilot acceleration).
        self.Q = np.eye(4) * 0.5

        self.I = np.eye(4)

    def predict(self):
        """Step 1: Predict the next state using linear physics."""
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        """
        Step 2: Update the state using the noisy polar measurement.
        This is where the non-linear EKF magic (Jacobian) happens.
        """
        px, py, vx, vy = self.x[0, 0], self.x[1, 0], self.x[2, 0], self.x[3, 0]

        # Prevent division by zero if the target flies directly over the radar (0,0)
        c1 = px**2 + py**2
        if c1 < 0.0001:
            print("Target too close to origin, skipping update to prevent singularity.")
            return

        range_pred = math.sqrt(c1)
        angle_pred = math.atan2(py, px)

        # h(x): Predicted measurement based on our current Cartesian state estimation
        z_pred = np.array([[range_pred], [angle_pred]])

        # y: Residual (Difference between actual measurement and predicted measurement)
        y = z - z_pred

        # CRITICAL: Normalize the angle difference to be within -pi to pi
        # This prevents the filter from spiraling out of control when crossing the 180-degree boundary.
        while y[1, 0] > math.pi:
            y[1, 0] -= 2 * math.pi
        while y[1, 0] < -math.pi:
            y[1, 0] += 2 * math.pi

        # Hj: The Jacobian Matrix (Partial derivatives of h(x) evaluated at current state)
        # This "linearizes" the non-linear conversion right at our current position.
        Hj = np.array([
            [px / range_pred, py / range_pred, 0.0, 0.0],
            [-py / c1,        px / c1,         0.0, 0.0],
        ])

        # Standard Kalman Filter math, but using Hj instead of a static H
        S = np.dot(np.dot(Hj, self.P), Hj.T) + self.R
        K = np.dot(np.dot(self.P, Hj.T), np.linalg.inv(S))

        # Update the state and covariance
        self.x = self.x + np.dot(K, y)
        self.P = np.dot((self.I - np.dot(K, Hj)), self.P)

    def process_telemetry(self):
        """
        Listens for enriched telemetry from the ingestion pipeline via a
        PULL socket, runs the EKF, and profiles performance.

        v2 changes vs original process_telemetry():
          - Uses zmq.PULL (not zmq.SUB) — connects to ingestion_pipeline PUSH.
          - Parses JSON directly (recv_json); no "RADAR " prefix to strip.
          - Measures pipeline_latency = now - data['ingestion_ts'].
          - Detects pipeline_seq gaps (dropped messages).
        """
        import zmq
        import json
        import time

        # Setup ZeroMQ PULL socket — connect to the pipeline's PUSH output
        context = zmq.Context()
        socket = context.socket(zmq.PULL)
        socket.connect("tcp://localhost:5556")  # v2: connect to ingestion pipeline

        print("--- EKF Tracking Engine v2 Online: Pulling from tcp://localhost:5556 ---")

        raw_errors, ekf_errors, processing_times, pipeline_latencies = [], [], [], []
        expected_pipeline_seq = 0
        total_seq_gaps = 0

        try:
            while True:
                # Block and wait for the next enriched message from the pipeline
                data = socket.recv_json()

                # Capture time-of-arrival for pipeline latency measurement
                arrival_ts = time.perf_counter()

                # --- Sequence-gap detection ---
                pipeline_seq = data.get("pipeline_seq", expected_pipeline_seq)
                if pipeline_seq != expected_pipeline_seq:
                    gap = pipeline_seq - expected_pipeline_seq
                    total_seq_gaps += gap
                    print(
                        f"[WARN] Sequence gap detected: expected={expected_pipeline_seq} "
                        f"got={pipeline_seq} (gap={gap})"
                    )
                expected_pipeline_seq = pipeline_seq + 1

                # --- Pipeline latency ---
                ingestion_ts = data.get("ingestion_ts", arrival_ts)
                pipeline_latency = (arrival_ts - ingestion_ts) * 1000  # ms
                pipeline_latencies.append(pipeline_latency)

                # --- Extract measurement fields ---
                measured_r = data["noisy_range"]
                measured_theta = data["noisy_angle"]
                true_x = data["true_x"]
                true_y = data["true_y"]

                # --- EKF Loop ---
                start_time = time.perf_counter()
                z = np.array([[measured_r], [measured_theta]])
                self.predict()
                self.update(z)
                end_time = time.perf_counter()

                processing_times.append(end_time - start_time)

                # --- Error Metrics ---
                est_x, est_y = self.x[0, 0], self.x[1, 0]
                raw_x = measured_r * math.cos(measured_theta)
                raw_y = measured_r * math.sin(measured_theta)

                raw_errors.append((raw_x - true_x)**2 + (raw_y - true_y)**2)
                ekf_errors.append((est_x - true_x)**2 + (est_y - true_y)**2)

                # --- Periodic console output ---
                ts = data.get("timestamp", 0)
                if int(ts) % 10 == 0:
                    print(
                        f"[RX] seq={pipeline_seq:04d}  "
                        f"Tracked: X={est_x:.1f}, Y={est_y:.1f}  "
                        f"EKF={( end_time-start_time)*1000:.3f}ms  "
                        f"Pipeline={pipeline_latency:.3f}ms"
                    )

        except KeyboardInterrupt:
            import platform

            if not processing_times:
                print("No data processed.")
                return

            raw_rmse = math.sqrt(sum(raw_errors) / len(raw_errors))
            ekf_rmse = math.sqrt(sum(ekf_errors) / len(ekf_errors))
            avg_ekf_latency = (sum(processing_times) / len(processing_times)) * 1000
            avg_pipeline_latency = sum(pipeline_latencies) / len(pipeline_latencies)
            max_throughput = 1.0 / (avg_ekf_latency / 1000.0)

            os_name = platform.system()
            os_release = platform.release()
            cpu_processor = platform.processor()

            print("\n" + "=" * 50)
            print("=== EKF TRACKER v2 — PERFORMANCE REPORT ===")
            print("=" * 50)
            print(f"Hardware/OS        : {os_name} {os_release} | {cpu_processor}")
            print("-" * 50)
            print(f"Packets Processed  : {len(processing_times)}")
            print(f"Pipeline Seq Gaps  : {total_seq_gaps}")
            print(f"Raw Sensor RMSE    : {raw_rmse:.2f} meters")
            print(f"EKF Filter RMSE    : {ekf_rmse:.2f} meters")
            print(f"Noise Reduction    : {((raw_rmse - ekf_rmse) / raw_rmse) * 100:.1f}%")
            print("-" * 50)
            print(f"Avg EKF Latency    : {avg_ekf_latency:.4f} ms")
            print(f"Avg Pipeline Lat.  : {avg_pipeline_latency:.4f} ms")
            print(f"Max Throughput     : {max_throughput:,.0f} Hz (updates/sec)")
            print("=" * 50)

        finally:
            socket.close()
            context.term()


# Execution block
if __name__ == "__main__":
    tracker = RadarEKF()
    tracker.process_telemetry()
