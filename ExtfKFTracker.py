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
        self.F = np.array([[1.0, 0.0, self.dt, 0.0],
                           [0.0, 1.0, 0.0, self.dt],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]])
                           
        # 4. Measurement Noise Matrix (R)
        # These values should roughly match the sensor's hardware specs (the noise we injected).
        self.R = np.array([[50.0**2, 0.0],         # Range variance
                           [0.0,     0.02**2]])    # Angle variance
                           
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
        px, py, vx, vy = self.x[0,0], self.x[1,0], self.x[2,0], self.x[3,0]
        
        # Prevent division by zero if the target flies directly over the radar (0,0)
        c1 = px**2 + py**2
        if c1 < 0.0001:
            print("Target too close to origin, skipping update to prevent singularity.")
            return

        range_pred = math.sqrt(c1)
        angle_pred = math.atan2(py, px)
        
        # h(x): Predicted measurement based on our current Cartesian state estimation
        z_pred = np.array([[range_pred], 
                           [angle_pred]])
        
        # y: Residual (Difference between actual measurement and predicted measurement)
        y = z - z_pred
        
        # CRITICAL: Normalize the angle difference to be within -pi to pi
        # This prevents the filter from spiraling out of control when crossing the 180-degree boundary.
        while y[1,0] > math.pi: y[1,0] -= 2 * math.pi
        while y[1,0] < -math.pi: y[1,0] += 2 * math.pi

        # Hj: The Jacobian Matrix (Partial derivatives of h(x) evaluated at current state)
        # This "linearizes" the non-linear conversion right at our current position.
        Hj = np.array([
            [ px / range_pred,   py / range_pred,   0.0, 0.0 ],
            [-py / c1,           px / c1,           0.0, 0.0 ]
        ])

        # Standard Kalman Filter math, but using Hj instead of a static H
        S = np.dot(np.dot(Hj, self.P), Hj.T) + self.R
        K = np.dot(np.dot(self.P, Hj.T), np.linalg.inv(S))
        
        # Update the state and covariance
        self.x = self.x + np.dot(K, y)
        self.P = np.dot((self.I - np.dot(K, Hj)), self.P)

    def process_telemetry(self):
        """Listens for live telemetry via ZeroMQ, runs the EKF, and profiles performance."""
        import zmq
        import json
        import time
        
        # Setup ZeroMQ Subscriber
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.connect("tcp://localhost:5555")
        socket.setsockopt_string(zmq.SUBSCRIBE, "RADAR") # Only listen to RADAR topics
        
        print("--- EKF Tracking Engine Online: Listening on tcp://localhost:5555 ---")
        
        raw_errors, ekf_errors, processing_times = [], [], []
        
        try:
            while True: # Infinite loop to continuously listen for live data
                # Block and wait for the next message
                raw_msg = socket.recv_string()
                
                # Start latency timer the microsecond the packet is received
                start_time = time.perf_counter()
                
                # Strip the "RADAR " topic prefix and parse the JSON
                json_str = raw_msg.split(" ", 1)[1]
                data = json.loads(json_str)
                
                measured_r = data['noisy_range']
                measured_theta = data['noisy_angle']
                true_x = data['true_x']
                true_y = data['true_y']
                
                # EKF Loop
                z = np.array([[measured_r], [measured_theta]])
                self.predict()
                self.update(z)
                
                # Stop latency timer
                end_time = time.perf_counter()
                processing_times.append(end_time - start_time)
                
                # Calculate Error Metrics
                est_x, est_y = self.x[0,0], self.x[1,0]
                raw_x = measured_r * math.cos(measured_theta)
                raw_y = measured_r * math.sin(measured_theta)
                
                raw_errors.append((raw_x - true_x)**2 + (raw_y - true_y)**2)
                ekf_errors.append((est_x - true_x)**2 + (est_y - true_y)**2)
                
                if data['timestamp'] % 10 == 0:
                    print(f"[RX] Tracked Pos: X={est_x:.1f}, Y={est_y:.1f} | EKF Latency: {(end_time-start_time)*1000:.3f}ms")

        except KeyboardInterrupt:
            import platform # Import the platform module to get hardware specs
            
            raw_rmse = math.sqrt(sum(raw_errors) / len(raw_errors))
            ekf_rmse = math.sqrt(sum(ekf_errors) / len(ekf_errors))
            avg_latency = (sum(processing_times) / len(processing_times)) * 1000
            max_throughput = 1.0 / (avg_latency / 1000.0)
            
            # Dynamically grab the system hardware information
            os_name = platform.system()
            os_release = platform.release()
            cpu_processor = platform.processor()
            
            print("\n" + "="*40)
            print("=== LIVE STREAM PERFORMANCE REPORT ===")
            print("="*40)
            print(f"Hardware/OS   : {os_name} {os_release} | {cpu_processor}")
            print("-" * 40)
            print(f"Packets Processed : {len(processing_times)}")
            print(f"Raw Sensor RMSE   : {raw_rmse:.2f} meters")
            print(f"EKF Filter RMSE   : {ekf_rmse:.2f} meters")
            print(f"Noise Reduction   : {((raw_rmse - ekf_rmse) / raw_rmse) * 100:.1f}%")
            print("-" * 40)
            print(f"Avg EKF Latency   : {avg_latency:.4f} ms")
            print(f"Max Throughput    : {max_throughput:,.0f} Hz (updates/sec)")
            print("="*40)
# Execution block
if __name__ == "__main__":
    tracker = RadarEKF()
    tracker.process_telemetry()