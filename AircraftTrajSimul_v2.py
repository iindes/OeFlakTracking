"""
AircraftTrajSimul_v2.py  —  Radar Telemetry Simulator (v2)
Copied from AircraftTrajSimul.py; changes:
  - Adds 'seq' (sequence number) to each published payload so downstream
    components can detect dropped or out-of-order messages.
  - Accepts a configurable startup_delay so the ingestion pipeline has time
    to bind its PUSH socket before the first message is sent.
"""

import numpy as np
import csv
import math


class RadarSimulator:
    def __init__(self, dt=1.0, noise_std_range=50.0, noise_std_angle=0.02):
        """
        Initialize the radar simulator.
        :param dt: Radar scan interval (seconds)
        :param noise_std_range: Standard deviation of range measurement noise (meters)
        :param noise_std_angle: Standard deviation of azimuth angle measurement noise (radians)
        """
        self.dt = dt
        self.noise_std_range = noise_std_range
        self.noise_std_angle = noise_std_angle

        # The initial 'true' state of the target [X position, Y position, X velocity, Y velocity]
        # e.g., Starting at X=10000m, Y=5000m, flying at -200m/s on the X-axis and -50m/s on the Y-axis
        self.true_state = np.array([10000.0, 5000.0, -200.0, -50.0])

    def generate_true_position(self):
        """Calculates the target's next 'true' position based on a constant velocity model."""
        self.true_state[0] += self.true_state[2] * self.dt  # X = X + Vx*dt
        self.true_state[1] += self.true_state[3] * self.dt  # Y = Y + Vy*dt
        return self.true_state[0], self.true_state[1]

    def measure_polar_with_noise(self, true_x, true_y):
        """Converts Cartesian coordinates (X, Y) to polar coordinates (range, azimuth) and injects Gaussian noise."""
        # 1. Calculate perfect polar coordinates (Ground Truth)
        true_range = math.sqrt(true_x**2 + true_y**2)
        true_angle = math.atan2(true_y, true_x)

        # 2. Inject mechanical limitations (noise) of the radar
        noisy_range = true_range + np.random.normal(0, self.noise_std_range)
        noisy_angle = true_angle + np.random.normal(0, self.noise_std_angle)

        return noisy_range, noisy_angle

    def run_simulation(self, steps=60, startup_delay=1.5):
        """
        Streams live radar telemetry over a ZeroMQ PUB socket.

        :param steps: Number of scan intervals to simulate.
        :param startup_delay: Seconds to wait after binding before sending the
                               first message, giving downstream sockets time to
                               connect. (v2 addition)
        """
        import zmq
        import json
        import time

        # Setup ZeroMQ Publisher
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        socket.bind("tcp://*:5555")  # Broadcasting on port 5555

        print("--- Radar Node Active: Broadcasting telemetry on tcp://*:5555 ---")
        print(f"    Waiting {startup_delay}s for subscribers to connect …")
        time.sleep(startup_delay)  # ZeroMQ PUB/SUB slow-joiner safeguard

        for t in range(steps):
            true_x, true_y = self.generate_true_position()
            noisy_range, noisy_angle = self.measure_polar_with_noise(true_x, true_y)

            # Package the data as a JSON payload
            # v2: 'seq' added so the pipeline can detect gaps/drops
            payload = {
                "seq": t,
                "timestamp": t * self.dt,
                "noisy_range": noisy_range,
                "noisy_angle": noisy_angle,
                "true_x": true_x,
                "true_y": true_y,
            }

            # Publish the message with a "RADAR" topic prefix
            message = f"RADAR {json.dumps(payload)}"
            socket.send_string(message)

            if t % 10 == 0:
                print(f"[TX] seq={t:04d}  range={payload['noisy_range']:.1f}m")

            # Pause for dt to simulate real-time physical radar rotation
            time.sleep(self.dt)

        print("--- Radar Node Offline ---")
        socket.close()
        context.term()


# Execution block
if __name__ == "__main__":
    radar = RadarSimulator()
    radar.run_simulation(steps=60)
