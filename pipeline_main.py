"""
pipeline_main.py  —  Pipeline Orchestrator
==========================================
Launches all three pipeline components as independent OS processes:

    ┌──────────────────────────┐
    │  RadarSimulator          │  AircraftTrajSimul_v2.py
    │  PUB  tcp://*:5555       │
    └──────────┬───────────────┘
               │  ZeroMQ PUB/SUB  (raw RADAR telemetry)
    ┌──────────▼───────────────┐
    │  TelemetryIngestionPipeline  │  ingestion_pipeline.py
    │  SUB  tcp://localhost:5555   │
    │  PUSH tcp://*:5556           │
    │  • validates messages        │
    │  • enriches (seq + ts)       │
    │  • buffers in queue          │
    │  • freshness-first drops     │
    └──────────┬───────────────┘
               │  ZeroMQ PUSH/PULL  (enriched telemetry)
    ┌──────────▼───────────────┐
    │  RadarEKF                │  ExtfKFTracker_v2.py
    │  PULL tcp://localhost:5556│
    │  • predict / update loop  │
    │  • performance metrics    │
    └──────────────────────────┘

Start order
-----------
  1. IngestionPipeline — binds both sockets first so the simulator and
     the EKF have endpoints to connect to.
  2. EKFTracker — connects PULL to pipeline's PUSH (port 5556).
  3. RadarSimulator — connects PUB last; waits startup_delay before
     sending so subscribers are ready (ZeroMQ slow-joiner safeguard).

Usage
-----
  python pipeline_main.py [--steps N] [--dt DT]

  Ctrl-C  →  gracefully terminates all three child processes.
"""

import multiprocessing
import signal
import sys
import time
import argparse


# ---------------------------------------------------------------------------
# Worker entry points (run in child processes)
# ---------------------------------------------------------------------------

def _run_pipeline(sub_addr: str, push_addr: str, queue_maxsize: int) -> None:
    from ingestion_pipeline import TelemetryIngestionPipeline

    pipeline = TelemetryIngestionPipeline(
        sub_addr=sub_addr,
        push_addr=push_addr,
        queue_maxsize=queue_maxsize,
        stats_interval=10.0,
    )
    pipeline.run()


def _run_tracker(pull_addr: str) -> None:
    import zmq, json, time, math, platform
    import numpy as np
    from ExtfKFTracker_v2 import RadarEKF

    tracker = RadarEKF()
    tracker.process_telemetry()


def _run_simulator(steps: int, dt: float, startup_delay: float) -> None:
    from AircraftTrajSimul_v2 import RadarSimulator

    radar = RadarSimulator(dt=dt)
    radar.run_simulation(steps=steps, startup_delay=startup_delay)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="OeFlakTrack — pipeline orchestrator")
    parser.add_argument("--steps", type=int, default=120,
                        help="Number of radar scan intervals to simulate (default: 120)")
    parser.add_argument("--dt", type=float, default=1.0,
                        help="Scan interval in seconds (default: 1.0)")
    parser.add_argument("--queue", type=int, default=100,
                        help="Ingestion pipeline buffer size (default: 100)")
    args = parser.parse_args()

    # ZeroMQ addresses
    SIM_PUB   = "tcp://*:5555"
    PIPE_SUB  = "tcp://localhost:5555"
    PIPE_PUSH = "tcp://*:5556"
    EKF_PULL  = "tcp://localhost:5556"  # used inside ExtfKFTracker_v2 directly

    print("=" * 60)
    print(" OeFlakTrack — Real-Time Telemetry Ingestion Pipeline")
    print("=" * 60)
    print(f"  Simulator : PUB on {SIM_PUB}  ({args.steps} steps × {args.dt}s)")
    print(f"  Pipeline  : SUB {PIPE_SUB} → PUSH {PIPE_PUSH}  (buffer={args.queue})")
    print(f"  EKF       : PULL {EKF_PULL}")
    print("-" * 60)

    # --- Spawn processes in start-order ---

    # 1. Ingestion pipeline — must bind sockets before others connect
    pipeline_proc = multiprocessing.Process(
        target=_run_pipeline,
        args=(PIPE_SUB, PIPE_PUSH, args.queue),
        name="IngestionPipeline",
        daemon=True,
    )
    pipeline_proc.start()
    time.sleep(0.4)  # Let PUSH socket bind before EKF connects

    # 2. EKF tracker — connects PULL to pipeline's PUSH
    tracker_proc = multiprocessing.Process(
        target=_run_tracker,
        args=(EKF_PULL,),
        name="EKFTracker",
        daemon=True,
    )
    tracker_proc.start()
    time.sleep(0.2)  # Brief pause before simulator starts publishing

    # 3. Simulator — publishes after startup_delay (slow-joiner safeguard)
    sim_proc = multiprocessing.Process(
        target=_run_simulator,
        args=(args.steps, args.dt, 1.5),
        name="RadarSimulator",
        daemon=True,
    )
    sim_proc.start()

    print(f"\n[Main] All components running. PIDs: "
          f"pipeline={pipeline_proc.pid}, "
          f"tracker={tracker_proc.pid}, "
          f"simulator={sim_proc.pid}")
    print("[Main] Press Ctrl-C to stop.\n")

    # --- Wait for natural completion or Ctrl-C ---
    try:
        sim_proc.join()          # Simulator finishes after `steps * dt` seconds
        print("[Main] Simulator finished. Waiting 3s for pipeline to drain …")
        time.sleep(3)
    except KeyboardInterrupt:
        print("\n[Main] Interrupted — shutting down.")
    finally:
        for proc in (sim_proc, tracker_proc, pipeline_proc):
            if proc.is_alive():
                proc.terminate()
        for proc in (sim_proc, tracker_proc, pipeline_proc):
            proc.join(timeout=5)
        print("[Main] All processes stopped. Goodbye.")


if __name__ == "__main__":
    # Required on macOS / Windows for multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
