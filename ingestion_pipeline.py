"""
ingestion_pipeline.py  —  ZeroMQ Telemetry Ingestion Pipeline
==============================================================
Sits between the radar simulator (PUB) and the EKF tracker (PULL),
fully decoupling data *reception* from state *estimation*.

Data-flow
---------
  RadarSimulator  ──PUB:5555──▶  [Receiver thread]
                                      │ validates, enriches
                                      ▼
                              thread-safe Queue (buffer)
                                      │
                              [Forwarder thread]
                                      │
                  ◀──PUSH:5556──  EKFTracker

Key responsibilities
--------------------
  1. Receive raw RADAR PUB messages over SUB socket.
  2. Validate required fields and numeric types.
  3. Enrich each message with pipeline sequence number and ingestion timestamp.
  4. Buffer messages in a thread-safe FIFO queue.
  5. Apply backpressure: when the queue is full, drop the *oldest* message so
     the pipeline never blocks the simulator and always holds the freshest data.
  6. Forward enriched messages to the EKF tracker via a PUSH socket.
  7. Emit periodic pipeline statistics (throughput, queue depth, drop rate).
"""

import json
import queue
import threading
import time
import signal
import sys
import zmq


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class TelemetryIngestionPipeline:
    """
    Decouples ZeroMQ message reception from the state-estimation engine by
    running two threads connected by an internal bounded queue.

    Parameters
    ----------
    sub_addr : str
        Address of the radar simulator's PUB socket, e.g. "tcp://localhost:5555".
    push_addr : str
        Address to bind the PUSH socket for the EKF consumer, e.g. "tcp://*:5556".
    queue_maxsize : int
        Maximum number of messages held in the internal buffer.  When full the
        oldest message is evicted to make room for the newest (freshness-first
        drop policy).
    stats_interval : float
        How often (seconds) to print pipeline statistics to stdout.
    """

    # Required fields and their expected types
    _REQUIRED_FIELDS = {
        "seq": (int, float),
        "timestamp": (int, float),
        "noisy_range": (int, float),
        "noisy_angle": (int, float),
        "true_x": (int, float),
        "true_y": (int, float),
    }

    def __init__(
        self,
        sub_addr: str = "tcp://localhost:5555",
        push_addr: str = "tcp://*:5556",
        queue_maxsize: int = 100,
        stats_interval: float = 10.0,
    ):
        self.sub_addr = sub_addr
        self.push_addr = push_addr
        self.queue_maxsize = queue_maxsize
        self.stats_interval = stats_interval

        # Bounded FIFO buffer shared between receiver and forwarder threads
        self._queue: queue.Queue = queue.Queue(maxsize=queue_maxsize)

        # Pipeline statistics (updated by receiver/forwarder threads)
        self._stats = {
            "received": 0,
            "forwarded": 0,
            "dropped": 0,
            "validation_errors": 0,
            "queue_peak_depth": 0,
        }
        self._stats_lock = threading.Lock()
        self._pipeline_seq = 0          # monotone counter added to each enriched message
        self._start_wall: float = 0.0   # wall-clock time of pipeline start (set in start())

        self._running = False
        self._recv_thread: threading.Thread | None = None
        self._fwd_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate(self, data: dict) -> bool:
        """Return True iff *data* contains all required fields with numeric values."""
        for field, types in self._REQUIRED_FIELDS.items():
            if field not in data:
                return False
            if not isinstance(data[field], types):
                return False
        # Sanity: range must be positive
        if data["noisy_range"] <= 0:
            return False
        return True

    def _increment_stat(self, key: str, delta: int = 1) -> None:
        with self._stats_lock:
            self._stats[key] += delta

    # ------------------------------------------------------------------
    # Receiver thread — SUB socket → internal queue
    # ------------------------------------------------------------------

    def _receiver_loop(self) -> None:
        """
        Subscribes to the simulator's PUB socket.  For every valid message:
          - enriches it with pipeline_seq and ingestion_timestamp
          - attempts to push it onto the internal queue
          - if the queue is full, evicts the oldest message first (freshness-first)
        """
        context = zmq.Context()
        sub = context.socket(zmq.SUB)
        sub.connect(self.sub_addr)
        sub.setsockopt_string(zmq.SUBSCRIBE, "RADAR")
        sub.setsockopt(zmq.RCVTIMEO, 200)  # 200 ms poll timeout

        print(f"[Pipeline|Receiver] Subscribed to {self.sub_addr}")

        while self._running:
            try:
                raw_msg = sub.recv_string()
            except zmq.Again:
                # No message within timeout — check _running flag and loop
                continue
            except zmq.ZMQError as exc:
                if self._running:
                    print(f"[Pipeline|Receiver] ZMQ error: {exc}")
                break

            self._increment_stat("received")

            # Strip "RADAR " topic prefix
            try:
                _, json_str = raw_msg.split(" ", 1)
                data: dict = json.loads(json_str)
            except (ValueError, json.JSONDecodeError):
                self._increment_stat("validation_errors")
                continue

            if not self._validate(data):
                self._increment_stat("validation_errors")
                continue

            # Enrich: add pipeline bookkeeping fields
            data["pipeline_seq"] = self._pipeline_seq
            data["ingestion_ts"] = time.perf_counter()  # high-res wall-clock
            self._pipeline_seq += 1

            # Backpressure: evict oldest if full, then insert newest
            if self._queue.full():
                try:
                    self._queue.get_nowait()   # discard oldest
                except queue.Empty:
                    pass
                self._increment_stat("dropped")

            self._queue.put_nowait(data)

            # Track peak queue depth
            depth = self._queue.qsize()
            with self._stats_lock:
                if depth > self._stats["queue_peak_depth"]:
                    self._stats["queue_peak_depth"] = depth

        sub.close()
        context.term()
        print("[Pipeline|Receiver] Thread stopped.")

    # ------------------------------------------------------------------
    # Forwarder thread — internal queue → PUSH socket
    # ------------------------------------------------------------------

    def _forwarder_loop(self) -> None:
        """
        Dequeues enriched messages and pushes them to the EKF tracker.
        Binds the PUSH socket so the EKF (PULL) can connect at will.
        """
        context = zmq.Context()
        push = context.socket(zmq.PUSH)
        push.bind(self.push_addr)

        print(f"[Pipeline|Forwarder] Bound PUSH socket on {self.push_addr}")

        while self._running:
            try:
                data = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                push.send_json(data)
                self._increment_stat("forwarded")
            except zmq.ZMQError as exc:
                print(f"[Pipeline|Forwarder] ZMQ error: {exc}")
                # Put the message back so it isn't silently lost
                try:
                    self._queue.put_nowait(data)
                except queue.Full:
                    pass

        push.close()
        context.term()
        print("[Pipeline|Forwarder] Thread stopped.")

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _stats_loop(self) -> None:
        """Periodically prints a one-line pipeline health summary."""
        while self._running:
            time.sleep(self.stats_interval)
            if not self._running:
                break
            self._print_stats()

    def _print_stats(self) -> None:
        elapsed = time.perf_counter() - self._start_wall
        with self._stats_lock:
            s = dict(self._stats)
        queue_now = self._queue.qsize()
        rx_rate = s["received"] / elapsed if elapsed > 0 else 0.0
        fwd_rate = s["forwarded"] / elapsed if elapsed > 0 else 0.0
        drop_pct = (s["dropped"] / s["received"] * 100) if s["received"] > 0 else 0.0

        print(
            f"[Pipeline|Stats] "
            f"elapsed={elapsed:.0f}s  "
            f"rx={s['received']} ({rx_rate:.1f}/s)  "
            f"fwd={s['forwarded']} ({fwd_rate:.1f}/s)  "
            f"dropped={s['dropped']} ({drop_pct:.1f}%)  "
            f"val_err={s['validation_errors']}  "
            f"queue={queue_now}/{self.queue_maxsize}  "
            f"queue_peak={s['queue_peak_depth']}"
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start receiver, forwarder, and stats threads."""
        self._running = True
        self._start_wall = time.perf_counter()

        self._recv_thread = threading.Thread(
            target=self._receiver_loop, name="Pipeline-Receiver", daemon=True
        )
        self._fwd_thread = threading.Thread(
            target=self._forwarder_loop, name="Pipeline-Forwarder", daemon=True
        )
        self._stats_thread = threading.Thread(
            target=self._stats_loop, name="Pipeline-Stats", daemon=True
        )

        self._fwd_thread.start()   # bind PUSH socket first
        time.sleep(0.1)            # brief pause so PUSH binds before PULL connects
        self._recv_thread.start()
        self._stats_thread.start()

        print(
            f"[Pipeline] Started — buffering {self.queue_maxsize} messages, "
            f"stats every {self.stats_interval}s"
        )

    def stop(self) -> None:
        """Signal all threads to stop and wait for them to finish."""
        print("[Pipeline] Shutting down …")
        self._running = False
        if self._recv_thread:
            self._recv_thread.join(timeout=3)
        if self._fwd_thread:
            self._fwd_thread.join(timeout=3)

    def run(self) -> None:
        """
        Blocking entry point.  Starts the pipeline and blocks until
        SIGINT (Ctrl-C), then prints a final stats report.
        """
        self.start()

        def _handle_signal(sig, frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
            print("\n" + "=" * 50)
            print("=== INGESTION PIPELINE — FINAL REPORT ===")
            print("=" * 50)
            self._print_stats()
            print("=" * 50)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pipeline = TelemetryIngestionPipeline(
        sub_addr="tcp://localhost:5555",
        push_addr="tcp://*:5556",
        queue_maxsize=100,
        stats_interval=10.0,
    )
    pipeline.run()
