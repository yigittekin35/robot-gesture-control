import logging
import os
import threading
import time
from collections import deque
from typing import Generator, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("Camera")


class CameraWorker:
    """Manages a single, persistent OpenCV VideoCapture connection to an ESP32-CAM stream.

    Runs a dedicated background thread to keep only the newest frame in memory,
    measures true incoming camera FPS, exposes thread-safe access to Flask,
    and automatically reconnects on network dropouts.
    """

    def __init__(self, stream_url: str, reconnect_delay: float = 1.0):
        self.stream_url = stream_url
        self.reconnect_delay = reconnect_delay

        # OpenCV & Threading
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._new_frame_event = threading.Event()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # State according to specification
        self.connected = False
        self.connection_state = "Offline"  # Live, Reconnecting, Offline
        self.latest_frame: Optional[np.ndarray] = None
        self.last_frame_time = 0.0
        self.frame_width = 0
        self.frame_height = 0
        self.error_count = 0

        # FPS Tracking (rolling 2.0s measurement window)
        self._camera_frame_times = deque()
        self._last_web_frame_time = 0.0
        self.camera_fps = 0.0
        self.web_fps = 0.0

        # Pre-encoded JPEG cache to avoid redundant compressions across clients
        self._cached_jpeg: Optional[bytes] = None
        self._cached_jpeg_time = 0.0

    @property
    def is_connected(self) -> bool:
        """Backwards-compatible alias for self.connected."""
        return self.connected

    @property
    def status_message(self) -> str:
        """Backwards-compatible alias for connection_state."""
        return self.connection_state

    def start(self) -> "CameraWorker":
        """Start the background worker capture thread."""
        if self._running:
            return self

        print(f"Camera source: {self.stream_url}")
        print("Camera worker started", flush=True)
        logger.info("Camera source: %s", self.stream_url)
        logger.info("Camera worker started")

        self._running = True
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="CameraWorkerThread",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop background capture cleanly and release video capture."""
        self._running = False
        self._new_frame_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._release_capture()
        logger.info("[Camera] Stopped")

    def _release_capture(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as e:
                logger.warning("Error releasing VideoCapture: %s", e)
            self._cap = None

    def _open_capture(self) -> bool:
        """Open the single persistent VideoCapture with socket timeouts."""
        self._release_capture()

        cap = cv2.VideoCapture()
        # Set 3 second socket open and read timeouts to prevent native OpenCV hangs
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)

        opened = cap.open(self.stream_url)
        if not opened or not cap.isOpened():
            cap.release()
            return False

        # Read initial frame to verify live data
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            return False

        h, w = frame.shape[:2]
        now = time.time()

        # Cache initial JPEG bytes
        success, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        jpeg_bytes = encoded.tobytes() if success else None

        with self._lock:
            self._cap = cap
            self.latest_frame = frame
            self.last_frame_time = now
            self.frame_width = w
            self.frame_height = h
            self.connected = True
            self.connection_state = "Live"
            self.error_count = 0
            self._camera_frame_times.clear()
            self._camera_frame_times.append(now)
            self._cached_jpeg = jpeg_bytes
            self._cached_jpeg_time = now

        print(f"[Camera] Connected (resolution: {w}x{h})", flush=True)
        logger.info("[Camera] Connected")

        self._new_frame_event.set()
        return True

    def _worker_loop(self) -> None:
        """Continuously reads frames and maintains connection state."""
        while self._running:
            if not self.connected or self._cap is None or not self._cap.isOpened():
                with self._lock:
                    self.connection_state = "Reconnecting"
                    self.connected = False
                print("[Camera] Reconnecting...", flush=True)
                logger.info("[Camera] Reconnecting...")

                connected = self._open_capture()
                if not connected:
                    with self._lock:
                        self.error_count += 1
                        self.connection_state = "Reconnecting"
                        self.connected = False
                        self.camera_fps = 0.0
                    time.sleep(self.reconnect_delay)
                    continue

            # Read next frame
            ret, frame = self._cap.read()
            now = time.time()

            if not ret or frame is None:
                print("[Camera] Stream lost", flush=True)
                logger.warning("[Camera] Stream lost")

                self._release_capture()
                with self._lock:
                    self.connected = False
                    self.connection_state = "Reconnecting"
                    self.camera_fps = 0.0
                    self.error_count += 1

                time.sleep(self.reconnect_delay)
                continue

            h, w = frame.shape[:2]

            # Update rolling 2.0s window for camera FPS
            self._camera_frame_times.append(now)
            while self._camera_frame_times and self._camera_frame_times[0] < now - 2.0:
                self._camera_frame_times.popleft()

            if len(self._camera_frame_times) > 1:
                duration = self._camera_frame_times[-1] - self._camera_frame_times[0]
                if duration > 0:
                    self.camera_fps = round((len(self._camera_frame_times) - 1) / duration, 1)

            # Pre-encode JPEG once in background thread for all web clients
            success, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            jpeg_bytes = encoded.tobytes() if success else None

            # Store newest frame safely
            with self._lock:
                self.latest_frame = frame
                self.last_frame_time = now
                self.frame_width = w
                self.frame_height = h
                self.connected = True
                self.connection_state = "Live"
                self._cached_jpeg = jpeg_bytes
                self._cached_jpeg_time = now

            self._new_frame_event.set()

        self._release_capture()
        with self._lock:
            self.connected = False
            self.connection_state = "Offline"

    def get_latest_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Safely returns a copy of the latest frame so caller cannot mutate stored frame."""
        with self._lock:
            if self.connected and self.latest_frame is not None:
                return True, self.latest_frame.copy()
            return False, None

    def get_latest_jpeg(self) -> Optional[bytes]:
        """Returns the pre-encoded JPEG bytes for streaming, or a placeholder if offline."""
        with self._lock:
            if self.connected and self._cached_jpeg is not None:
                return self._cached_jpeg

        # Fallback offline placeholder frame
        placeholder = self._create_placeholder(640, 480)
        success, encoded = cv2.imencode(".jpg", placeholder, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        return encoded.tobytes() if success else None

    def generate_mjpeg_stream(self, max_fps: int = 30) -> Generator[bytes, None, None]:
        """Yields multipart MJPEG chunks to browser clients using the stored latest frame."""
        interval = 1.0 / max_fps
        last_sent_time = 0.0
        client_frame_times = deque()

        while self._running:
            # Wait for next frame event or timeout
            self._new_frame_event.wait(timeout=interval)
            self._new_frame_event.clear()

            now = time.time()
            with self._lock:
                is_live = self.connected
                frame_time = self.last_frame_time
                jpeg_bytes = self._cached_jpeg if is_live else None

            # If stream is offline or reconnecting, serve placeholder at ~2 FPS without spinning
            if not is_live:
                if now - last_sent_time < 0.5:
                    time.sleep(0.05)
                    continue
                jpeg_bytes = self.get_latest_jpeg()
                frame_time = now

            if jpeg_bytes and (frame_time != last_sent_time or not is_live):
                last_sent_time = frame_time

                # Track rolling 2.0s window for this browser client stream
                client_frame_times.append(now)
                while client_frame_times and client_frame_times[0] < now - 2.0:
                    client_frame_times.popleft()

                if len(client_frame_times) > 1:
                    duration = client_frame_times[-1] - client_frame_times[0]
                    if duration > 0:
                        fps = round((len(client_frame_times) - 1) / duration, 1)
                        with self._lock:
                            self.web_fps = fps
                            self._last_web_frame_time = now

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
                )

    def _create_placeholder(self, width: int = 640, height: int = 480) -> np.ndarray:
        """Generates an informative placeholder frame when disconnected."""
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = (26, 20, 18)

        status_text = f"Status: {self.connection_state}"
        color = (100, 255, 100) if self.connected else (100, 100, 255)

        cv2.putText(img, "ESP32-CAM Feed Offline", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"Target: {self.stream_url}", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(img, status_text, (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        cv2.putText(img, f"Time: {time.strftime('%H:%M:%S')}", (30, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
        if self.connection_state == "Reconnecting":
            cv2.putText(img, "Attempting auto-reconnect...", (30, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1, cv2.LINE_AA)
        return img

    def get_status_dict(self) -> dict:
        """Returns JSON metrics dictionary matching the existing UI contract."""
        now = time.time()
        # Expire web FPS if no client pulled frames in > 2.5s
        if now - self._last_web_frame_time > 2.5:
            self.web_fps = 0.0

        return {
            "stream_url": self.stream_url,
            "is_connected": self.connected,
            "connected": self.connected,
            "connection_state": self.connection_state,
            "status_message": self.connection_state,
            "camera_fps": self.camera_fps,
            "web_fps": self.web_fps,
            "browser_fps": self.web_fps,  # backwards compatibility
            "fps": self.camera_fps,       # backwards compatibility
            "resolution": f"{self.frame_width}x{self.frame_height}" if self.connected else "N/A",
            "last_frame_age_seconds": round(now - self.last_frame_time, 2) if self.last_frame_time > 0 else None,
            "error_count": self.error_count,
        }
