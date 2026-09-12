import logging
import threading
import time
from collections import deque
from typing import Generator, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraStream:
    """Manages persistent OpenCV VideoCapture connection to an ESP32-CAM MJPEG stream.

    Runs an independent background capture thread to ensure minimal latency,
    caches the latest frame and pre-encoded JPEG, and handles disconnects/reconnects
    gracefully without blocking browser requests.
    """

    def __init__(self, stream_url: str, reconnect_interval: float = 2.0):
        self.stream_url = stream_url
        self.reconnect_interval = reconnect_interval

        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._new_frame_event = threading.Event()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # State
        self.is_connected = False
        self.status_message = "Initializing..."
        self.last_frame: Optional[np.ndarray] = None
        self.last_jpeg_bytes: Optional[bytes] = None
        self.last_frame_time = 0.0
        self.frame_width = 0
        self.frame_height = 0
        self.error_count = 0

        # FPS Tracking (moving window of 3.0 seconds)
        self._in_frame_times = deque()
        self._out_frame_times = deque()
        self.camera_fps = 0.0
        self.browser_fps = 0.0

        # Placeholder cache
        self._placeholder_jpeg: Optional[bytes] = None
        self._placeholder_time = 0.0

    def start(self) -> "CameraStream":
        """Start background frame capture thread."""
        if self._running:
            return self

        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="CameraCaptureThread",
            daemon=True,
        )
        self._thread.start()
        logger.info("CameraStream thread started for URL: %s", self.stream_url)
        return self

    def stop(self) -> None:
        """Stop background capture and release resources."""
        self._running = False
        self._new_frame_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._release_capture()
        logger.info("CameraStream stopped.")

    def _release_capture(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as e:
                logger.warning("Error releasing VideoCapture: %s", e)
            self._cap = None

    def _connect(self) -> bool:
        """Attempt to open cv2.VideoCapture with non-blocking timeouts."""
        self.status_message = f"Connecting to {self.stream_url}..."
        logger.info(self.status_message)
        self._release_capture()

        cap = cv2.VideoCapture()
        # Set 3 second open/read timeouts to prevent blocking indefinitely
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)

        opened = cap.open(self.stream_url)
        if not opened or not cap.isOpened():
            self.error_count += 1
            self.is_connected = False
            self.status_message = f"Failed to open stream at {self.stream_url}"
            logger.warning(self.status_message)
            cap.release()
            return False

        # Try reading initial frame
        ret, frame = cap.read()
        if not ret or frame is None:
            self.error_count += 1
            self.is_connected = False
            self.status_message = "Stream opened but failed to read initial frame"
            logger.warning(self.status_message)
            cap.release()
            return False

        self._cap = cap
        h, w = frame.shape[:2]
        now = time.time()

        # Pre-encode JPEG once
        success, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        jpeg_bytes = encoded.tobytes() if success else None

        with self._lock:
            self.last_frame = frame
            self.last_jpeg_bytes = jpeg_bytes
            self.frame_width = w
            self.frame_height = h
            self.last_frame_time = now
            self.is_connected = True
            self.status_message = "Live"
            self.error_count = 0
            self._in_frame_times.clear()
            self._in_frame_times.append(now)

        # Explicit diagnostic log
        print("\n==============================")
        print("Camera connected")
        print(f"Input resolution: {w}x{h}")
        print("==============================\n", flush=True)
        logger.info("Camera connected. Input resolution: %dx%d", w, h)

        self._new_frame_event.set()
        return True

    def _capture_loop(self) -> None:
        """Background thread loop for continuous reading and latency prevention."""
        while self._running:
            if not self.is_connected or self._cap is None or not self._cap.isOpened():
                success = self._connect()
                if not success:
                    time.sleep(self.reconnect_interval)
                    continue

            # Read next frame
            ret, frame = self._cap.read()
            now = time.time()

            if not ret or frame is None:
                self.error_count += 1
                self.is_connected = False
                self.status_message = "Lost connection to ESP32-CAM stream"
                self.camera_fps = 0.0
                logger.warning(
                    "Frame read failed. Reconnecting in %.1f seconds...",
                    self.reconnect_interval,
                )
                self._release_capture()
                time.sleep(self.reconnect_interval)
                continue

            h, w = frame.shape[:2]

            # Calculate incoming camera FPS using 3.0s moving window
            self._in_frame_times.append(now)
            while self._in_frame_times and self._in_frame_times[0] < now - 3.0:
                self._in_frame_times.popleft()

            if len(self._in_frame_times) > 1:
                duration = self._in_frame_times[-1] - self._in_frame_times[0]
                if duration > 0:
                    self.camera_fps = round((len(self._in_frame_times) - 1) / duration, 1)

            # Pre-encode JPEG once in worker thread (avoids redundant encoding per browser request)
            success, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            jpeg_bytes = encoded.tobytes() if success else None

            # Store latest frame and notify waiters
            with self._lock:
                self.last_frame = frame
                self.last_jpeg_bytes = jpeg_bytes
                self.last_frame_time = now
                self.frame_width = w
                self.frame_height = h
                self.is_connected = True
                self.status_message = "Live"

            self._new_frame_event.set()

        self._release_capture()

    def get_latest_frame(self) -> Tuple[bool, np.ndarray]:
        """Get the most recent frame or a generated status frame if offline."""
        with self._lock:
            if self.is_connected and self.last_frame is not None:
                return True, self.last_frame.copy()

        placeholder = self._create_placeholder_frame()
        return False, placeholder

    def _create_placeholder_frame(self, width: int = 640, height: int = 480) -> np.ndarray:
        """Create an informative dark placeholder frame when the camera is not feeding."""
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = (26, 20, 18)

        cv2.putText(
            img,
            "ESP32-CAM Feed Offline / Connecting",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            f"Target: {self.stream_url}",
            (30, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            f"Status: {self.status_message}",
            (30, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (100, 100, 255) if not self.is_connected else (100, 255, 100),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            (30, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            "Reconnection attempts in progress...",
            (30, 280),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (140, 140, 140),
            1,
            cv2.LINE_AA,
        )
        return img

    def get_jpeg_bytes(self) -> Optional[bytes]:
        """Get cached JPEG bytes or encode fallback placeholder."""
        with self._lock:
            if self.is_connected and self.last_jpeg_bytes is not None:
                return self.last_jpeg_bytes

        # Cache placeholder JPEG every 1 second to avoid re-encoding on every tick
        now = time.time()
        if self._placeholder_jpeg is None or (now - self._placeholder_time) > 1.0:
            placeholder = self._create_placeholder_frame()
            success, encoded = cv2.imencode(".jpg", placeholder, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if success:
                self._placeholder_jpeg = encoded.tobytes()
                self._placeholder_time = now

        return self._placeholder_jpeg

    def generate_mjpeg_stream(self, max_fps: int = 30) -> Generator[bytes, None, None]:
        """Generator yielding multipart MJPEG chunks for Flask responses."""
        interval = 1.0 / max_fps
        last_sent_time = 0.0

        while self._running:
            # Wait for a new frame event or timeout
            self._new_frame_event.wait(timeout=interval)
            self._new_frame_event.clear()

            now = time.time()
            with self._lock:
                frame_time = self.last_frame_time
                jpeg_bytes = self.last_jpeg_bytes if self.is_connected else self.get_jpeg_bytes()

            # If disconnected, send placeholder at ~2 FPS
            if not self.is_connected:
                if now - last_sent_time < 0.5:
                    time.sleep(0.1)
                    continue
                jpeg_bytes = self.get_jpeg_bytes()

            if jpeg_bytes and (frame_time != last_sent_time or not self.is_connected):
                last_sent_time = frame_time if self.is_connected else now

                # Measure browser delivery FPS (3.0s moving window)
                self._out_frame_times.append(now)
                while self._out_frame_times and self._out_frame_times[0] < now - 3.0:
                    self._out_frame_times.popleft()

                if len(self._out_frame_times) > 1:
                    duration = self._out_frame_times[-1] - self._out_frame_times[0]
                    if duration > 0:
                        self.browser_fps = round((len(self._out_frame_times) - 1) / duration, 1)

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
                )

    def get_status_dict(self) -> dict:
        """Returns JSON-serializable status dictionary with metrics."""
        now = time.time()
        # Expire browser fps if no active stream in > 3s
        if self._out_frame_times and (now - self._out_frame_times[-1]) > 3.0:
            self.browser_fps = 0.0

        return {
            "stream_url": self.stream_url,
            "is_connected": self.is_connected,
            "status_message": self.status_message,
            "camera_fps": self.camera_fps,
            "browser_fps": self.browser_fps,
            "fps": self.camera_fps,
            "resolution": f"{self.frame_width}x{self.frame_height}" if self.is_connected else "N/A",
            "last_frame_age_seconds": round(now - self.last_frame_time, 2) if self.last_frame_time > 0 else None,
            "error_count": self.error_count,
        }
