import logging
import threading
import time
from typing import Generator, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraStream:
    """Manages OpenCV VideoCapture connection to an ESP32-CAM MJPEG stream.

    Runs a background reading thread to keep frame latency minimal,
    handles network dropouts gracefully, and auto-reconnects.
    """

    def __init__(self, stream_url: str, reconnect_interval: float = 2.0):
        self.stream_url = stream_url
        self.reconnect_interval = reconnect_interval

        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # State
        self.is_connected = False
        self.status_message = "Initializing..."
        self.last_frame: Optional[np.ndarray] = None
        self.last_frame_time = 0.0
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 0.0
        self.error_count = 0

    def start(self) -> "CameraStream":
        """Start background frame capture thread."""
        if self._running:
            return self

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, name="CameraCaptureThread", daemon=True)
        self._thread.start()
        logger.info("CameraStream thread started for URL: %s", self.stream_url)
        return self

    def stop(self) -> None:
        """Stop background capture and release resources."""
        self._running = False
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
        """Attempt to open cv2.VideoCapture."""
        self.status_message = f"Connecting to {self.stream_url}..."
        logger.info(self.status_message)
        self._release_capture()

        cap = cv2.VideoCapture(self.stream_url)
        if not cap.isOpened():
            self.error_count += 1
            self.is_connected = False
            self.status_message = f"Failed to open stream at {self.stream_url}"
            logger.warning(self.status_message)
            cap.release()
            return False

        # Try reading the first frame to confirm it's actually streaming
        ret, frame = cap.read()
        if not ret or frame is None:
            self.error_count += 1
            self.is_connected = False
            self.status_message = "Stream opened but failed to read initial frame"
            logger.warning(self.status_message)
            cap.release()
            return False

        self._cap = cap
        with self._lock:
            self.last_frame = frame
            self.frame_height, self.frame_width = frame.shape[:2]
            self.last_frame_time = time.time()
            self.is_connected = True
            self.status_message = "Connected"
            self.error_count = 0

        logger.info(
            "Connected to ESP32-CAM stream. Resolution: %dx%d",
            self.frame_width,
            self.frame_height,
        )
        return True

    def _capture_loop(self) -> None:
        """Background thread loop for continuous reading and latency prevention."""
        frame_counter = 0
        fps_timer = time.time()

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
                logger.warning("Frame read failed. Reconnecting in %s seconds...", self.reconnect_interval)
                self._release_capture()
                time.sleep(self.reconnect_interval)
                continue

            # Update FPS calculation
            frame_counter += 1
            elapsed = now - fps_timer
            if elapsed >= 1.0:
                self.fps = round(frame_counter / elapsed, 1)
                frame_counter = 0
                fps_timer = now

            # Store latest frame
            with self._lock:
                self.last_frame = frame
                self.last_frame_time = now
                self.frame_height, self.frame_width = frame.shape[:2]
                self.is_connected = True
                self.status_message = "Live"

        self._release_capture()

    def get_latest_frame(self) -> Tuple[bool, np.ndarray]:
        """Get the most recent frame or a generated status frame if offline."""
        with self._lock:
            if self.is_connected and self.last_frame is not None:
                return True, self.last_frame.copy()

        # Generate a placeholder frame indicating connection state
        placeholder = self._create_placeholder_frame()
        return False, placeholder

    def _create_placeholder_frame(self, width: int = 640, height: int = 480) -> np.ndarray:
        """Create an informative dark placeholder frame when the camera is not feeding."""
        img = np.zeros((height, width, 3), dtype=np.uint8)
        # Background dark slate
        img[:] = (26, 20, 18)

        # Status text
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
        """Encode the current frame as JPEG bytes."""
        _, frame = self.get_latest_frame()
        success, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if success:
            return encoded.tobytes()
        return None

    def generate_mjpeg_stream(self, target_fps: int = 30) -> Generator[bytes, None, None]:
        """Generator function yielding multipart MJPEG chunks for Flask responses."""
        interval = 1.0 / target_fps
        while self._running:
            start_t = time.time()
            jpeg_bytes = self.get_jpeg_bytes()
            if jpeg_bytes:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
                )
            sleep_time = interval - (time.time() - start_t)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_status_dict(self) -> dict:
        """Returns JSON-serializable status dictionary."""
        return {
            "stream_url": self.stream_url,
            "is_connected": self.is_connected,
            "status_message": self.status_message,
            "fps": self.fps,
            "resolution": f"{self.frame_width}x{self.frame_height}" if self.is_connected else "N/A",
            "last_frame_age_seconds": round(time.time() - self.last_frame_time, 2) if self.last_frame_time > 0 else None,
            "error_count": self.error_count,
        }
