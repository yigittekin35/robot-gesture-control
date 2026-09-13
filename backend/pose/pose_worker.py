"""Pose estimation and gesture diagnostic worker using MediaPipe PoseLandmarker.
Optimized for distant detection in low-resolution (320x240) indoor environments.
"""
import logging
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Generator, Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

logger = logging.getLogger("PoseWorker")

# MediaPipe Pose Landmark Indices:
# 11: left_shoulder, 12: right_shoulder
# 13: left_elbow,    14: right_elbow
# 15: left_wrist,    16: right_wrist
# 23: left_hip,      24: right_hip
# 25: left_knee,     26: right_knee
# 27: left_ankle,    28: right_ankle

ESSENTIAL_LANDMARKS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

SKELETON_CONNECTIONS = [
    # Shoulders
    (11, 12),
    # Left arm
    (11, 13),
    (13, 15),
    # Right arm
    (12, 14),
    (14, 16),
    # Torso
    (11, 23),
    (12, 24),
    (23, 24),
    # Left leg
    (23, 25),
    (25, 27),
    # Right leg
    (24, 26),
    (26, 28),
]


class PoseWorker:
    """Consumes frames from CameraWorker, runs MediaPipe pose detection,
    draws essential skeleton landmarks, diagnoses raised hands,
    and caches annotated JPEG frames for web streaming.
    """

    def __init__(self, camera_worker, model_path: Optional[str] = None):
        self.camera_worker = camera_worker

        if model_path is None:
            model_path = str(Path(__file__).parent / "models" / "pose_landmarker_full.task")
        self.model_path = model_path

        self._detector: Optional[vision.PoseLandmarker] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._new_annotated_event = threading.Event()

        # State
        self.pose_detected = False
        self.pose_state = "NO POSE"
        self.confidence_score = 0.0
        self.pose_fps = 0.0
        self.last_process_time = 0.0

        # Smoothing window for state (last 3 detections)
        self._state_history = deque(maxlen=3)
        self._pose_frame_times = deque()

        # Cached annotated frame
        self._annotated_jpeg: Optional[bytes] = None
        self._last_camera_frame_time = 0.0

        # CLAHE instance for contrast enhancement
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def start(self) -> "PoseWorker":
        """Initialize MediaPipe and start processing thread."""
        if self._running:
            return self

        logger.info("Initializing MediaPipe PoseLandmarker from: %s", self.model_path)
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        # Optimized thresholds: 0.3 for distant and low-resolution kitchen silhouettes
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=0.25,
            min_pose_presence_confidence=0.25,
            min_tracking_confidence=0.25,
        )
        self._detector = vision.PoseLandmarker.create_from_options(options)

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, name="PoseWorkerThread", daemon=True)
        self._thread.start()
        logger.info("PoseWorker thread started.")
        return self

    def stop(self) -> None:
        """Stop processing thread and clean up resources."""
        self._running = False
        self._new_annotated_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._detector:
            try:
                self._detector.close()
            except Exception as e:
                logger.warning("Error closing detector: %s", e)
            self._detector = None
        logger.info("PoseWorker stopped.")

    def _run_loop(self) -> None:
        """Background loop reading newest frames from CameraWorker."""
        while self._running:
            # Check if CameraWorker has a fresh frame
            with self.camera_worker._lock:
                current_time = self.camera_worker.last_frame_time
                is_live = self.camera_worker.connected

            if not is_live or current_time == self._last_camera_frame_time:
                time.sleep(0.02)
                continue

            # Fetch safe copy of latest frame
            has_frame, frame = self.camera_worker.get_latest_frame()
            if not has_frame or frame is None:
                time.sleep(0.02)
                continue

            self._last_camera_frame_time = current_time

            # Process frame with MediaPipe
            annotated_frame, detected, state, conf = self._process_frame(frame)
            now = time.time()

            # Measure pose FPS (rolling 2s window)
            self._pose_frame_times.append(now)
            while self._pose_frame_times and self._pose_frame_times[0] < now - 2.0:
                self._pose_frame_times.popleft()

            pose_fps = 0.0
            if len(self._pose_frame_times) > 1:
                duration = self._pose_frame_times[-1] - self._pose_frame_times[0]
                if duration > 0:
                    pose_fps = round((len(self._pose_frame_times) - 1) / duration, 1)

            # Pre-encode JPEG once
            success, encoded = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            jpeg_bytes = encoded.tobytes() if success else None

            with self._lock:
                self.pose_detected = detected
                self.pose_state = state
                self.confidence_score = conf
                self.pose_fps = pose_fps
                self.last_process_time = now
                self._annotated_jpeg = jpeg_bytes

            self._new_annotated_event.set()

    def _process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, bool, str, float]:
        """Detect pose, draw skeleton, determine gesture state, and overlay diagnostic text.
        Includes distance super-resolution upscale & CLAHE contrast boost for distant kitchen detection.
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]

        # 1. 2x upscale (e.g. 320x240 -> 640x480) so distant human silhouettes enter detector anchor receptive field
        upscaled = cv2.resize(frame, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

        # 2. Contrast enhancement via CLAHE on L-channel (helps dim indoor lighting at distance)
        lab = cv2.cvtColor(upscaled, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        cl = self._clahe.apply(l_chan)
        enhanced_lab = cv2.merge((cl, a_chan, b_chan))
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        rgb_frame = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self._detector.detect(mp_image)

        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            self._state_history.append("NO POSE")
            smoothed_state = self._get_smoothed_state()
            self._draw_hud(annotated, False, smoothed_state, 0.0)
            return annotated, False, smoothed_state, 0.0

        landmarks = results.pose_landmarks[0]

        # Average visibility/confidence for essential landmarks
        conf_values = [
            landmarks[idx].visibility
            for idx in ESSENTIAL_LANDMARKS
            if hasattr(landmarks[idx], "visibility") and landmarks[idx].visibility is not None
        ]
        avg_conf = round(float(np.mean(conf_values)) * 100, 1) if conf_values else 0.0

        # Draw skeleton connections (lines) - threshold 0.2 for distant landmarks
        for idx1, idx2 in SKELETON_CONNECTIONS:
            lm1 = landmarks[idx1]
            lm2 = landmarks[idx2]

            v1 = getattr(lm1, "visibility", 1.0) or 1.0
            v2 = getattr(lm2, "visibility", 1.0) or 1.0

            if v1 > 0.15 and v2 > 0.15:
                x1, y1 = int(lm1.x * w), int(lm1.y * h)
                x2, y2 = int(lm2.x * w), int(lm2.y * h)
                cv2.line(annotated, (x1, y1), (x2, y2), (255, 200, 0), 2, cv2.LINE_AA)

        # Draw landmark points (circles)
        for idx in ESSENTIAL_LANDMARKS:
            lm = landmarks[idx]
            v = getattr(lm, "visibility", 1.0) or 1.0
            if v > 0.15:
                cx, cy = int(lm.x * w), int(lm.y * h)
                if idx in (15, 16):  # Wrists highlighted in cyan/white
                    cv2.circle(annotated, (cx, cy), 5, (0, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(annotated, (cx, cy), 7, (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    cv2.circle(annotated, (cx, cy), 4, (16, 220, 100), -1, cv2.LINE_AA)

        # Raised-hand diagnostic
        # In MediaPipe normalized Y: smaller Y is HIGHER in image.
        margin = 0.02  # ~5-6 pixels tolerance for distant kitchen poses

        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]

        left_up = (left_wrist.y < left_shoulder.y - margin)
        right_up = (right_wrist.y < right_shoulder.y - margin)

        if left_up and right_up:
            raw_state = "BOTH HANDS UP"
        elif left_up:
            raw_state = "LEFT HAND UP"
        elif right_up:
            raw_state = "RIGHT HAND UP"
        else:
            raw_state = "POSE DETECTED"

        self._state_history.append(raw_state)
        smoothed_state = self._get_smoothed_state()

        self._draw_hud(annotated, True, smoothed_state, avg_conf)
        return annotated, True, smoothed_state, avg_conf

    def _get_smoothed_state(self) -> str:
        """Majority voting over last 3 frames to avoid single-frame flickering."""
        if not self._state_history:
            return "NO POSE"
        counts = {}
        for s in self._state_history:
            counts[s] = counts.get(s, 0) + 1
        return max(counts, key=lambda k: (counts[k], self._state_history[-1] == k))

    def _draw_hud(self, img: np.ndarray, detected: bool, state: str, conf: float) -> None:
        """Overlays a clean diagnostic badge directly on top of the annotated image."""
        h, w = img.shape[:2]

        cv2.rectangle(img, (0, 0), (w, 24), (15, 12, 10), -1)

        if state == "BOTH HANDS UP":
            color = (0, 220, 255)
        elif "HAND UP" in state:
            color = (255, 180, 0)
        elif detected:
            color = (80, 220, 100)
        else:
            color = (100, 100, 220)

        cv2.putText(img, state, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)

        if detected:
            conf_text = f"Conf: {conf:.0f}%"
            cv2.putText(img, conf_text, (w - 85, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    def generate_pose_stream(self, max_fps: int = 30) -> Generator[bytes, None, None]:
        """Yields multipart MJPEG chunks of annotated pose frames to Flask."""
        interval = 1.0 / max_fps
        last_sent_time = 0.0

        while self._running:
            self._new_annotated_event.wait(timeout=interval)
            self._new_annotated_event.clear()

            now = time.time()
            with self._lock:
                frame_time = self.last_process_time
                jpeg_bytes = self._annotated_jpeg

            if jpeg_bytes and frame_time != last_sent_time:
                last_sent_time = frame_time
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
                )

    def get_status_dict(self) -> dict:
        """Returns JSON status dictionary with pose metrics."""
        now = time.time()
        with self._lock:
            detected = self.pose_detected
            state = self.pose_state
            conf = self.confidence_score
            p_fps = self.pose_fps
            age = round(now - self.last_process_time, 2) if self.last_process_time > 0 else None

        cam_status = self.camera_worker.get_status_dict()

        return {
            "pose_detected": detected,
            "pose_state": state,
            "confidence": conf,
            "pose_fps": p_fps,
            "camera_fps": cam_status.get("camera_fps", 0.0),
            "camera_connected": cam_status.get("connected", False),
            "resolution": cam_status.get("resolution", "N/A"),
            "last_pose_age_seconds": age,
        }
