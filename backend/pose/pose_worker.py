"""Pose estimation and gesture diagnostic worker using MediaPipe PoseLandmarker.
Optimized for distant detection in low-resolution (320x240 / 640x480) indoor environments.
Features real-time temporal stabilization with dropout grace periods and cooldown.
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

# =====================================================================
# TEMPORAL GESTURE STABILIZATION CONSTANTS
# =====================================================================
RAISED_HAND_CONFIRM_SECONDS = 0.8
GESTURE_LOST_GRACE_SECONDS = 0.6
POSE_LOST_GRACE_SECONDS = 0.6
GESTURE_COOLDOWN_SECONDS = 2.0

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
    applies time-based temporal stabilization (with dropout grace and cooldown),
    and caches annotated JPEG frames for web streaming.
    """

    def __init__(self, camera_worker, model_path: Optional[str] = None):
        self.camera_worker = camera_worker

        if model_path is None:
            model_path = str(Path(__file__).parent / "models" / "pose_landmarker_heavy.task")
        self.model_path = model_path

        self._detector: Optional[vision.PoseLandmarker] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._new_annotated_event = threading.Event()

        # State
        self.pose_detected = False
        self.confidence_score = 0.0
        self.pose_fps = 0.0
        self.last_process_time = 0.0

        # Temporal stabilization state (time.monotonic based)
        self._candidate_gesture: Optional[str] = None
        self._candidate_started_at: float = 0.0
        self._gesture_last_seen: float = 0.0
        self._pose_last_seen: float = 0.0
        self._confirmed_gesture: Optional[str] = None
        self._confirmed_at: float = 0.0
        self._cooldown_until: float = 0.0

        # Public states
        self.raw_state: str = "NO POSE"
        self.stable_state: str = "NO POSE"
        self.pose_state: str = "NO POSE"  # Backwards-compatible alias for stable_state
        self.hold_duration: float = 0.0
        self.in_cooldown: bool = False
        self.cooldown_remaining: float = 0.0

        self._pose_frame_times = deque()

        # Cached annotated frame
        self._annotated_jpeg: Optional[bytes] = None
        self._last_camera_frame_time = 0.0

    def start(self) -> "PoseWorker":
        """Initialize MediaPipe and start processing thread."""
        if self._running:
            return self

        logger.info("Initializing MediaPipe PoseLandmarker from: %s", self.model_path)
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=0.50,
            min_pose_presence_confidence=0.50,
            min_tracking_confidence=0.50,
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

            # Process frame with MediaPipe and temporal stabilizer
            (
                annotated_frame,
                detected,
                raw_st,
                stable_st,
                hold_dur,
                in_cd,
                cd_rem,
                conf,
            ) = self._process_frame(frame)
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
                self.raw_state = raw_st
                self.stable_state = stable_st
                self.pose_state = stable_st  # Backwards compatibility
                self.hold_duration = hold_dur
                self.in_cooldown = in_cd
                self.cooldown_remaining = cd_rem
                self.confidence_score = conf
                self.pose_fps = pose_fps
                self.last_process_time = now
                self._annotated_jpeg = jpeg_bytes

            self._new_annotated_event.set()

    def _update_temporal_state(self, raw_state: str, now: float) -> Tuple[str, float, float]:
        """Stabilizes raw gesture classifications using elapsed real time (monotonic clock)
        and short dropout grace periods.
        
        Returns:
            (stable_state, hold_duration, cooldown_remaining)
        """
        # 1. Pose detection tracking
        if raw_state != "NO POSE":
            self._pose_last_seen = now

        is_raised_hand = raw_state in ("LEFT HAND UP", "RIGHT HAND UP", "BOTH HANDS UP")
        in_cooldown = (now < self._cooldown_until)
        cooldown_remaining = max(0.0, self._cooldown_until - now) if in_cooldown else 0.0

        # 2. Check pose dropout grace
        pose_grace_expired = (now - self._pose_last_seen > POSE_LOST_GRACE_SECONDS) if self._pose_last_seen > 0 else True
        if raw_state == "NO POSE" and pose_grace_expired:
            self._candidate_gesture = None
            self._candidate_started_at = 0.0
            return "NO POSE", 0.0, cooldown_remaining

        # 3. Handle Cooldown active period
        if in_cooldown:
            # During cooldown, do not start new gesture candidates
            self._candidate_gesture = None
            self._candidate_started_at = 0.0

            # Show confirmed state while user still has hand raised or for 1.0s after confirmation
            if is_raised_hand or (now - self._confirmed_at < 1.0):
                stable = f"{self._confirmed_gesture} - CONFIRMED" if self._confirmed_gesture else "POSE DETECTED"
            else:
                stable = "POSE DETECTED"
            return stable, 0.0, cooldown_remaining

        # 4. Normal operation (not in cooldown)
        if is_raised_hand:
            self._gesture_last_seen = now
            if self._candidate_gesture == raw_state:
                # Same gesture continues
                hold_duration = now - self._candidate_started_at
                if hold_duration >= RAISED_HAND_CONFIRM_SECONDS:
                    # CONFIRMED!
                    self._confirmed_gesture = self._candidate_gesture
                    self._confirmed_at = now
                    self._cooldown_until = now + GESTURE_COOLDOWN_SECONDS
                    self._candidate_gesture = None
                    self._candidate_started_at = 0.0
                    return f"{self._confirmed_gesture} - CONFIRMED", RAISED_HAND_CONFIRM_SECONDS, GESTURE_COOLDOWN_SECONDS
                else:
                    return f"{self._candidate_gesture} - CANDIDATE", hold_duration, 0.0
            else:
                # New candidate starts (or user switched hand)
                self._candidate_gesture = raw_state
                self._candidate_started_at = now
                return f"{self._candidate_gesture} - CANDIDATE", 0.0, 0.0
        else:
            # raw_state is POSE DETECTED or temporary NO POSE within grace
            if self._candidate_gesture is not None:
                if (now - self._gesture_last_seen) <= GESTURE_LOST_GRACE_SECONDS:
                    # Candidate survives short gesture dropout
                    hold_duration = now - self._candidate_started_at
                    return f"{self._candidate_gesture} - CANDIDATE", hold_duration, 0.0
                else:
                    # Dropout exceeded grace period -> reset candidate
                    self._candidate_gesture = None
                    self._candidate_started_at = 0.0

            return ("POSE DETECTED" if not pose_grace_expired else "NO POSE"), 0.0, 0.0

    def _process_frame(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, bool, str, str, float, bool, float, float]:
        """Detect pose, draw skeleton, determine gesture state, apply temporal stabilization,
        and overlay diagnostic HUD text.
        
        Returns:
            (annotated_frame, detected, raw_state, stable_state, hold_duration, in_cooldown, cd_remaining, conf)
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]

        # 1. 2x bicubic upscale for receptive field matching with natural RGB gradients
        upscaled = cv2.resize(frame, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        rgb_frame = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self._detector.detect(mp_image)

        # MULTI-SCALE FALLBACK FOR DISTANT DETECTION
        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            # 2D Grid Crops for extreme distance detection
            # Format: (name, x_start, x_end, y_start, y_end)
            crops = [
                ("left_half",   0,              int(w * 0.6), 0, h),
                ("right_half",  int(w * 0.4),   w,            0, h),
                # Distant people appear higher in the frame due to camera angle, crop top section for max zoom
                ("top_left",    0,              int(w * 0.6), 0, int(h * 0.75)),
                ("top_right",   int(w * 0.4),   w,            0, int(h * 0.75)),
            ]

            for name, x_start, x_end, y_start, y_end in crops:
                cw = x_end - x_start
                ch = y_end - y_start

                # Crop image in 2D
                crop_img = frame[y_start:y_end, x_start:x_end]

                # 2x upscale crop (gives MediaPipe huge optical detail for small subjects)
                crop_up = cv2.resize(crop_img, (cw * 2, ch * 2), interpolation=cv2.INTER_CUBIC)
                crop_rgb = cv2.cvtColor(crop_up, cv2.COLOR_BGR2RGB)
                crop_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)

                crop_results = self._detector.detect(crop_mp)
                if crop_results.pose_landmarks and len(crop_results.pose_landmarks) > 0:
                    # Adjust landmarks back to full frame space for both X and Y
                    for landmarks in crop_results.pose_landmarks:
                        for lm in landmarks:
                            lm.x = (lm.x * cw + x_start) / w
                            lm.y = (lm.y * ch + y_start) / h
                    results = crop_results
                    break

        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            now_mono = time.monotonic()
            raw_state = "NO POSE"
            stable_state, hold_duration, cd_remaining = self._update_temporal_state(raw_state, now_mono)
            in_cd = (now_mono < self._cooldown_until)
            is_detected = (stable_state != "NO POSE")
            self._draw_hud(annotated, is_detected, raw_state, stable_state, hold_duration, in_cd, cd_remaining, 0.0)
            return annotated, is_detected, raw_state, stable_state, hold_duration, in_cd, cd_remaining, 0.0

        landmarks = results.pose_landmarks[0]

        # Average visibility/confidence for essential landmarks
        conf_values = [
            landmarks[idx].visibility
            for idx in ESSENTIAL_LANDMARKS
            if hasattr(landmarks[idx], "visibility") and landmarks[idx].visibility is not None
        ]
        avg_conf = round(float(np.mean(conf_values)) * 100, 1) if conf_values else 0.0

        # Draw skeleton connections (lines) - threshold 0.08 for distant landmarks
        for idx1, idx2 in SKELETON_CONNECTIONS:
            lm1 = landmarks[idx1]
            lm2 = landmarks[idx2]

            v1 = getattr(lm1, "visibility", 1.0) or 1.0
            v2 = getattr(lm2, "visibility", 1.0) or 1.0

            if v1 > 0.08 and v2 > 0.08:
                x1, y1 = int(lm1.x * w), int(lm1.y * h)
                x2, y2 = int(lm2.x * w), int(lm2.y * h)
                cv2.line(annotated, (x1, y1), (x2, y2), (255, 200, 0), 2, cv2.LINE_AA)

        # Draw landmark points (circles)
        for idx in ESSENTIAL_LANDMARKS:
            lm = landmarks[idx]
            v = getattr(lm, "visibility", 1.0) or 1.0
            if v > 0.08:
                cx, cy = int(lm.x * w), int(lm.y * h)
                if idx in (15, 16):  # Wrists highlighted in cyan/white
                    cv2.circle(annotated, (cx, cy), 5, (0, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(annotated, (cx, cy), 7, (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    cv2.circle(annotated, (cx, cy), 4, (16, 220, 100), -1, cv2.LINE_AA)

        # Raised-hand diagnostic (perspective invariant relative to chin/nose)
        nose = landmarks[0]
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]

        left_v = getattr(left_wrist, "visibility", 1.0) or 1.0
        right_v = getattr(right_wrist, "visibility", 1.0) or 1.0

        # An arm is strictly raised if wrist is above chin level (nose.y + 0.05)
        # and has good visibility (> 0.65)
        left_up = (left_v > 0.65) and (left_wrist.y < nose.y + 0.05)
        right_up = (right_v > 0.65) and (right_wrist.y < nose.y + 0.05)

        if left_up and right_up:
            raw_state = "BOTH HANDS UP"
        elif left_up:
            raw_state = "LEFT HAND UP"
        elif right_up:
            raw_state = "RIGHT HAND UP"
        else:
            raw_state = "POSE DETECTED"

        # Highlight raised wrists with on-screen target circle and text badge
        if left_up:
            cx, cy = int(left_wrist.x * w), int(left_wrist.y * h)
            cv2.circle(annotated, (cx, cy), 8, (0, 220, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, "L-UP", (cx - 16, max(cy - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1, cv2.LINE_AA)

        if right_up:
            cx, cy = int(right_wrist.x * w), int(right_wrist.y * h)
            cv2.circle(annotated, (cx, cy), 8, (0, 220, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, "R-UP", (cx - 16, max(cy - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1, cv2.LINE_AA)

        # Temporal gesture stabilization
        now_mono = time.monotonic()
        stable_state, hold_duration, cd_remaining = self._update_temporal_state(raw_state, now_mono)
        in_cd = (now_mono < self._cooldown_until)
        is_detected = True

        self._draw_hud(annotated, is_detected, raw_state, stable_state, hold_duration, in_cd, cd_remaining, avg_conf)
        return annotated, is_detected, raw_state, stable_state, hold_duration, in_cd, cd_remaining, avg_conf

    def _draw_hud(
        self,
        img: np.ndarray,
        detected: bool,
        raw_state: str,
        stable_state: str,
        hold_duration: float,
        in_cooldown: bool,
        cooldown_remaining: float,
        conf: float,
    ) -> None:
        """Overlays a clean diagnostic badge directly on top of the annotated image,
        displaying both Raw and Stabilized state, hold progress, and cooldown.
        """
        h, w = img.shape[:2]

        cv2.rectangle(img, (0, 0), (w, 26), (15, 12, 10), -1)

        if "CONFIRMED" in stable_state:
            color = (0, 255, 128)      # Emerald
        elif "CANDIDATE" in stable_state:
            color = (0, 190, 255)      # Amber
        elif detected:
            color = (80, 220, 100)     # Light Green
        else:
            color = (100, 100, 220)    # Soft Red

        # Stabilized state
        cv2.putText(img, stable_state, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 2, cv2.LINE_AA)

        # Raw state
        raw_str = f"Raw: {raw_state}"
        cv2.putText(img, raw_str, (max(w - 230, 150), 18), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (170, 170, 170), 1, cv2.LINE_AA)

        # Hold or Cooldown info
        if "CANDIDATE" in stable_state:
            hold_str = f"{min(hold_duration, RAISED_HAND_CONFIRM_SECONDS):.2f}/{RAISED_HAND_CONFIRM_SECONDS:.2f}s"
            cv2.putText(img, hold_str, (max(w - 110, 220), 18), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 220, 255), 1, cv2.LINE_AA)
            # Progress line at bottom edge
            progress = min(1.0, hold_duration / RAISED_HAND_CONFIRM_SECONDS)
            cv2.line(img, (0, 25), (int(w * progress), 25), (0, 220, 255), 2)
        elif in_cooldown and cooldown_remaining > 0:
            cd_str = f"CD: {cooldown_remaining:.1f}s"
            cv2.putText(img, cd_str, (max(w - 75, 220), 18), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 180, 0), 1, cv2.LINE_AA)
        elif detected and conf > 0:
            conf_str = f"{conf:.0f}%"
            cv2.putText(img, conf_str, (w - 40, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 180, 180), 1, cv2.LINE_AA)

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
        """Returns JSON status dictionary with stabilized and raw pose metrics."""
        now = time.time()
        with self._lock:
            detected = self.pose_detected
            raw_st = self.raw_state
            stable_st = self.stable_state
            hold_dur = self.hold_duration
            in_cd = self.in_cooldown
            cd_rem = self.cooldown_remaining
            conf = self.confidence_score
            p_fps = self.pose_fps
            age = round(now - self.last_process_time, 2) if self.last_process_time > 0 else None

        cam_status = self.camera_worker.get_status_dict()

        return {
            "pose_detected": detected,
            "raw_state": raw_st,
            "stable_state": stable_st,
            "pose_state": stable_st,  # Backwards compatibility
            "candidate_gesture": self._candidate_gesture,
            "hold_duration": round(hold_dur, 2),
            "hold_target": RAISED_HAND_CONFIRM_SECONDS,
            "in_cooldown": in_cd,
            "cooldown_remaining": round(cd_rem, 1),
            "confidence": conf,
            "pose_fps": p_fps,
            "camera_fps": cam_status.get("camera_fps", 0.0),
            "camera_connected": cam_status.get("connected", False),
            "resolution": cam_status.get("resolution", "N/A"),
            "last_pose_age_seconds": age,
        }
