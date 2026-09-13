"""Camera streaming and worker module for ESP32-CAM."""
from .camera_worker import CameraWorker

# Keep CameraStream alias for full backwards compatibility
CameraStream = CameraWorker

__all__ = ["CameraWorker", "CameraStream"]
