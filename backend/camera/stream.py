"""Backwards-compatibility module pointing to camera_worker."""
from .camera_worker import CameraWorker, CameraWorker as CameraStream

__all__ = ["CameraWorker", "CameraStream"]
