import logging
import os
import sys
from pathlib import Path
from flask import Flask, Response, jsonify, render_template
from dotenv import load_dotenv

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env from project root
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from backend.camera import CameraWorker
from backend.pose import PoseWorker

app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
)

stream_url = os.getenv("ESP32_STREAM_URL", "http://192.168.178.67:81/stream")
camera_worker = CameraWorker(stream_url=stream_url)
camera_worker.start()

# Initialize PoseWorker attached to the single CameraWorker instance
pose_worker = PoseWorker(camera_worker=camera_worker)
pose_worker.start()


@app.route("/")
def index():
    """Renders the main live dashboard page."""
    return render_template("index.html")


@app.route("/calibrate")
def calibrate():
    """Renders the Calibration Wizard v1 page."""
    return render_template("calibrate.html")


@app.route("/pose-test")
def pose_test():
    """Renders the Pose Detection & Distance Validation test page."""
    return render_template("pose_test.html")


@app.route("/video_feed")
def video_feed():
    """Raw camera video streaming route. Returns multipart MJPEG from CameraWorker."""
    return Response(
        camera_worker.generate_mjpeg_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/pose_feed")
def pose_feed():
    """Pose annotated video streaming route. Returns multipart MJPEG from PoseWorker."""
    return Response(
        pose_worker.generate_pose_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/status")
def status():
    """Returns JSON with current camera connection status and metrics."""
    return jsonify(camera_worker.get_status_dict())


@app.route("/pose_status")
def pose_status():
    """Returns JSON with current pose detection state and diagnostic metrics."""
    return jsonify(pose_worker.get_status_dict())


if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "False").lower() in ("true", "1")

    print(f"Starting Robot Gesture Control gateway on http://127.0.0.1:{port}")
    try:
        app.run(host=host, port=port, debug=debug, threaded=True)
    finally:
        pose_worker.stop()
        camera_worker.stop()
