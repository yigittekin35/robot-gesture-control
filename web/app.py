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

from backend.camera import CameraStream

app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
)

stream_url = os.getenv("ESP32_STREAM_URL", "http://192.168.178.67:81/stream")
camera = CameraStream(stream_url=stream_url)
camera.start()


@app.route("/")
def index():
    """Renders the main dashboard page."""
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """Video streaming route. Returns multipart MJPEG."""
    return Response(
        camera.generate_mjpeg_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/status")
def status():
    """Returns JSON with current camera connection status and metrics."""
    return jsonify(camera.get_status_dict())


if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "False").lower() in ("true", "1")

    print(f"Starting Robot Gesture Control gateway on http://127.0.0.1:{port}")
    print(f"Streaming target ESP32-CAM: {stream_url}")
    try:
        app.run(host=host, port=port, debug=debug, threaded=True)
    finally:
        camera.stop()
