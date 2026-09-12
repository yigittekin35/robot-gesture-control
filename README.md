# robot-gesture-control

A gesture-based robot control pipeline connecting an ESP32-CAM video stream through a local gateway on Windows (and later Raspberry Pi) to control a Xiaomi robot vacuum.

## Architecture Overview

```
ESP32-CAM (http://192.168.178.67:81/stream)
    │  MJPEG over Wi-Fi
    ▼
Windows / Raspberry Pi Gateway
    │  OpenCV VideoCapture
    ▼
Flask Streaming Web Service (http://127.0.0.1:5000)
    │  MJPEG multipart stream
    ▼
Web Browser (Live Dashboard)
```

## Project Structure

```
robot-gesture-control/
├── .env                  # Local environment configuration (gitignored)
├── .env.example          # Template environment file
├── .gitignore            # Git ignore rules
├── requirements.txt      # Python dependencies
├── README.md
├── backend/
│   ├── __init__.py
│   └── camera/
│       ├── __init__.py
│       └── stream.py     # Threaded OpenCV stream consumer with auto-reconnect
└── web/
    ├── app.py            # Flask web server
    └── templates/
        └── index.html    # Dashboard frontend
```

## Quick Start (Windows)

### 1. Activate the Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` if not already present:
```powershell
cp .env.example .env
```
Default configuration:
```ini
ESP32_STREAM_URL=http://192.168.178.67:81/stream
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_DEBUG=False
```

### 3. Run the Application
```powershell
.\.venv\Scripts\python.exe web/app.py
```

Open your browser at:
**[http://127.0.0.1:5000](http://127.0.0.1:5000)**
