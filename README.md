# Safety Driver AI Service

Independent Python AI backend for the **Safety Driver** university graduation project — an in-cabin driver monitoring system.

This service analyzes camera frames from **inside the vehicle cabin** and returns structured detection results for the main **.NET backend**.

It does **not** handle authentication, trips, databases, alerts, or frontend real-time fan-out. Those remain .NET responsibilities.

---

## 1. Project overview

Safety Driver monitors the driver (and prepares for passengers) using computer vision:

| Feature | Method | Temporal gate |
|--------|--------|---------------|
| Face detection + landmarks | MediaPipe Face Landmarker (Tasks API) | Per frame |
| Drowsiness | Eye Aspect Ratio (EAR) | Consecutive closed-eye frames + duration |
| Yawning | Mouth Aspect Ratio (MAR) | Consecutive open-mouth frames + duration |
| Phone presence | YOLO (COCO cell phone) | Per frame |
| Phone usage | Phone near driver face | Consecutive near-face frames |
| Distraction | Head pose (yaw/pitch) | Consecutive look-away frames + duration |
| Abnormal movement | Landmark displacement heuristic | Consecutive high-motion frames |

**Scope limits**

- In-cabin only (no road, lanes, signs, or navigation)
- Abnormal movement is a **behavioral heuristic**, not a medical diagnosis
- Confidence values come from real metrics/models — never fabricated

---

## 2. Technology stack

- Python 3.12+ (developed/tested against the environment available on the machine; see Prerequisites)
- FastAPI + Uvicorn
- OpenCV (`opencv-python-headless`)
- MediaPipe Face Mesh
- Ultralytics YOLO
- NumPy, Pydantic, Pydantic Settings
- Pytest + HTTPX

---

## 3. Architecture

```
app/
  main.py                 # FastAPI app + lifespan model loading
  api/v1/endpoints/       # HTTP layer only
  core/                   # Settings + logging
  schemas/                # Pydantic API contracts
  services/               # Detection business logic (independent services)
  models/model_loader.py  # Centralized model load once at startup
  utils/                  # Image/video/geometry helpers
models/                   # Local weight files (gitignored)
tests/
```

**Design choices**

1. **Separation of concerns** — API validates/uploads; services infer; schemas define the .NET contract.
2. **Models load once** — via FastAPI lifespan; missing optional YOLO weights degrade gracefully.
3. **Session-based temporal state** — pass the same `session_id` (form or `X-Session-Id`) across frames.
4. **Independent services** — each detector can be unit-tested and toggled via warnings when unavailable.

---

## 4. Prerequisites

- Windows / Linux / macOS
- Python **3.12 recommended** (3.13/3.14 may work if wheels exist for MediaPipe & Ultralytics)
- pip + venv
- ~500MB disk for YOLO nano weights (downloaded on first use)

### Check Python

```powershell
python --version
py --list
```

If Python 3.12 is missing on Windows:

1. Install from https://www.python.org/downloads/release/python-31210/
2. Enable **Add python.exe to PATH**
3. Recreate the venv with `py -3.12 -m venv .venv`

---

## 5. Python environment setup

```powershell
cd D:\Graduation-Project\safety-driver-ai-service
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

---

## 6. Dependency installation

```powershell
pip install -r requirements.txt
```

If MediaPipe or Ultralytics fails on your Python version, install **Python 3.12** and recreate the venv (see above).

---

## 7. Environment configuration

```powershell
Copy-Item .env.example .env
```

Key variables (all documented in `.env.example`):

- `EAR_THRESHOLD`, `DROWSINESS_CONSECUTIVE_FRAMES`
- `MAR_THRESHOLD`, `YAWNING_CONSECUTIVE_FRAMES`
- `PHONE_DETECTION_CONFIDENCE`, `PHONE_USAGE_*`
- `HEAD_YAW_THRESHOLD_DEGREES`, `DISTRACTION_*`
- `YOLO_MODEL_PATH`
- `MAX_IMAGE_SIZE_BYTES`

---

## 8. Model weights setup

### MediaPipe Face Landmarker

Required file: `models/face_landmarker.task`

Downloaded automatically on first startup if missing (needs network), from:

`https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`

Manual download (PowerShell):

```powershell
Invoke-WebRequest `
  -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" `
  -OutFile "models\face_landmarker.task"
```

> Note: MediaPipe 0.10.x on current Python builds uses the **Tasks API** (not the legacy `mp.solutions` Face Mesh API).

### YOLO (phone detection)

Default path: `models/yolov8n.pt`

On first YOLO load, Ultralytics downloads `yolov8n.pt` if missing (needs network).

Manual option:

```powershell
# With venv active and ultralytics installed
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
# Then move/copy yolov8n.pt into models/
```

Large `.pt` files are **gitignored** — do not commit them.

If YOLO cannot load, the API still runs; `phone_*` detections return `detected=false` with a warning.

---

## 9. Running the FastAPI server

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Or:

```powershell
python -m app.main
```

---

## 10. Swagger documentation

With the server running:

- Swagger UI: http://127.0.0.1:8000/docs
- OpenAPI JSON: http://127.0.0.1:8000/openapi.json
- ReDoc: http://127.0.0.1:8000/redoc

---

## 11. API endpoints

### `GET /`

Service info.

### `GET /health`

Liveness + model load status (`healthy` / `degraded`).

### `POST /api/v1/analyze-frame`

Multipart form:

| Field | Type | Description |
|-------|------|-------------|
| `file` | file | JPEG/PNG/WebP/BMP cabin frame |
| `session_id` | string (optional) | Temporal session key |
| `fps` | float (optional) | Source FPS hint |

Header alternative: `X-Session-Id`

**curl example**

```powershell
curl -X POST "http://127.0.0.1:8000/api/v1/analyze-frame" `
  -F "file=@cabin_frame.jpg" `
  -F "session_id=trip-123" `
  -F "fps=10"
```

**Response contract (simplified)**

```json
{
  "success": true,
  "timestamp": "2026-11-15T22:00:00+00:00",
  "session_id": "trip-123",
  "frame_index": 1,
  "detections": [
    {
      "type": "drowsiness",
      "detected": false,
      "confidence": 0.82,
      "severity": "none",
      "metadata": {
        "ear": 0.29,
        "eye_status": "open",
        "consecutive_closed_frames": 0,
        "duration_seconds": 0.0
      }
    }
  ],
  "processing_time_ms": 45.2,
  "warnings": []
}
```

`confidence` may be `null` when a metric/model is unavailable — never a fake placeholder.

### `POST /api/v1/analyze-video`

Practical offline helper: samples up to `max_frames` and runs sequential `analyze-frame` logic. Prefer `analyze-frame` for live ESP32 / camera streams.

---

## 12. Running tests

```powershell
.\.venv\Scripts\Activate.ps1
pytest -v
```

Tests mock MediaPipe/YOLO for unit and API contract coverage. They do **not** require real weight downloads.

---

## 13. Integration with the .NET backend

Recommended flow:

1. .NET receives camera frames (ESP32 / mobile / hub).
2. .NET calls `POST /api/v1/analyze-frame` with the image bytes.
3. Reuse one `session_id` per trip/stream for temporal detectors.
4. Persist detections, apply business risk rules, notify clients — in .NET.

**HTTP conventions**

| Code | Meaning |
|------|---------|
| 200 | Analysis OK |
| 400 | Invalid image/video |
| 413 | Payload too large |
| 422 | Validation error |
| 500 | Processing/internal error |

Error envelope (returned as the JSON body, not nested under `detail`):

```json
{
  "success": false,
  "timestamp": "...",
  "error": { "code": "INVALID_IMAGE", "message": "...", "details": null }
}
```

Configure base URL via .NET settings, e.g. `http://localhost:8000`.

No auth is enforced here — place the AI service on a private network / sidecar and let .NET gate public access.

---

## 14. Current limitations and future improvements

**Current**

- Single primary face treated as driver candidate (passenger roles are labeled, not fully tracked)
- Phone usage is a proximity heuristic, not hand-phone pose classification
- Abnormal movement is landmark-delta based only
- In-memory sessions (lost on process restart)
- No WebSocket streaming yet (intentionally deferred)

**Future**

- Multi-face driver/passenger assignment with seat zones
- Hand landmarks + phone interaction model
- Redis-backed session state for multi-instance deploy
- Optional WebSocket/gRPC once .NET streaming requirements are fixed
- Quantized models for edge devices

---

## License / academic use

Built for a university graduation project. Use responsibly; detection outputs are assistive signals, not safety-certified guarantees.
