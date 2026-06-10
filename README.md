# AI_CAM — Industrial VIN Detection & OCR (MES)

Real-time system that reads **17-digit VINs** from metal car bodies on a conveyor, using a **SICK LECTOR652** camera, **YOLOv8n** plate detection, **EasyOCR**, **SQLite**, and a **FastAPI + MES-style web UI**.

Built on the proven `lector652_pipeline` camera code: CoLa-A control on **port 2111**, `mLIStart` live BLOB streaming on **port 2113**, fast 8bpp-BMP→numpy decode, an async OCR queue, and anti-duplicate VIN logic. Frames are processed in their **raw, original orientation** (no rotation).

---

## Architecture & Tech Stack

```
SICK Lector652
   │  CoLa-A (ASCII, STX/ETX) — TCP 2111   →  connect handshake + sMN mLIStart 0
   │  BLOB image stream        — TCP 2113   →  camera auto-pushes BMP frames
   ▼
backend/camera/
   camera_client.py  Lector652Client · mLIStart live push · 16B/STX frame parse · decode_bmp() 8bpp→numpy (raw)
   (cola_client.py / blob_stream.py are deprecated shims that re-export the above)
   ▼
backend/pipeline.py  (orchestrator, background stream thread)
   read_stream_frame → decode_bmp (raw, no rotation) → live frame (MJPEG)
            → YOLOv8n detect → draw bbox → high-conf ROI → OCR queue (non-blocking)
            → low-conf detect (>0.40) → dataset auto-collection (non-blocking)
   ▼
backend/ai/
   detector.py      YOLOv8n plate detection + teal bounding boxes
   ocr_worker.py    EasyOCR background queue · light CLAHE · VIN validation · anti-duplicate
   ▼
backend/database/db.py   SQLite: id, timestamp, detected_vin, confidence, image_path
   ▼
backend/server.py   FastAPI · MJPEG /video_feed · REST API · CSV/Excel export
frontend/           Dashboard (live + logs) · History (table + export) — Deep Blue/Teal MES theme
```

**Stack:** Python 3.10+ · FastAPI · Uvicorn · OpenCV · Ultralytics (YOLOv8n) · EasyOCR · SQLite · openpyxl · vanilla HTML/CSS/JS.

### Directory structure

```
AI_CAM/
├── run.py                      # entry point → python run.py
├── requirements.txt
├── README.md
├── TRAINING_GUIDE.md
├── backend/
│   ├── config.py               # all settings (IP, ports, thresholds)
│   ├── logger.py               # rotating logs + in-memory ring buffer for UI
│   ├── pipeline.py             # orchestrator (camera → AI → DB)
│   ├── server.py               # FastAPI app + MJPEG + API
│   ├── camera/
│   │   ├── cola_client.py
│   │   └── blob_stream.py
│   ├── ai/
│   │   ├── detector.py
│   │   └── ocr_worker.py
│   └── database/
│       └── db.py
├── frontend/
│   ├── templates/  dashboard.html · history.html
│   └── static/     css/style.css · js/dashboard.js · js/history.js
├── models/         (optional stock weights; trained model lives under runs/)
├── runs/detect/vin_plate_model/weights/best.pt   (trained YOLOv8n — loaded directly)
├── data/           ai_cam.db · crops/
└── logs/           ai_cam.log
```

---

## Setup

```bash
cd AI_CAM
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
# source venv/bin/activate

pip install -r requirements.txt
```

> First run downloads EasyOCR + YOLO weights (a few hundred MB). For GPU, install the CUDA build of PyTorch from pytorch.org and set `DETECTION.device = "cuda:0"`, `OCR.use_gpu = True` in `config.py`.

---

## Configure the camera (SOPAS ET — from your lector652 setup)

1. Connect the Lector652 over Ethernet; PC on the same subnet (camera default `192.168.0.1:2111`).
2. In **SOPAS ET**, log in (`client` / `client`) and enable **Image / BLOB transfer** (TCP push) on **port 2113**.
3. Live streaming is driven by the app via `sMN mLIStart 0` (camera auto-pushes frames) — no per-frame trigger needed. Save settings to EEPROM.
4. Confirm values in `backend/config.py` → `CAMERA` (`ip`, `cola_port`, `blob_port`, `password`). The camera IP is also set live from the dashboard.

---

## Run

```bash
python run.py
```

Open **http://localhost:8000/dashboard**

- **Connect Camera** → opens CoLa-A + BLOB stream (`mLIStart`), raw live video starts.
- **Start Processing** → YOLO boxes + OCR + DB writes + dataset auto-collection.
- **Stop** / **Disconnect** as needed.
- **History** page → sortable table, live auto-refresh, **Export CSV / Excel**.

---

## How the key requirements are met

| Requirement | Where |
|---|---|
| CoLa-A + mLIStart live stream (exact working code) | `camera_client.py` (`Lector652Client`) |
| BMP→numpy decode → MJPEG (raw frame, no rotation) | `camera_client.decode_bmp`, `pipeline._stream_loop`, `server.py` `/video_feed` |
| Dataset auto-collection (frame + YOLO label) | `ai/dataset_collector.py`, `pipeline._stream_loop` |
| YOLOv8n detect + live bounding box | `detector.py`, `pipeline._on_frame` |
| ROI crop on high confidence | `detector.crop_roi`, `best_detection` |
| EasyOCR in separate background queue | `ocr_worker.py` (Queue + thread) |
| VIN validation (17 chars, no I/O/Q) | `ocr_worker.is_valid_vin` |
| Anti-duplicate (recent identical VIN) | `ocr_worker._is_duplicate` |
| SQLite: id, timestamp, VIN, confidence, image | `database/db.py` |
| Real-time logging (connect/detect/OCR/errors) | `logger.py` + UI log panel |
| MES UI, Deep Blue #1F3C88 / Teal #1ABC9C | `frontend/` |
| Camera disconnect handling | auto-reconnect in both camera modules |

See **TRAINING_GUIDE.md** to train your plate detector; the app loads `runs/detect/vin_plate_model/weights/best.pt` directly.
