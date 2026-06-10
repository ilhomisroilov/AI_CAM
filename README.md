# AI_CAM — Industrial VIN / Serial-Number Vision System

Real-time recognition of **engraved / etched 17-character VINs** stamped on metal car bodies on a moving conveyor, using a **SICK Lector 652** industrial camera, **YOLOv8n** plate detection, and **PaddleOCR** text recognition — served through a **FastAPI + MES-style web dashboard** with live video, a results history, and lightweight authentication.

> Status: working, production-like. OCR engine migrated from EasyOCR → **PaddleOCR** (see `OCR_MIGRATION_REPORT.md`). This README reflects the codebase **after** that migration and the trigger/auth/GPU work.

---

## Project Overview

**Purpose.** Read the VIN/serial that is engraved or dot-peen marked into a metal vehicle body as it passes a fixed camera station, validate it against the standard VIN format, and persist each unique read (with a cropped proof image) to a local database — without an operator manually typing numbers.

**Industrial use case.** Automotive body/weld shop traceability: a SICK Lector 652 mounted over the line streams frames; YOLO finds the VIN plate region; PaddleOCR reads the characters; the result is logged and shown on a shop-floor dashboard accessible from any PC on the factory LAN.

**Supported input formats.**
- **Live:** raw BMP frames pushed by the SICK Lector 652 over its BLOB socket (`mLIStart` live mode, 8-bit grayscale). Frames are decoded straight to NumPy — no rotation, raw orientation.
- **Offline / benchmarking:** standard image files (`.jpg`, `.jpeg`, `.png`, `.bmp`) via `tools/benchmark_ocr.py` and the detection/OCR functions (they operate on NumPy arrays, so any decoded image works).

**Supported character set.**
- Uppercase **A–Z** and digits **0–9**.
- VIN validation excludes **I, O, Q** (ISO 3779) and requires exactly **17** characters: regex `^[A-HJ-NPR-Z0-9]{17}$`.

---

## Features

- **YOLOv8n detection** of the `vin_plate` ROI (single trained class, `best.pt`).
- **ROI extraction** with configurable padding around the detected box.
- **PaddleOCR recognition** (PP-OCRv4), GPU-accelerated when an NVIDIA card is present, with safe CPU fallback.
- **Image preprocessing** tuned for etched metal: grayscale → bicubic upscale → CLAHE → light unsharp mask.
- **Real-time processing** — asynchronous camera stream + background OCR queue; the live MJPEG feed never blocks on OCR.
- **Single-shot trigger** — OCR fires **once per plate event** (confidence ≥ 0.95) with debounce/cooldown, so a plate sitting across many frames is not read repeatedly.
- **Result validation** — strict VIN regex + per-fragment and overall confidence thresholds + anti-duplicate window.
- **Dataset auto-collection ("data loop")** — low-confidence-and-up detections auto-save frame + YOLO-format label for retraining.
- **Web dashboard (MES style)** — live video, Connect/Start/Stop controls, dynamic camera IP, real-time logs, OCR monitoring stats.
- **History + export** — sortable records table with **CSV / Excel** export.
- **Lightweight authentication** — login gate over all pages and APIs (session cookie).
- **Batch processing** — not a built-in service, but `tools/benchmark_ocr.py` runs the OCR stage over a folder of crops (useful for evaluation/A-B testing).

---

## System Architecture

```
                 SICK Lector 652 (industrial camera)
                          │  CoLa-A control  : TCP 2111  (handshake, mLIStart)
                          │  BLOB image push : TCP 2113  (raw 8bpp BMP frames)
                          ▼
        ┌──────────────────────────────────────────────┐
        │  camera_client.Lector652Client                │  decode_bmp() → raw NumPy frame
        └──────────────────────────────────────────────┘   (no rotation)
                          ▼
        ┌──────────────────────────────────────────────┐
        │  pipeline._stream_loop  (background thread)    │
        │    • YOLOv8n detect (best.pt)                  │
        │    • draw bbox → MJPEG live frame              │
        │    • single-shot trigger (conf ≥ 0.95)         │
        │    • dataset auto-collection (conf ≥ 0.40)     │
        └──────────────────────────────────────────────┘
                          ▼  crop ROI (+padding)
        ┌──────────────────────────────────────────────┐
        │  OCRWorker  (background queue, drop-oldest)    │
        │    preprocess → PaddleOCR → assemble L→R       │
        │    → VIN regex verify → confidence gates       │
        │    → anti-duplicate                            │
        └──────────────────────────────────────────────┘
                          ▼  on_result(vin, conf, crop)
        ┌──────────────────────────────────────────────┐
        │  SQLite (vin_records)  +  crop image saved     │
        └──────────────────────────────────────────────┘
                          ▼
        FastAPI  →  /dashboard (live + logs)  ·  /history (table + export)
                    protected by login middleware
```

**Condensed flow:**

```
Camera/Image → YOLO Detection → ROI Extraction → Image Preprocessing → PaddleOCR → Post-processing → Validation → Final Result (DB + UI)
```

---

## OCR Pipeline (stage by stage)

1. **Image acquisition** — `camera_client.py` connects CoLa-A (2111), runs the SOPAS handshake, sends `mLIStart 0`; the camera auto-pushes BMP frames on the BLOB socket (2113). `decode_bmp()` parses the 8-bit BMP directly into a NumPy array (≈3–6× faster than `cv2.imdecode`), in **raw orientation (no rotation)**.
2. **Detection** — `detector.PlateDetector.detect()` runs YOLOv8n (`best.pt`) at `conf_threshold = 0.40` to surface candidate `vin_plate` boxes (low base threshold also feeds dataset collection).
3. **Cropping** — the highest-confidence box that clears `ocr_trigger_conf = 0.95` is cropped with `crop_padding = 8` px. The **single-shot trigger** (`pipeline._handle_trigger`) ensures one crop per plate event: it locks after firing and re-arms only after the plate is absent for `ocr_rearm_absent_frames = 3` frames, with a `ocr_cooldown_sec = 2.0` safety.
4. **Preprocessing** — `OCRWorker._preprocess()`: BGR→gray → bicubic **upscale** to `upscale_height = 96` px → **CLAHE** (`clip 3.0`, `8×8`) → light **unsharp mask**.
5. **OCR recognition** — `_PaddleEngine.read()` runs PaddleOCR (det+rec by default, or rec-only via `paddle_det = False`) and returns uniform `(box, text, conf)` fragments.
6. **Post-processing** — fragments below `min_char_confidence = 0.30` are dropped; the rest are sorted **left-to-right by bbox-x** and concatenated; `normalize_vin()` upper-cases and strips to `A–Z0–9`.
7. **Validation** — `verify_vin()` enforces `^[A-HJ-NPR-Z0-9]{17}$`; the mean confidence must clear `min_confidence = 0.45`; the **anti-duplicate** guard rejects the same VIN within `duplicate_window_sec = 90`. Accepted reads are written to SQLite and the crop image is saved.

---

## Algorithms Used

### YOLOv8n (Ultralytics)
- **Why:** fast, accurate single-class object detector; the *n* (nano) variant runs in real time on CPU and is trivial on GPU.
- **Input:** full camera frame (BGR NumPy). **Output:** list of `(x1,y1,x2,y2,conf)` boxes for `vin_plate`.
- **Benefits:** real-time, robust to lighting/position variation, easy to retrain on a single class.
- **Limitations:** needs a trained `best.pt` for your plates (falls back to stock `yolov8n.pt` with a warning, which won't reliably find VIN plates).

### PaddleOCR (PP-OCRv4 recognizer; optional detector)
- **Why:** strong recognition of dense alphanumeric/industrial text, smaller/faster models and lower memory than the previous EasyOCR engine (see migration report).
- **Input:** preprocessed grayscale/BGR crop. **Output:** `(box, text, confidence)` per text fragment.
- **Benefits:** GPU acceleration, `rec-only` mode ideal for tight single-line crops, lower latency/RAM.
- **Limitations:** downloads models on first run (pre-place for offline factories); detector can occasionally split sparse dot-peen text — prefer `paddle_det = False` for single-line crops.

### CLAHE — Contrast Limited Adaptive Histogram Equalization (active)
- **Why:** lifts local contrast of low-contrast etched characters without blowing out highlights.
- **Input:** grayscale crop. **Output:** contrast-enhanced grayscale. **Benefit:** the single highest-value step for engraved metal. **Limitation:** too-high clip amplifies noise (`clahe_clip` tunable).

### Bicubic upscaling (active)
- **Why:** small etched characters recognize far better when enlarged. **Input/Output:** grayscale → larger grayscale (to `upscale_height`). **Benefit:** big accuracy gain on small text. **Limitation:** beyond ~2× adds latency without gains.

### Unsharp mask (Gaussian blur + weighted add) (active)
- **Why:** sharpens character edges / dot-peen dots. **Input/Output:** grayscale → edge-enhanced grayscale. **Benefit:** crisper strokes. **Limitation:** excessive amounts amplify noise.

### Adaptive thresholding & morphological ops (available / recommended, **off by default**)
- Documented in `OCR_MIGRATION_REPORT.md` as practical options for reflective/very-low-contrast plates (black-hat for glare, small morphological close to connect dots, adaptive threshold as an alternative branch). **Not enabled** in the default chain because aggressive thresholding can erase dot-peen dots; enable selectively and validate with the benchmark.

---

## Installation

**Python:** 3.11 (64-bit).

### Common
```bash
git clone <your-repo-url> AI_CAM
cd AI_CAM
```

### Windows (PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python run.py
```

### Linux (Ubuntu — production server, NVIDIA GPU)
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# GPU build of PaddlePaddle (match your CUDA; example CUDA 11.8):
pip uninstall -y paddlepaddle
pip install paddlepaddle-gpu==2.6.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

python run.py
```

Then open **http://localhost:8000** (or `http://<server-LAN-IP>:8000` from another PC).

### Dependencies (`requirements.txt`)
FastAPI · Uvicorn · Jinja2 · python-multipart · PyTorch + TorchVision (YOLO) · Ultralytics (YOLOv8n) · OpenCV · **PaddleOCR + PaddlePaddle** · Pillow · NumPy (pinned **1.26.4**, must stay on the 1.x line) · openpyxl.

### GPU / CUDA
- **Optional but recommended** on the Ubuntu box. With `DETECTION.device = "auto"` and `OCR.use_gpu = True`, both YOLO and PaddleOCR use `cuda:0` when available and **fall back to CPU automatically** otherwise.
- PyTorch GPU build from pytorch.org; PaddlePaddle GPU build as shown above. Match wheels to your installed CUDA.
- **Offline/SSL-restricted networks:** run once with internet to populate model caches (`~/.paddleocr/…`), then copy to the air-gapped server (the logs showed an EasyOCR first-run SSL failure — same applies to any first-run download).

---

## Configuration

All tunables live in `backend/config.py` (editing this one file is enough for deployment).

### Detection (`DetectionConfig`)
| Parameter | Default | Meaning |
|---|---|---|
| `model_path` | `runs/detect/vin_plate_model/weights/best.pt` | Trained YOLOv8n weights (auto-resolved from project root) |
| `conf_threshold` | `0.40` | Base YOLO detection confidence (also feeds dataset collection) |
| `iou_threshold` | `0.45` | NMS IoU |
| `ocr_trigger_conf` | `0.95` | **OCR fires only above this** (95%) |
| `device` | `"auto"` | `auto` → cuda:0 if available else cpu; or `"cuda:0"`/`"cpu"` |
| `crop_padding` | `8` | Pixels added around the ROI before OCR |
| `ocr_rearm_absent_frames` | `3` | Plate-absent frames before the trigger re-arms |
| `ocr_cooldown_sec` | `2.0` | Min seconds between OCR triggers |
| `collect_enabled` / `collect_conf` | `True` / `0.40` | Dataset auto-collection on/threshold |
| `collect_min_interval_sec` | `1.0` | Auto-collection throttle (FPS protection) |

### OCR (`OCRConfig`)
| Parameter | Default | Meaning |
|---|---|---|
| `lang` | `"en"` | PaddleOCR language |
| `use_gpu` | `True` | Use GPU if CUDA present (else CPU) |
| `apply_clahe` / `clahe_clip` / `clahe_grid` | `True` / `3.0` / `8` | CLAHE contrast |
| `upscale_height` | `96` | Upscale small crops to this height (0 = off) |
| `sharpen` | `True` | Light unsharp mask |
| `paddle_det` | `True` | det+rec; set `False` for faster **rec-only** on single-line crops |
| `use_angle_cls` | `False` | Angle classifier (off — metal text is upright) |
| `drop_score` | `0.30` | PaddleOCR drops results below this |
| `rec_batch_num` / `det_limit_side_len` | `6` / `960` | Recognizer batch / detector max side |
| `min_char_confidence` | `0.30` | Per-fragment confidence floor |
| `min_confidence` | `0.45` | Overall (mean) confidence floor |
| `queue_maxsize` | `50` | OCR queue size (back-pressure) |
| `min_interval_sec` | `0.0` | Optional throttle between OCR runs |
| `duplicate_window_sec` | `90.0` | Anti-duplicate window |

### Camera (`CameraConfig`)
`ip` (UI-overridable default `192.168.1.10`) · `cola_port = 2111` · `blob_port = 2113` · `password = "A89A6E74"` · timeouts/reconnect.

### Server (`ServerConfig`)
`host = "0.0.0.0"` (LAN-accessible) · `port = 8000` · `mjpeg_fps = 25` · `jpeg_quality = 80`.

### Auth (`AuthConfig`)
`enabled = True` · `username = "admin"` · `password = "vin_factory2026"` · `cookie_name = "ai_cam_session"` · `session_ttl_sec = 43200` (12 h). **Change the password for real deployments.**

### Paths
`BASE_DIR/data/ai_cam.db` (SQLite) · `data/crops/` (saved crops) · `dataset/collected_raw/` (auto-collected) · `logs/ai_cam.log` · `models/` · `runs/detect/vin_plate_model/weights/best.pt`.

---

## Performance

> The table below is **expected / typical** (not measured on your data). Generate authoritative numbers on your hardware with the included harness:
> ```bash
> python tools/benchmark_ocr.py --crops data/crops --runs 5 --gpu --gt-from-filename --paddle-rec-only
> ```
> It reports p50/p95/mean latency, peak memory, and accuracy for **both** engines on identically preprocessed crops.

**Per single VIN crop (~300×64), identical preprocessing — expected ranges:**

| Stage / Engine | CPU | GPU (NVIDIA) | Peak RAM | Accuracy (17-char) |
|---|---|---|---|---|
| YOLOv8n detection (per frame) | ~15–40 ms | ~3–8 ms | model ~12 MB | — |
| **EasyOCR** (previous) | ~120–300 ms | ~20–45 ms | ~0.7–1.2 GB | baseline |
| **PaddleOCR det+rec** (current) | ~70–180 ms | ~10–25 ms | ~0.4–0.8 GB | +2–6 pts |
| **PaddleOCR rec-only** (current) | ~25–70 ms | ~5–15 ms | ~0.3–0.6 GB | +1–5 pts |

Because the single-shot trigger fires OCR **once per plate event**, total per-vehicle OCR cost is one inference — a few milliseconds on GPU in rec-only mode. CPU/GPU utilization depends on conveyor throughput; measure with the harness above. Full discussion in `OCR_MIGRATION_REPORT.md`.

---

## Project Structure

```
AI_CAM/
├── run.py                       # Entry point: prints LAN URL, launches uvicorn (0.0.0.0:8000)
├── requirements.txt             # Pinned deps (Python 3.11)
├── README.md                    # This document
├── TRAINING_GUIDE.md            # How to train best.pt + dataset auto-collection
├── OCR_MIGRATION_REPORT.md      # EasyOCR → PaddleOCR analysis, benchmark, recommendations
│
├── backend/
│   ├── config.py                # ALL tunables (camera, detection, OCR, server, auth)
│   ├── logger.py                # Rotating file log + in-memory ring buffer for the UI
│   ├── server.py                # FastAPI app: pages, REST API, MJPEG, auth middleware
│   ├── pipeline.py              # Orchestrator: stream loop, YOLO, single-shot trigger, OCR submit, DB write
│   ├── auth.py                  # Session-cookie auth (login/validate/destroy)
│   ├── camera/
│   │   ├── camera_client.py     # SICK Lector 652 CoLa-A + BLOB (mLIStart) + decode_bmp()
│   │   ├── cola_client.py       # Deprecated shim → re-exports Lector652Client
│   │   └── blob_stream.py       # Deprecated shim → re-exports decode_bmp
│   ├── ai/
│   │   ├── detector.py          # YOLOv8n load (weights_only shim) + detect + draw + crop
│   │   ├── ocr_worker.py        # PaddleOCR background worker (preprocess, verify, anti-dup, stats)
│   │   └── dataset_collector.py # Auto-saves frame + YOLO label (data loop)
│   └── database/
│       └── db.py                # SQLite schema + CRUD + export queries
│
├── frontend/
│   ├── templates/  login.html · dashboard.html · history.html
│   └── static/     css/style.css · js/dashboard.js · js/history.js
│
├── tools/
│   └── benchmark_ocr.py         # EasyOCR vs PaddleOCR latency/memory/accuracy harness
│
├── models/                      # (optional stock weights)
├── runs/detect/vin_plate_model/weights/best.pt   # trained YOLOv8n (loaded directly)
├── data/        ai_cam.db · crops/
├── dataset/collected_raw/       # auto-collected images + YOLO labels
└── logs/        ai_cam.log
```

---

## Future Improvements (roadmap)

- **Character-level classification / confidence map** — per-character scores and a check-digit (ISO 3779 position-9) validator to auto-flag suspect reads.
- **Custom OCR training** — fine-tune the PaddleOCR recognizer on collected dot-peen crops (the data loop already gathers labeled samples) for a domain-specific model.
- **Industrial deployment hardening** — run as a systemd/Windows service, HTTPS/reverse proxy, persistent sessions, role-based accounts, PLC handshake to gate on body-in-position.
- **Multi-camera support** — multiple `Lector652Client` instances / camera registry, per-station pipelines and dashboards.
- **API integration** — webhook/REST push of accepted VINs to MES/ERP, Postgres option for multi-node, message-queue (MQTT/Kafka) emit.
- **Batch service** — a first-class endpoint to OCR an uploaded folder/zip of crops for offline reprocessing.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| **YOLO fails to load** (`WeightsUnpickler` / `UnsupportedGlobal`) | New PyTorch (`weights_only=True`) + ultralytics 8.2.x | Already handled by the `weights_only` shim in `detector.py`; or `pip install -U ultralytics` |
| Dashboard shows **YOLO: Not loaded** | `best.pt` missing at the configured path | Train per `TRAINING_GUIDE.md`; confirm `runs/detect/vin_plate_model/weights/best.pt` |
| **PaddleOCR fails to load** (SSL/urlopen) | First-run model download blocked by proxy | Pre-place models in `~/.paddleocr/…` from an online machine |
| **GPU not used** | CUDA/driver or CPU-only Paddle wheel | Install `paddlepaddle-gpu` matching CUDA; check `nvidia-smi`; `use_gpu` auto-falls back to CPU |
| **Two OpenCV versions** / `cv2` conflicts | a headless build co-installed | `pip uninstall -y opencv-python-headless && pip install --force-reinstall opencv-python==4.10.0.84` |
| Other PCs **can't reach** the dashboard | Windows Firewall blocking 8000 | Allow inbound TCP 8000 (admin): `New-NetFirewallRule -DisplayName "AI_CAM 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow` |
| Camera **won't connect / times out** | SOPAS ET still holding the single connection, wrong IP/port | Close SOPAS ET; verify IP in the dashboard field; camera on 2111/2113 |
| **No frames / decode warnings** | BLOB transfer off or wrong port | Enable Image/BLOB transfer (port 2113) in SOPAS; check `logs/ai_cam.log` |
| **Same VIN logged repeatedly** | — | By design it's suppressed within `duplicate_window_sec`; raise it if needed |
| **Good plate never triggers OCR** | `ocr_trigger_conf = 0.95` too strict for some etched plates | Lower it slightly (e.g. 0.90); single-shot lock still prevents repeats |

---

## License

No open-source license file is currently included. By default this project is **proprietary — © 2026, all rights reserved** to the project owner, intended for internal industrial use.

To open-source it (e.g. for portfolio/GitHub), add a `LICENSE` file with your chosen license (**MIT** or **Apache-2.0** are common for CV projects) and replace this section accordingly. Note that third-party dependencies retain their own licenses (Ultralytics YOLOv8 is **AGPL-3.0** unless you hold a commercial license — review this before any commercial/closed distribution).

---

*Built on the proven `lector652_pipeline` camera methodology · YOLOv8n + PaddleOCR · FastAPI MES dashboard. See `OCR_MIGRATION_REPORT.md` and `TRAINING_GUIDE.md` for deeper detail.*
