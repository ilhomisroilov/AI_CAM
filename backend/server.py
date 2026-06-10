"""
============================================================
server.py  —  FastAPI backend + MJPEG live stream + REST API
============================================================
Endpointlar:
  GET  /                      -> /dashboard ga yo'naltirish
  GET  /dashboard             -> dashboard sahifasi
  GET  /history               -> history sahifasi
  GET  /video_feed            -> MJPEG live stream (YOLO bbox bilan)
  POST /api/camera/connect    -> Connect Camera
  POST /api/processing/start  -> Start
  POST /api/processing/stop   -> Stop
  GET  /api/status            -> tizim holati + statistika
  GET  /api/logs?since=ID     -> yangi loglar (UI yon panel pollingi)
  GET  /api/records           -> history jadval ma'lumotlari (saralash)
  GET  /api/export?fmt=csv|xlsx -> eksport
  GET  /crops/{name}          -> saqlangan crop rasmlari
"""
from __future__ import annotations

import asyncio
import csv
import io
from pathlib import Path
import torch
import ultralytics

# PyTorch'ga YOLO modelini xavfsiz deb tanitish
torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])

from fastapi import FastAPI, Query
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               RedirectResponse, StreamingResponse)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from .config import CROPS_DIR, SERVER
from .database import db
from .logger import log, ui_handler
from .pipeline import pipeline

# --- Papka manzillari ---
_HERE = Path(__file__).resolve().parent
_FRONTEND = _HERE.parent / "frontend"

app = FastAPI(title="AI_CAM — Industrial VIN Vision", version="1.0")

# Statik fayllar va shablonlar
app.mount("/static", StaticFiles(directory=str(_FRONTEND / "static")), name="static")
templates = Jinja2Templates(directory=str(_FRONTEND / "templates"))


@app.on_event("startup")
def _startup() -> None:
    db.init_db()
    log.info("AI_CAM server ishga tushdi.")


# ===================================================================
# Sahifalar
# ===================================================================
@app.get("/", response_class=RedirectResponse)
def root() -> RedirectResponse:
    return RedirectResponse(url="/dashboard")


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request=request, name="dashboard.html")


@app.get("/history", response_class=HTMLResponse)
def history(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request=request, name="history.html")

# ===================================================================
# MJPEG live stream
# ===================================================================
async def _mjpeg_generator():
    """Multipart MJPEG: pipeline'dan oxirgi kadrni doimiy uzatadi."""
    boundary = b"--frame"
    interval = 1.0 / max(1, SERVER.mjpeg_fps)
    while True:
        jpeg = pipeline.get_jpeg()
        if jpeg is not None:
            yield (boundary + b"\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                   + jpeg + b"\r\n")
        await asyncio.sleep(interval)


@app.get("/video_feed")
def video_feed() -> StreamingResponse:
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ===================================================================
# Boshqaruv API (Connect / Start / Stop)
# ===================================================================
@app.post("/api/camera/connect")
async def api_connect(request: Request) -> JSONResponse:
    """
    Kamera IP manzili frontend dan dinamik keladi (JSON: {"ip": "192.168.1.10"}).
    Hardcode IP ISHLATILMAYDI. BLOB ulanishi Port 2113 da amalga oshiriladi.
    """
    ip = None
    try:
        body = await request.json()
        ip = (body or {}).get("ip")
    except Exception:
        ip = None  # tana bo'sh bo'lsa — config standart IP ishlatiladi
    if not ip:
        return JSONResponse(
            {"ok": False, "error": "Camera IP kiritilmadi.", "status": pipeline.status()},
            status_code=400,
        )
    ok = pipeline.connect_camera(ip=ip)
    return JSONResponse({"ok": ok, "ip": ip, "status": pipeline.status()})


@app.post("/api/camera/disconnect")
def api_disconnect() -> JSONResponse:
    pipeline.disconnect_camera()
    return JSONResponse({"ok": True, "status": pipeline.status()})


@app.post("/api/processing/start")
def api_start() -> JSONResponse:
    ok = pipeline.start_processing()
    return JSONResponse({"ok": ok, "status": pipeline.status()})


@app.post("/api/processing/stop")
def api_stop() -> JSONResponse:
    pipeline.stop_processing()
    return JSONResponse({"ok": True, "status": pipeline.status()})


@app.get("/api/status")
def api_status() -> JSONResponse:
    return JSONResponse(pipeline.status())


# ===================================================================
# Real-time loglar (UI yon panel pollingi)
# ===================================================================
@app.get("/api/logs")
def api_logs(since: int = Query(0)) -> JSONResponse:
    return JSONResponse({"logs": ui_handler.get_since(since)})


# ===================================================================
# History — yozuvlar va eksport
# ===================================================================
@app.get("/api/records")
def api_records(sort_by: str = "timestamp", order: str = "DESC",
                limit: int = 500) -> JSONResponse:
    return JSONResponse({"records": db.get_records(limit=limit, order=order, sort_by=sort_by)})


@app.get("/api/export")
def api_export(fmt: str = "csv"):
    """Yozuvlarni CSV yoki Excel (.xlsx) ko'rinishida eksport qiladi."""
    records = db.get_all_records()
    headers = ["id", "timestamp", "detected_vin", "confidence", "image_path"]

    if fmt == "xlsx":
        try:
            from openpyxl import Workbook
        except Exception:
            return JSONResponse({"error": "openpyxl o'rnatilmagan"}, status_code=500)
        wb = Workbook()
        ws = wb.active
        ws.title = "VIN Records"
        ws.append([h.upper() for h in headers])
        for r in records:
            ws.append([r[h] for h in headers])
        buf = io.BytesIO()
        wb.save(buf)
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=vin_records.xlsx"},
        )

    # default: CSV
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=headers)
    writer.writeheader()
    writer.writerows(records)
    return StreamingResponse(
        io.BytesIO(buf.getvalue().encode("utf-8-sig")),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=vin_records.csv"},
    )


# ===================================================================
# Saqlangan crop rasmlari
# ===================================================================
@app.get("/crops/{name}")
def get_crop(name: str):
    path = Path(CROPS_DIR) / name
    if not path.exists():
        return JSONResponse({"error": "topilmadi"}, status_code=404)
    return FileResponse(str(path))
