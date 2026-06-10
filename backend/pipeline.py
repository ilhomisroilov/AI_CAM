"""
============================================================
pipeline.py  —  Tizim orkestratori (yagona boshqaruv markazi)
============================================================
Tarmoq logikasi `lector652_pipeline` loyihasidagi ISHLAYDIGAN koddan
olingan (camera_client.Lector652Client). Taxmin qilingan generic TCP
reader EMAS — aniq mLIStart live-push + BLOB framing ishlatiladi.

Oqim:
   SICK Lector652
       │  CoLa-A (2111): connect handshake + sMN mLIStart 0
       ▼  BLOB (2113): kamera AVTOMATIK frame push qiladi
   read_stream_frame() -> decode_bmp()  (XOM grayscale, AYLANTIRISH YO'Q)
       │
       ├─► live_frame (MJPEG uchun)
       ├─► YOLOv8n aniqlash + bbox chizish   (faqat "Start" bosilganda)
       └─► yuqori ishonchli ROI ──► OCR navbati ──► DB

UI semantikasi:
  Connect Camera : ulanadi + live stream boshlanadi (xom video ko'rinadi)
  Start Processing: YOLO + OCR yoqiladi (bbox + VIN o'qish + DB)
  Stop           : YOLO + OCR o'chadi (xom video davom etadi)
  Disconnect     : stream va ulanish yopiladi
"""
from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .ai.detector import PlateDetector
from .ai.dataset_collector import DatasetCollector
from .ai.ocr_worker import OCRWorker
from .camera.camera_client import Lector652Client, decode_bmp
from .config import CAMERA, CROPS_DIR, DATASET_DIR, DETECTION, SERVER
from .database import db
from .logger import log


class Pipeline:
    """Butun AI_CAM ish jarayonini boshqaradi (ishlaydigan tarmoq logikasi bilan)."""

    def __init__(self) -> None:
        self.detector = PlateDetector()
        self.ocr = OCRWorker(on_result=self._on_ocr_result)
        self.client: Optional[Lector652Client] = None

        # Avtomatik dataset yig'uvchi (fon thread + navbat)
        self.collector = DatasetCollector(
            out_dir=str(DATASET_DIR),
            collect_conf=DETECTION.collect_conf,
            class_id=DETECTION.collect_class_id,
            min_interval_sec=DETECTION.collect_min_interval_sec,
        )

        # Live kadr (JPEG bytes) — MJPEG stream shu yerdan o'qiydi
        self._live_jpeg: Optional[bytes] = None
        self._frame_lock = threading.Lock()

        self._stream_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._camera_connected = False
        self._processing = False           # YOLO + OCR yoniq/ o'chiq

        # Statistika (UI uchun)
        self.stats = {"frames": 0, "detections": 0, "vins": 0, "fps": 0.0}
        self._ts_buf: list[float] = []     # FPS hisoblash uchun

        # --- Single-shot trigger holati (debounce / state-lock) ---
        # Bitta plastinka kadrda bir necha kadr turganda OCR FAQAT BIR MARTA
        # ishga tushadi. Plastinka kadrdan ketganda qulf bo'shaydi (re-arm).
        self._plate_locked = False         # True -> bu plastinka allaqachon OCR ga yuborilgan
        self._absent_frames = 0            # ketma-ket "plastinka yo'q" kadrlar soni
        self._last_ocr_ts = -1e9           # oxirgi OCR trigger vaqti (birinchi trigger doim o'tadi)

    # ===============================================================
    # Kamera ulanish (Connect Camera tugmasi) — IP UI dan dinamik
    # ===============================================================
    def connect_camera(self, ip: str | None = None) -> bool:
        """
        Kameraga ulanadi va live streamni boshlaydi.
        IP UI dan dinamik keladi (hardcode YO'Q). BLOB Port 2113.
        """
        if self._camera_connected:
            log.info("Kamera allaqachon ulangan.")
            return True

        ip = (ip or CAMERA.ip).strip()
        CAMERA.ip = ip
        log.info(f"Kamera IP: {ip} (CoLa {CAMERA.cola_port}, BLOB {CAMERA.blob_port})")

        # Ishlaydigan klient — on_log bizning loggerga ulanadi
        self.client = Lector652Client(
            ip=ip,
            control_port=CAMERA.cola_port,
            blob_port=CAMERA.blob_port,
            password=CAMERA.password,
            on_log=log.info,
        )
        try:
            self.client.connect()                 # CoLa handshake + BLOB socket
        except Exception as exc:
            log.error(f"Kamera ulanishi muvaffaqiyatsiz: {exc}")
            self.client = None
            return False

        # Live stream loop (mLIStart) ni alohida threadda boshlaymiz
        self._stop_event.clear()
        self._stream_thread = threading.Thread(
            target=self._stream_loop, name="lector-stream", daemon=True
        )
        self._stream_thread.start()
        self._camera_connected = True
        return True

    def disconnect_camera(self) -> None:
        self.stop_processing()
        self._stop_event.set()
        if self._stream_thread:
            self._stream_thread.join(timeout=8.0)
            self._stream_thread = None
        if self.client:
            try:
                self.client.disconnect()
            except Exception:
                pass
            self.client = None
        self._camera_connected = False
        with self._frame_lock:
            self._live_jpeg = None
        log.info("Kamera uzildi.")

    # ===============================================================
    # YOLO + OCR yoqish / o'chirish (Start / Stop)
    # ===============================================================
    def start_processing(self) -> bool:
        if not self._camera_connected:
            log.warning("Avval kamerani ulang (Connect Camera).")
            return False
        if self._processing:
            return True
        self.ocr.start()
        if DETECTION.collect_enabled:
            self.collector.start()               # dataset auto-collection (fon thread)
        self._processing = True
        log.info("AI ishlov berish yoqildi (YOLO + OCR + dataset yig'ish).")
        return True

    def stop_processing(self) -> None:
        if not self._processing:
            return
        self._processing = False
        self.ocr.stop()
        self.collector.stop()
        log.info("AI ishlov berish o'chirildi (xom video davom etadi).")

    # ===============================================================
    # Live stream loop — ishlaydigan _fast_loop ga mos
    # ===============================================================
    def _stream_loop(self) -> None:
        """
        mLIStart bilan kameradan frame oqimini o'qiydi.
        Har frame: decode_bmp (XOM) -> (ixtiyoriy) YOLO+OCR -> MJPEG buffer.
        """
        try:
            self.client.start_stream()            # sMN mLIStart 0
        except Exception as exc:
            log.error(f"start_stream xatosi: {exc}")
            return

        errors = 0
        timeout_count = 0
        while not self._stop_event.is_set():
            try:
                # timeout endi socket darajasida (SO_RCVTIMEO=2s) boshqariladi
                bmp = self.client.read_stream_frame()
                img = decode_bmp(bmp)             # XOM grayscale (AYLANTIRISH YO'Q)
                if img is None:
                    continue

                # YOLO/MJPEG rangli bbox uchun BGR ga o'tkazamiz (xom piksellar saqlanadi)
                frame = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img

                annotated = frame
                if self._processing and self.detector.ready:
                    dets = self.detector.detect(frame)
                    if dets:
                        self.stats["detections"] += 1
                        annotated = self.detector.draw_boxes(frame, dets)
                        # Dataset auto-collection: ASL (kesilmagan) kadr + YOLO
                        # yorliqlar. Throttle + disk yozish fon threadda — FPS tushmaydi.
                        self.collector.maybe_collect(frame, dets)

                    # Single-shot trigger logikasi (debounce / state-lock)
                    self._handle_trigger(frame, dets)

                # MJPEG buffer — xom yoki annotatsiyalangan kadr
                ok, buf = cv2.imencode(
                    ".jpg", annotated,
                    [int(cv2.IMWRITE_JPEG_QUALITY), SERVER.jpeg_quality],
                )
                if ok:
                    with self._frame_lock:
                        self._live_jpeg = buf.tobytes()

                # Statistika
                self.stats["frames"] += 1
                self._tick_fps()
                errors = 0
                timeout_count = 0

            except TimeoutError:
                if self._stop_event.is_set():
                    break
                timeout_count += 1
                log.warning(f"Frame timeout #{timeout_count} — kutilmoqda...")
                if timeout_count >= 5:
                    log.error("5 timeout — kamera push qilmayapti, stream to'xtatildi.")
                    break
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                errors += 1
                log.error(f"Stream xatosi #{errors}: {exc}")
                if errors >= 5:
                    log.error("5 xato — stream to'xtatildi.")
                    break
                time.sleep(0.1)

        # Loop tugadi — streamni to'xtatamiz
        try:
            self.client.stop_stream()
        except Exception:
            pass

    # ===============================================================
    # Single-shot trigger (debounce / state-lock)
    # ===============================================================
    def _handle_trigger(self, frame: np.ndarray, dets: list) -> None:
        """
        Bitta plastinka hodisasi uchun OCR ni FAQAT BIR MARTA ishga tushiradi.

        Mantiq (tracking state):
          * plate_present = kadrda har qanday plastinka bor (>= conf_threshold).
            -> hisoblagichni nollaydi; plastinka hali kadrda turibdi.
          * high = ishonchi >= ocr_trigger_conf (0.95) bo'lgan eng yaxshi aniqlov.
          * Faqat QULF OCHIQ va cooldown o'tgan bo'lsa -> OCR ga 1 marta yuboriladi,
            so'ng QULF YOPILADI (keyingi kadrlarda qayta yuborilmaydi).
          * Plastinka ocr_rearm_absent_frames kadr ko'rinmasa -> QULF OCHILADI
            (keyingi avtomobil yangi hodisa sifatida qayta trigger bo'ladi).
        """
        plate_present = len(dets) > 0

        if plate_present:
            self._absent_frames = 0
            if self._plate_locked:
                return  # bu plastinka allaqachon ishlangan — keyingi kadrlarni o'tkazamiz

            high = self.detector.best_detection(dets)   # conf >= 0.95
            if high is None:
                return  # yuqori ishonch yo'q — hali trigger qilmaymiz

            now = time.monotonic()
            if (now - self._last_ocr_ts) < DETECTION.ocr_cooldown_sec:
                return  # cooldown — xavfsizlik kechikishi

            roi = self.detector.crop_roi(frame, high)
            if roi is None:
                return
            self.ocr.submit(roi)
            self._plate_locked = True
            self._last_ocr_ts = now
            log.info(f"TRIGGER (single-shot): VIN plate conf={high[4]:.2f} >= "
                     f"{DETECTION.ocr_trigger_conf:.2f} -> OCR (1 marta).")
        else:
            # Kadrda plastinka yo'q — ketganini tasdiqlash uchun sanaymiz
            self._absent_frames += 1
            if self._plate_locked and self._absent_frames >= DETECTION.ocr_rearm_absent_frames:
                self._plate_locked = False
                log.info("Plastinka kadrdan ketdi — trigger qayta yoqildi (re-arm).")

    def _tick_fps(self) -> None:
        now = time.monotonic()
        self._ts_buf.append(now)
        if len(self._ts_buf) > 60:
            self._ts_buf = self._ts_buf[-60:]
        if len(self._ts_buf) >= 2:
            self.stats["fps"] = round(
                (len(self._ts_buf) - 1) / (self._ts_buf[-1] - self._ts_buf[0]), 1
            )

    # ===============================================================
    # Callback: OCR yaroqli (dublikat bo'lmagan) VIN qaytardi
    # ===============================================================
    def _on_ocr_result(self, vin: str, confidence: float, crop: np.ndarray) -> None:
        ts = datetime.now()
        ts_iso = ts.strftime("%Y-%m-%d %H:%M:%S")

        fname = f"{vin}_{ts.strftime('%Y%m%d_%H%M%S')}.jpg"
        fpath = Path(CROPS_DIR) / fname
        rel_path = None
        try:
            cv2.imwrite(str(fpath), crop)
            rel_path = f"crops/{fname}"
        except Exception as exc:
            log.error(f"Crop saqlanmadi: {exc}")

        try:
            rec_id = db.insert_record(ts_iso, vin, confidence, rel_path)
            self.stats["vins"] += 1
            log.info(f"DB ga yozildi #{rec_id}: VIN={vin} conf={confidence:.2f}")
        except Exception as exc:
            log.error(f"DB yozuv xatosi: {exc}")

    # ===============================================================
    # MJPEG stream uchun: oxirgi JPEG kadr
    # ===============================================================
    def get_jpeg(self) -> Optional[bytes]:
        with self._frame_lock:
            return self._live_jpeg

    # ===============================================================
    # Holat (UI status uchun)
    # ===============================================================
    def status(self) -> dict:
        return {
            "camera_connected": self._camera_connected,
            "processing": self._processing,
            "yolo_ready": self.detector.ready,
            "stats": self.stats,
            "ocr": self.ocr.get_stats(),     # OCR monitoring (navbat, ms, accepted...)
        }


# Global yagona pipeline nusxasi (server import qiladi)
pipeline = Pipeline()
