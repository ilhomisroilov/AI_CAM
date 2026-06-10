"""
============================================================
ocr_worker.py  —  EasyOCR fon (background) navbat ishchisi
============================================================
Maqsad: YOLO bergan crop'lardan VIN ni YUQORI ANIQLIK bilan o'qish,
ishonch chegaralarini boshqarish, va live oqim lag bermasligi uchun
tezlikni nazorat qilish.

Asoslar (lector652 VIN-OCR metodologiyasi):
  * EasyOCR (Tesseract EMAS) — dot-peen (nuqtali) etched VIN uchun.
  * OCR ALOHIDA fon threadida navbat (Queue) orqali — kamera FPS tushmaydi.
  * Anti-duplicate: bir xil VIN duplicate_window_sec ichida qayta yozilmaydi.
  * VIN validatsiya: 17 belgi, I/O/Q yo'q.

Aniqlik uchun (yangi):
  * Crop -> grayscale -> CLAHE -> kattalashtirish (upscale) -> yengil sharpen.
  * readtext: allowlist (VIN charset) + beamsearch + tuned thresholds + mag_ratio.
  * Bo'laklar bbox X bo'yicha CHAPDAN-O'NGGA tartiblanadi (to'g'ri ketma-ketlik).
  * Per-fragment va umumiy ishonch chegaralari.

Tezlik / nazorat (yangi):
  * Drop-oldest navbat (eng eski crop tashlanadi).
  * min_interval_sec throttle (ixtiyoriy).
  * Runtime stats (get_stats) + dinamik boshqaruv (set_enabled / set_min_confidence).
"""
from __future__ import annotations

import queue
import re
import threading
import time
from collections import deque
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from ..config import OCR, VIN_ALLOWED, VIN_INVALID_CHARS, VIN_LENGTH
from ..logger import log

# on_result(vin: str, confidence: float, crop_bgr: np.ndarray)
ResultCallback = Callable[[str, float, np.ndarray], None]

# VIN ruxsat etilgan belgilaridan EasyOCR allowlist (I/O/Q yo'q)
_ALLOWLIST = "".join(sorted(VIN_ALLOWED))


def normalize_vin(text: str) -> str:
    """OCR matnini VIN ko'rinishiga keltiradi: katta harf, faqat A-Z0-9."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def is_valid_vin(vin: str) -> bool:
    """VIN qoidasi: aniq 17 belgi, I/O/Q yo'q, faqat ruxsat etilgan belgilar."""
    if len(vin) != VIN_LENGTH:
        return False
    if any(c in VIN_INVALID_CHARS for c in vin):
        return False
    return all(c in VIN_ALLOWED for c in vin)


def _fragment_left_x(box) -> float:
    """EasyOCR bbox (4 nuqta) dan eng chap X koordinatani oladi (tartiblash uchun)."""
    try:
        return float(min(pt[0] for pt in box))
    except Exception:
        return 0.0


class OCRWorker:
    """EasyOCR ni fon threadida navbat orqali ishlatadi (aniqlik + nazorat)."""

    def __init__(self, on_result: ResultCallback) -> None:
        self.on_result = on_result
        self._queue: "queue.Queue" = queue.Queue(maxsize=OCR.queue_maxsize)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._reader = None                      # EasyOCR Reader (lazy)

        # Dinamik boshqaruv
        self._enabled = True
        self._min_conf = OCR.min_confidence
        self._last_run = 0.0                     # throttle (monotonic)

        # Anti-duplicate
        self._last_vin: Optional[str] = None
        self._last_vin_ts: float = 0.0
        self._dup_lock = threading.Lock()

        # CLAHE
        self._clahe = cv2.createCLAHE(
            clipLimit=OCR.clahe_clip, tileGridSize=(OCR.clahe_grid, OCR.clahe_grid)
        )

        # Monitoring statistikasi
        self._stats_lock = threading.Lock()
        self._processed = 0
        self._accepted = 0
        self._rejected = 0
        self._dropped = 0
        self._last_ms = 0.0
        self._ms_hist: deque = deque(maxlen=30)
        self._last_vin_read = ""
        self._last_conf = 0.0

    # ===============================================================
    # Hayotiy sikl
    # ===============================================================
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="ocr-worker", daemon=True)
        self._thread.start()
        log.info("OCR worker ishga tushdi (EasyOCR, fon navbat, beamsearch).")

    def stop(self) -> None:
        self._running = False
        try:
            self._queue.put_nowait(None)         # sentinel
        except queue.Full:
            pass
        log.info("OCR worker to'xtatildi.")

    # ===============================================================
    # Dinamik nazorat (monitoring & control)
    # ===============================================================
    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)
        log.info("OCR yoqildi." if enabled else "OCR o'chirildi.")

    def set_min_confidence(self, value: float) -> None:
        self._min_conf = max(0.0, min(1.0, float(value)))
        log.info(f"OCR minimal ishonch chegarasi: {self._min_conf:.2f}")

    def get_stats(self) -> dict:
        with self._stats_lock:
            avg_ms = (sum(self._ms_hist) / len(self._ms_hist)) if self._ms_hist else 0.0
            return {
                "enabled": self._enabled,
                "running": self._running,
                "queue": self._queue.qsize(),
                "queue_max": OCR.queue_maxsize,
                "processed": self._processed,
                "accepted": self._accepted,
                "rejected": self._rejected,
                "dropped": self._dropped,
                "avg_ms": round(avg_ms, 1),
                "last_ms": round(self._last_ms, 1),
                "min_conf": round(self._min_conf, 2),
                "last_vin": self._last_vin_read,
                "last_conf": round(self._last_conf, 2),
            }

    # ===============================================================
    # Navbat (drop-oldest — FPS himoyasi)
    # ===============================================================
    def submit(self, crop_bgr: np.ndarray) -> None:
        if not self._enabled:
            return
        try:
            self._queue.put_nowait(crop_bgr)
        except queue.Full:
            try:
                self._queue.get_nowait()         # eng eskini tashlaymiz
                self._queue.put_nowait(crop_bgr)
                with self._stats_lock:
                    self._dropped += 1
            except queue.Empty:
                pass

    # ===============================================================
    # EasyOCR Reader (lazy)
    # ===============================================================
    def _ensure_reader(self) -> bool:
        if self._reader is not None:
            return True
        try:
            import easyocr
            log.info("EasyOCR modeli yuklanmoqda... (birinchi marta sekin).")
            self._reader = easyocr.Reader(OCR.languages, gpu=OCR.use_gpu)
            log.info(f"EasyOCR tayyor (gpu={OCR.use_gpu}, decoder={OCR.decoder}).")
            return True
        except Exception as exc:
            log.error(f"EasyOCR yuklanmadi: {exc}")
            return False

    def _loop(self) -> None:
        if not self._ensure_reader():
            self._running = False
            return
        while self._running:
            try:
                crop = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if crop is None:                     # stop sentinel
                break
            # Throttle: ketma-ket OCR orasidagi minimal vaqt
            if OCR.min_interval_sec > 0:
                dt = time.monotonic() - self._last_run
                if dt < OCR.min_interval_sec:
                    continue                     # bu crop'ni o'tkazib yuboramiz (lag himoyasi)
            self._last_run = time.monotonic()
            self._process(crop)

    # ===============================================================
    # Preprocess — aniqlik uchun (kattalashtirish + CLAHE + sharpen)
    # ===============================================================
    def _preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr

        # Kichik etched belgilarni kattalashtirish (recognizer aniqligi oshadi)
        if OCR.upscale_height and gray.shape[0] < OCR.upscale_height:
            scale = OCR.upscale_height / float(gray.shape[0])
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        if OCR.apply_clahe:
            gray = self._clahe.apply(gray)

        if OCR.sharpen:
            # Yengil unsharp mask — chekkalarni aniqlashtiradi (dot-peen uchun foydali)
            blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
            gray = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

        return gray

    # ===============================================================
    # OCR
    # ===============================================================
    def _read(self, img: np.ndarray) -> List[Tuple[list, str, float]]:
        """EasyOCR readtext — VIN uchun sozlangan parametrlar bilan."""
        return self._reader.readtext(
            img,
            detail=1,
            allowlist=_ALLOWLIST,
            decoder=OCR.decoder,
            beamWidth=OCR.beam_width,
            text_threshold=OCR.text_threshold,
            low_text=OCR.low_text,
            link_threshold=OCR.link_threshold,
            mag_ratio=OCR.mag_ratio,
            contrast_ths=OCR.contrast_ths,
            adjust_contrast=OCR.adjust_contrast,
            paragraph=False,
        )

    def _process(self, crop_bgr: np.ndarray) -> None:
        t0 = time.perf_counter()
        img = self._preprocess(crop_bgr)
        try:
            results = self._read(img)
        except Exception as exc:
            log.error(f"EasyOCR readtext xatosi: {exc}")
            return

        dt_ms = (time.perf_counter() - t0) * 1000.0

        # Past-ishonchli bo'laklarni tashlab, CHAPDAN-O'NGGA tartiblaymiz
        frags = [(box, txt, float(conf)) for (box, txt, conf) in results
                 if float(conf) >= OCR.min_char_confidence and txt.strip()]
        frags.sort(key=lambda r: _fragment_left_x(r[0]))

        combined = normalize_vin("".join(f[1] for f in frags))
        avg_conf = float(np.mean([f[2] for f in frags])) if frags else 0.0

        with self._stats_lock:
            self._processed += 1
            self._last_ms = dt_ms
            self._ms_hist.append(dt_ms)
            self._last_vin_read = combined
            self._last_conf = avg_conf

        # --- Validatsiya + ishonch chegaralari ---
        if not is_valid_vin(combined):
            with self._stats_lock:
                self._rejected += 1
            log.info(f"OCR: yaroqli VIN topilmadi ('{combined}', {dt_ms:.0f} ms).")
            return
        if avg_conf < self._min_conf:
            with self._stats_lock:
                self._rejected += 1
            log.info(f"OCR: ishonch past ({avg_conf:.2f}<{self._min_conf:.2f}) — '{combined}' rad etildi.")
            return

        # --- Anti-duplicate ---
        if self._is_duplicate(combined):
            log.info(f"DUPLICATE: '{combined}' yaqinda o'qilgan — e'tiborsiz qoldirildi.")
            return

        with self._stats_lock:
            self._accepted += 1
        log.info(f"VIN aniqlandi: {combined} (conf={avg_conf:.2f}, {dt_ms:.0f} ms)")
        try:
            self.on_result(combined, avg_conf, crop_bgr)
        except Exception as exc:
            log.error(f"on_result callback xatosi: {exc}")

    def _is_duplicate(self, vin: str) -> bool:
        now = time.time()
        with self._dup_lock:
            if (self._last_vin == vin
                    and (now - self._last_vin_ts) < OCR.duplicate_window_sec):
                return True
            self._last_vin = vin
            self._last_vin_ts = now
            return False
