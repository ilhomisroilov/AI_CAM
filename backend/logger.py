"""
============================================================
logger.py  —  Real-time logging tizimi
============================================================
Ikki vazifa:
  1. Diskka aylanuvchi (rotating) log fayl yozish.
  2. Web UI yon paneli uchun xotirada oxirgi N ta logni saqlash
     (thread-safe ring buffer). UI shu bufferdan o'qiydi.
Hodisalar: kamera ulandi, VIN aniqlandi, OCR tugadi, xatolar.
"""
from __future__ import annotations

import logging
import logging.handlers
import threading
import time
from collections import deque
from typing import Deque, Dict, List

from .config import LOGS_DIR

_LOG_FILE = LOGS_DIR / "ai_cam.log"
_MAX_UI_LOGS = 300          # UI panelida ko'rsatiladigan oxirgi yozuvlar soni


class _RingBufferHandler(logging.Handler):
    """Loglarni xotiradagi deque ga yozadi — UI shu yerdan o'qiydi."""

    def __init__(self, maxlen: int = _MAX_UI_LOGS) -> None:
        super().__init__()
        self._buf: Deque[Dict] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._seq = 0

    def emit(self, record: logging.LogRecord) -> None:
        with self._lock:
            self._seq += 1
            self._buf.append(
                {
                    "id": self._seq,
                    "ts": time.strftime("%H:%M:%S", time.localtime(record.created)),
                    "level": record.levelname,
                    "msg": record.getMessage(),
                }
            )

    def get_since(self, last_id: int = 0) -> List[Dict]:
        """last_id dan keyingi yangi loglarni qaytaradi (UI polling uchun)."""
        with self._lock:
            return [e for e in self._buf if e["id"] > last_id]

    def snapshot(self) -> List[Dict]:
        with self._lock:
            return list(self._buf)


# --- Yagona ring-buffer handler (UI bilan baham ko'riladi) ---
ui_handler = _RingBufferHandler()


def setup_logger(name: str = "ai_cam") -> logging.Logger:
    """Asosiy loggerni sozlaydi: konsol + fayl + UI buffer."""
    logger = logging.getLogger(name)
    if logger.handlers:                      # ikki marta sozlashning oldini olish
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)-5s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 1) Diskka aylanuvchi fayl
    fh = logging.handlers.RotatingFileHandler(
        _LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # 2) Konsol
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # 3) UI ring buffer
    ui_handler.setFormatter(fmt)
    logger.addHandler(ui_handler)

    return logger


# Global logger — barcha modullar shu nusxani import qiladi
log = setup_logger()
