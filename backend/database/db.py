"""
============================================================
db.py  —  SQLite ma'lumotlar bazasi qatlami
============================================================
Jadval: vin_records
  id              INTEGER PRIMARY KEY
  timestamp       TEXT (ISO 8601)
  detected_vin    TEXT
  confidence      REAL
  image_path      TEXT   (kesilgan VIN rasmi)

Thread-safe: har bir amal o'z ulanishini ochadi (check_same_thread=False
o'rniga ulanish-per-amal — fon threadlaridan xavfsiz chaqirish uchun).
"""
from __future__ import annotations

import sqlite3
import threading
from typing import Dict, List, Optional

from ..config import DB_PATH
from ..logger import log

_lock = threading.Lock()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Bazani va jadvalni yaratadi (agar mavjud bo'lmasa)."""
    with _lock, _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vin_records (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp     TEXT    NOT NULL,
                detected_vin  TEXT    NOT NULL,
                confidence    REAL    NOT NULL,
                image_path    TEXT
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_vin_ts ON vin_records(timestamp)"
        )
    log.info(f"Ma'lumotlar bazasi tayyor: {DB_PATH}")


def insert_record(timestamp: str, vin: str, confidence: float,
                  image_path: Optional[str]) -> int:
    """Yangi VIN yozuvini qo'shadi va yangi ID ni qaytaradi."""
    with _lock, _connect() as conn:
        cur = conn.execute(
            "INSERT INTO vin_records (timestamp, detected_vin, confidence, image_path) "
            "VALUES (?, ?, ?, ?)",
            (timestamp, vin, confidence, image_path),
        )
        return int(cur.lastrowid)


def get_records(limit: int = 500, order: str = "DESC",
                sort_by: str = "timestamp") -> List[Dict]:
    """Yozuvlarni qaytaradi. /history sahifasi uchun (sana bo'yicha saralash)."""
    sort_col = sort_by if sort_by in {"timestamp", "detected_vin", "confidence", "id"} else "timestamp"
    direction = "DESC" if order.upper() == "DESC" else "ASC"
    with _lock, _connect() as conn:
        rows = conn.execute(
            f"SELECT id, timestamp, detected_vin, confidence, image_path "
            f"FROM vin_records ORDER BY {sort_col} {direction} LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_records() -> List[Dict]:
    """Eksport uchun barcha yozuvlar."""
    return get_records(limit=1_000_000, order="DESC")


def count_records() -> int:
    with _lock, _connect() as conn:
        return int(conn.execute("SELECT COUNT(*) FROM vin_records").fetchone()[0])
