"""
============================================================
db.py  —  SQLite ma'lumotlar bazasi qatlami
============================================================
Jadval: vin_records
  id              INTEGER PRIMARY KEY
  timestamp       TEXT (ISO 8601)
  detected_vin    TEXT   (VALIDATED VIN — strukturaga ko'ra saralangan)
  raw_vin         TEXT   (XOM OCR natijasi — audit/traceability uchun)
  model           TEXT   (QY | BL7M — YOLO klassifikatsiyasidan)
  confidence      REAL   (final_score)
  image_path      TEXT   (kesilgan VIN rasmi)

Migratsiya: eski bazada raw_vin/model ustunlari bo'lmasa, avtomatik qo'shiladi
(idempotent). PostgreSQL uchun ekvivalent DDL pastdagi izohda.

  PostgreSQL (kelajakda PG ga o'tilsa):
    ALTER TABLE vin_records ADD COLUMN IF NOT EXISTS raw_vin VARCHAR(32);
    ALTER TABLE vin_records ADD COLUMN IF NOT EXISTS model   VARCHAR(8);

Thread-safe: har bir amal o'z ulanishini ochadi (fon threadlaridan xavfsiz).
"""
from __future__ import annotations

import sqlite3
import threading
from typing import Dict, List, Optional

from ..config import DB_PATH
from ..logger import log

_lock = threading.Lock()

# Jadval ustunlari (tartibda) — SELECT/eksport uchun
COLUMNS = ["id", "timestamp", "detected_vin", "raw_vin", "model", "confidence", "image_path"]


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


def _existing_columns(conn) -> set:
    rows = conn.execute("PRAGMA table_info(vin_records)").fetchall()
    return {r[1] for r in rows}


def _migrate(conn) -> None:
    """Eski bazaga yangi ustunlarni qo'shadi (idempotent)."""
    cols = _existing_columns(conn)
    if "raw_vin" not in cols:
        conn.execute("ALTER TABLE vin_records ADD COLUMN raw_vin TEXT")
        log.info("DB migratsiya: 'raw_vin' ustuni qo'shildi.")
    if "model" not in cols:
        conn.execute("ALTER TABLE vin_records ADD COLUMN model TEXT")
        log.info("DB migratsiya: 'model' ustuni qo'shildi.")


def init_db() -> None:
    """Bazani va jadvalni yaratadi + migratsiya qiladi."""
    with _lock, _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vin_records (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp     TEXT    NOT NULL,
                detected_vin  TEXT    NOT NULL,
                raw_vin       TEXT,
                model         TEXT,
                confidence    REAL    NOT NULL,
                image_path    TEXT
            )
            """
        )
        _migrate(conn)   # mavjud (eski) bazalar uchun
        conn.execute("CREATE INDEX IF NOT EXISTS idx_vin_ts ON vin_records(timestamp)")
    log.info(f"Ma'lumotlar bazasi tayyor: {DB_PATH}")


def insert_record(timestamp: str, vin: str, confidence: float,
                  image_path: Optional[str],
                  raw_vin: Optional[str] = None,
                  model: Optional[str] = None) -> int:
    """
    Yangi VIN yozuvini qo'shadi.
      vin       -> validated (saralangan) VIN
      raw_vin   -> xom OCR natijasi (audit)
      model     -> QY | BL7M
    """
    with _lock, _connect() as conn:
        cur = conn.execute(
            "INSERT INTO vin_records "
            "(timestamp, detected_vin, raw_vin, model, confidence, image_path) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (timestamp, vin, raw_vin, model, confidence, image_path),
        )
        return int(cur.lastrowid)


def get_records(limit: int = 500, order: str = "DESC",
                sort_by: str = "timestamp") -> List[Dict]:
    """Yozuvlarni qaytaradi. /history sahifasi uchun (saralash bilan)."""
    sort_col = sort_by if sort_by in {"timestamp", "detected_vin", "confidence", "id", "model"} else "timestamp"
    direction = "DESC" if order.upper() == "DESC" else "ASC"
    with _lock, _connect() as conn:
        rows = conn.execute(
            f"SELECT id, timestamp, detected_vin, raw_vin, model, confidence, image_path "
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
