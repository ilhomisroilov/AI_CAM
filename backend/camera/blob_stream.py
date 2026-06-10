"""
DEPRECATED — bu modul taxminga asoslangan edi va ISHLATILMAYDI.

Haqiqiy BLOB frame o'qish logikasi (mLIStart live-push, 16B/STX framing,
8bpp BMP -> numpy) endi `camera_client.py` da:
  - Lector652Client.read_stream_frame()
  - decode_bmp()

Eski importlar buzilmasligi uchun qayta-eksport qoldirildi.
"""
from __future__ import annotations

from .camera_client import decode_bmp  # noqa: F401

__all__ = ["decode_bmp"]
