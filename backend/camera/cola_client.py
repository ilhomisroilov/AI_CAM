"""
DEPRECATED — bu modul taxminga asoslangan edi va ISHLATILMAYDI.

Haqiqiy, sinovdan o'tgan SICK Lector 652 tarmoq logikasi endi
`camera_client.py` (Lector652Client) da. Pipeline o'shani import qiladi.

Eski importlar buzilmasligi uchun bu yerda faqat qayta-eksport qoldirildi.
"""
from __future__ import annotations

from .camera_client import Lector652Client  # noqa: F401

__all__ = ["Lector652Client"]
