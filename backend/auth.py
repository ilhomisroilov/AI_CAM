"""
============================================================
auth.py  —  Yengil sessiya-asosli autentifikatsiya
============================================================
Zavod ichki tarmog'ida ruxsatsiz kirishni oldini olish uchun oddiy,
toza mexanizm:

  * Hardcoded zavod login (config.AUTH): admin / vin_factory2026
  * Login muvaffaqiyatli bo'lsa -> tasodifiy sessiya tokeni yaratiladi
    va HttpOnly cookie sifatida o'rnatiladi.
  * Tokenlar xotirada saqlanadi (server qayta ishga tushsa tozalanadi —
    LAN uchun yetarli). TTL: AUTH.session_ttl_sec.
  * Parol hmac.compare_digest bilan solishtiriladi (timing-safe).

Sessiya holati cookie orqali saqlanadi — shuning uchun Dashboard <-> History
o'tishlarida login holati YO'QOLMAYDI.
"""
from __future__ import annotations

import hmac
import secrets
import threading
import time
from typing import Dict, Optional

from .config import AUTH
from .logger import log

# token -> tugash vaqti (epoch). Xotiradagi sessiya do'koni.
_sessions: Dict[str, float] = {}
_lock = threading.Lock()


def check_credentials(username: Optional[str], password: Optional[str]) -> bool:
    """Login/parolni timing-safe tekshiradi."""
    u_ok = hmac.compare_digest((username or ""), AUTH.username)
    p_ok = hmac.compare_digest((password or ""), AUTH.password)
    return u_ok and p_ok


def create_session() -> str:
    """Yangi sessiya tokeni yaratadi va saqlaydi."""
    token = secrets.token_urlsafe(32)
    with _lock:
        _sessions[token] = time.time() + AUTH.session_ttl_sec
    return token


def is_valid(token: Optional[str]) -> bool:
    """Token mavjud va muddati o'tmaganligini tekshiradi."""
    if not AUTH.enabled:
        return True                      # auth o'chirilgan bo'lsa hamma ruxsatli
    if not token:
        return False
    with _lock:
        exp = _sessions.get(token)
        if exp is None:
            return False
        if time.time() > exp:
            _sessions.pop(token, None)   # muddati o'tgan — tozalaymiz
            return False
        return True


def destroy(token: Optional[str]) -> None:
    """Sessiyani bekor qiladi (logout)."""
    if not token:
        return
    with _lock:
        _sessions.pop(token, None)


def cleanup_expired() -> None:
    """Muddati o'tgan tokenlarni tozalaydi (ixtiyoriy chaqiriladi)."""
    now = time.time()
    with _lock:
        for tok in [t for t, exp in _sessions.items() if now > exp]:
            _sessions.pop(tok, None)
