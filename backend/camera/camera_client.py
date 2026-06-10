"""
SICK Lector 652 - Camera Client  (STRICT BLOB protokol)
=========================================================================
Tarmoq logikasi `lector652_pipeline` loyihasining ISHLAYDIGAN kodiga
asoslangan va quyidagi qat'iy qoidalar bilan mustahkamlangan:

  1. Mustahkam socket: SO_RCVBUF=16MB, setblocking(True) + SO_RCVTIMEO=2s
     (select() ISHLATILMAYDI), dinamik IP, SO_LINGER (RST on close).
  2. _recv_exactly(num_bytes): recv_into() while-loop — TCP fragmentatsiyada
     bayt yo'qolmaydi.
  3. ANIQ paket ketma-ketligi (live, mLIStart 0):
        4B  Magic (BLOB_STX = 02 02 02 02)
        4B  Payload uzunligi (Big-Endian, >I)
        NB  Payload
        1B  Checksum
  4. Dekodlash (AYLANTIRISH YO'Q):
        - Payload boshida 19 baytlik SICK sub-header -> payload[19:]
        - np.frombuffer(..., uint8) + cv2.imdecode(..., IMREAD_GRAYSCALE)
        - Xom (aylantirilmagan) matritsa qaytariladi.

CoLa A protokoli port 2111, BLOB port 2113.
"""

from __future__ import annotations

import errno
import socket
import struct
import sys
import time
import logging
from typing import Callable, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

DEFAULT_CONTROL_PORT = 2111
DEFAULT_BLOB_PORT    = 2113
DEFAULT_PASSWORD     = "A89A6E74"

_SOCK_BUFSIZE        = 16 * 1024 * 1024   # 16 MB TCP receive buffer
_BLOB_RCV_TIMEOUT    = 2.0                # SO_RCVTIMEO (sekund)

# Live BLOB framing
BLOB_STX             = b"\x02\x02\x02\x02"   # 4 baytlik magic
SICK_SUBHEADER_LEN   = 19                    # payload boshidagi sub-header
_MAX_PAYLOAD         = 50 * 1024 * 1024      # 50 MB sanity cheklov
_MAX_RESYNC_SCAN     = 8 * 1024 * 1024       # desync da STX qidirish chegarasi

# recv() timeout sifatida talqin qilinadigan OS errno lar
_TIMEOUT_ERRNOS = {errno.EAGAIN, errno.EWOULDBLOCK, errno.ETIMEDOUT, 10060}  # 10060=WSAETIMEDOUT


# -- CoLa A (control kanal, port 2111) ---------------------------------------

def _cola_pack(cmd: str) -> bytes:
    return b"\x02" + cmd.encode("ascii") + b"\x03"


def _cola_recv(sock: socket.socket, timeout: float = 5.0) -> str:
    """Control socketdan bitta CoLa javob o'qiydi. sSN (async event) larni skip qiladi."""
    sock.settimeout(timeout)
    buf = bytearray()
    while True:
        while b"\x03" in buf:
            end  = buf.index(0x03) + 1
            raw  = bytes(buf[:end])
            buf  = bytearray(buf[end:])
            text = raw.lstrip(b"\x02").rstrip(b"\x03").decode("latin1", errors="replace").strip()
            if text.startswith("sSN"):
                log.debug("sSN skip: %s", text[:80])
                continue
            return text
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("Control socket yopildi")
        buf.extend(chunk)


# -- Socket sozlash ----------------------------------------------------------

def _apply_rcvtimeo(sock: socket.socket, timeout_sec: float) -> None:
    """SO_RCVTIMEO ni OS darajasida o'rnatadi (Python select() siz)."""
    if sys.platform == "win32":
        # Windows: DWORD millisekund
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVTIMEO,
                        struct.pack("I", int(timeout_sec * 1000)))
    else:
        # Linux / macOS: struct timeval (sec, usec)
        sec  = int(timeout_sec)
        usec = int((timeout_sec - sec) * 1_000_000)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVTIMEO,
                        struct.pack("ll", sec, usec))


def _configure_blob_socket(sock: socket.socket) -> None:
    """BLOB socketga talab qilingan mustahkam sozlamalar."""
    # 16 MB qabul buferi (yuqori fps da fragmentatsiyani kamaytiradi)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, _SOCK_BUFSIZE)
    # RST on close — kamera bitta ulanish qabul qiladi, darhol bo'shatiladi
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    # Bloklovchi rejim + OS-level 2s timeout (select() ISHLATILMAYDI)
    sock.setblocking(True)
    _apply_rcvtimeo(sock, _BLOB_RCV_TIMEOUT)


# -- Qat'iy bayt o'qish ------------------------------------------------------

def _recv_exactly(sock: socket.socket, num_bytes: int) -> bytes:
    """
    Socketdan ANIQ num_bytes o'qiydi (TCP fragmentatsiyaga chidamli).

    recv_into(view, remaining) while-loopda — bironta bayt yo'qolmaydi.
    SO_RCVTIMEO (2s) tufayli recv bloklanib qolmaydi; timeoutda TimeoutError.
    """
    if num_bytes <= 0:
        return b""
    buf  = bytearray(num_bytes)
    view = memoryview(buf)
    pos  = 0
    while pos < num_bytes:
        try:
            n = sock.recv_into(view[pos:], num_bytes - pos)
        except socket.timeout as exc:                       # settimeout bo'lsa
            raise TimeoutError("recv timeout ({}/{} bayt)".format(pos, num_bytes)) from exc
        except OSError as exc:
            if exc.errno in _TIMEOUT_ERRNOS:                # SO_RCVTIMEO timeout
                raise TimeoutError("recv timeout ({}/{} bayt)".format(pos, num_bytes)) from exc
            raise ConnectionError("recv xatosi: {}".format(exc)) from exc
        if n == 0:
            raise ConnectionError("Socket yopildi ({}/{} bayt)".format(pos, num_bytes))
        pos += n
    return bytes(buf)


def _resync_to_stx(sock: socket.socket, window: bytes) -> None:
    """
    Desync holatida oqimni BLOB_STX (02 02 02 02) ga qayta moslaydi.
    `window` — endigina o'qilgan, lekin STX bo'lmagan 4 bayt.
    4 baytlik oynani 1 baytdan suradi, magic topilguncha.
    """
    win = bytearray(window)
    scanned = 0
    while bytes(win) != BLOB_STX:
        win.pop(0)
        win.extend(_recv_exactly(sock, 1))
        scanned += 1
        if scanned > _MAX_RESYNC_SCAN:
            raise ValueError("BLOB_STX topilmadi — oqim desync (resync chegarasi).")
    log.warning("Desync: %d bayt o'tkazib STX ga qayta moslandi.", scanned)


def _read_blob_frame(sock: socket.socket) -> bytes:
    """
    ANIQ ketma-ketlik bo'yicha bitta live BLOB frame o'qiydi va PAYLOAD ni
    qaytaradi (sub-header bilan birga; checksum tashlanadi).

        4B  Magic     (BLOB_STX)
        4B  Length    (Big-Endian, >I)
        NB  Payload
        1B  Checksum  (o'qiladi va tashlanadi)
    """
    # 1) Magic (4B)
    magic = _recv_exactly(sock, 4)
    if magic != BLOB_STX:
        _resync_to_stx(sock, magic)        # desyncdan tiklanish

    # 2) Payload uzunligi (4B, Big-Endian unsigned int)
    (length,) = struct.unpack(">I", _recv_exactly(sock, 4))
    if length <= 0 or length > _MAX_PAYLOAD:
        raise ValueError("Noto'g'ri payload uzunligi: {}".format(length))

    # 3) Payload (NB)
    payload = _recv_exactly(sock, length)

    # 4) Checksum (1B) — o'qiladi, lekin ishlatilmaydi (stream sinxron qoladi)
    _recv_exactly(sock, 1)

    return payload


# -- Payload -> tasvir (AYLANTIRISH YO'Q) ------------------------------------

def _decode_bmp_8bpp(bmp: bytes) -> Optional[np.ndarray]:
    """Zaxira: 8bpp grayscale BMP ni to'g'ridan-to'g'ri numpy ga (imdecode ishlamasa)."""
    try:
        if len(bmp) < 54 or bmp[:2] != b"BM":
            return None
        px_off = int.from_bytes(bmp[10:14], "little")
        raw_w  = int.from_bytes(bmp[18:22], "little")
        raw_h_ = int.from_bytes(bmp[22:26], "little", signed=True)
        bpp    = int.from_bytes(bmp[28:30], "little")
        if bpp != 8 or raw_w <= 0 or raw_h_ == 0:
            return None
        top_down   = raw_h_ < 0
        raw_h      = abs(raw_h_)
        row_stride = ((raw_w + 3) // 4) * 4
        px_end     = px_off + row_stride * raw_h
        if px_end > len(bmp):
            return None
        arr = np.frombuffer(bmp[px_off:px_end], dtype=np.uint8)
        img = arr.reshape(raw_h, row_stride)[:, :raw_w]
        if not top_down:
            # BMP satr tartibini to'g'rilaydi (orientatsiya o'zgarmaydi).
            img = np.ascontiguousarray(img[::-1])
        return img
    except Exception:
        return None


def decode_bmp(payload: bytes) -> Optional[np.ndarray]:
    """
    Live BLOB payload -> grayscale numpy (XOM, AYLANTIRISH YO'Q).

    Asosiy yo'l (talab bo'yicha):
        1. Birinchi 19 baytlik SICK sub-header tashlanadi: payload[19:]
        2. cv2.imdecode(np.frombuffer(..., uint8), IMREAD_GRAYSCALE)

    Zaxira (sub-header uzunligi boshqacha bo'lsa sindirmaslik uchun):
        3. payload ichidan 'BM' magic topib, o'sha joydan imdecode
        4. 8bpp BMP to'g'ridan-to'g'ri numpy

    Hech qaysi bosqichda 180° aylantirish QO'LLANILMAYDI.
    """
    if not payload or len(payload) <= SICK_SUBHEADER_LEN:
        return None

    # 1+2) ANIQ talab: 19 baytni o'tkazib, grayscale imdecode
    body = payload[SICK_SUBHEADER_LEN:]
    img = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is not None:
        return img                                  # AYLANTIRISH YO'Q

    # 3) Zaxira: 'BM' ni topib o'sha joydan imdecode (sub-header != 19 bo'lsa)
    idx = payload.find(b"BM")
    if idx >= 0:
        bmp = payload[idx:]
        img = cv2.imdecode(np.frombuffer(bmp, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img
        # 4) Zaxira: 8bpp raw BMP parse
        return _decode_bmp_8bpp(bmp)

    log.warning("decode_bmp: tasvir dekodlanmadi (payload len=%d).", len(payload))
    return None


# -- Asosiy klient ------------------------------------------------------------

class Lector652Client:
    """
    SICK Lector 652 CoLa A (port 2111) + BLOB (port 2113) klient.

    Live streaming:
      connect() -> start_stream() -> loop: read_stream_frame() -> stop_stream() -> disconnect()
    """

    def __init__(
        self,
        ip: str,
        control_port: int = DEFAULT_CONTROL_PORT,
        blob_port: int    = DEFAULT_BLOB_PORT,
        password: str     = DEFAULT_PASSWORD,
        on_log: Callable[[str], None] | None = None,
    ):
        self.ip           = ip                       # dinamik IP (UI dan)
        self.control_port = control_port
        self.blob_port    = blob_port
        self.password     = password
        self._on_log      = on_log or log.info
        self._ctrl: socket.socket | None = None
        self._blob: socket.socket | None = None
        self._live_active  = False
        self._last_recv_ms = 0.0

    # -- Ulanish --------------------------------------------------------------

    def connect(self, timeout: float = 8.0, retries: int = 3) -> None:
        """CoLa control ulanishi + handshake + BLOB socket (dinamik IP)."""
        last_err = None
        for attempt in range(1, retries + 1):
            self._info("Control -> {}:{} (urinish {}/{})".format(
                self.ip, self.control_port, attempt, retries))
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, _SOCK_BUFSIZE)
                s.settimeout(timeout)
                s.connect((self.ip, self.control_port))
                self._ctrl = s
                break
            except socket.timeout:
                last_err = ConnectionError(
                    "Kamera {}:{} javob bermadi ({}s, urinish {}/{}). "
                    "Kamera faqat bitta ulanish qabul qiladi — SOPAS ET yopiqligini tekshiring.".format(
                        self.ip, self.control_port, timeout, attempt, retries))
                if attempt < retries:
                    self._info("[!] Timeout — {}s kutib qayta urinilmoqda...".format(attempt * 3))
                    time.sleep(attempt * 3)
            except OSError as exc:
                last_err = ConnectionError("Kamera {}:{} ga ulanib bo'lmadi: {}".format(
                    self.ip, self.control_port, exc))
                break
        else:
            raise last_err

        # Parol (xato bo'lsa davom etamiz)
        try:
            self._cola("sMN CheckPassword 3 {}".format(self.password))
        except Exception as e:
            self._info("[!] CheckPassword: {} — davom etmoqda".format(e))

        # SOPAS ET tartibiga mos handshake ketma-ketligi
        self._cola("sRN DeviceIdent")
        self._cola("sEN DemoModeState 1")
        self._cola("sRN DemoModeState")
        self._cola("sMN GetBlobClientConfig")
        self._cola("sEN ImgBlobTransfer 1")
        self._cola("sEN LIIsActive 1")
        self._cola("sRN LIIsActive")
        self._cola("sEN VIStatDisp 1")
        self._cola("sRN VIStatDisp")
        self._cola("sEN VITmeStatDisp 1")
        self._cola("sRN VITmeStatDisp")
        resp = self._cola("sMN mDIGetEffImgSize")
        self._info("Effektiv rasm o'lchami: {}".format(resp))

        # BLOB socket (dinamik IP) — mustahkam sozlamalar bilan
        self._info("Blob -> {}:{}".format(self.ip, self.blob_port))
        self._blob = socket.create_connection((self.ip, self.blob_port), timeout=timeout)
        _configure_blob_socket(self._blob)
        self._info("Kamera tayyor (BLOB: 16MB buf, blocking + 2s SO_RCVTIMEO).")

    def disconnect(self) -> None:
        if self._live_active:
            try:
                self.stop_stream()
            except Exception:
                pass
        if self._ctrl:
            for cmd in ("sEN VITmeStatDisp 0", "sEN VIStatDisp 0",
                        "sEN LIIsActive 0", "sEN ImgBlobTransfer 0",
                        "sEN DemoModeState 0"):
                try:
                    self._cola(cmd)
                except Exception:
                    pass
            try:
                self._ctrl.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                self._ctrl.close()
            except Exception:
                pass
            self._ctrl = None
        if self._blob:
            try:
                self._blob.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                self._blob.close()
            except Exception:
                pass
            self._blob = None
        self._info("Ulanish yopildi.")

    @property
    def connected(self) -> bool:
        return self._ctrl is not None and self._blob is not None

    # -- Live streaming (mLIStart) --------------------------------------------

    def start_stream(self) -> None:
        """sMN mLIStart 0 — kamera avtomatik frame push qiladi."""
        if not self._blob or not self._ctrl:
            raise RuntimeError("Ulanish yo'q")
        self._info("Live streaming boshlanyapti (mLIStart 0)...")
        resp = self._cola_wait("sMN mLIStart 0", timeout=8.0)
        self._live_active = True
        self._info("mLIStart javob: {}".format(resp))

    def stop_stream(self) -> None:
        """sMN mLIStop — live streamingni to'xtatadi."""
        if not self._ctrl:
            return
        try:
            resp = self._cola_wait("sMN mLIStop", timeout=5.0)
            self._info("mLIStop javob: {}".format(resp))
        except Exception as e:
            self._info("[!] stop_stream: {}".format(e))
        self._live_active = False

    def read_stream_frame(self) -> bytes:
        """
        Bitta live frame PAYLOAD ini o'qiydi (qat'iy STX/len/payload/checksum
        ketma-ketligi). decode_bmp() bu payload boshidagi 19B sub-headerni
        tashlab tasvirga o'giradi.
        """
        if not self._blob:
            raise RuntimeError("Ulanish yo'q")
        t0 = time.monotonic()
        payload = _read_blob_frame(self._blob)
        self._last_recv_ms = (time.monotonic() - t0) * 1000.0
        return payload

    @property
    def last_recv_ms(self) -> float:
        return self._last_recv_ms

    # -- Ichki (CoLa) ---------------------------------------------------------

    def _cola(self, cmd: str) -> str:
        self._info("  C->K  {}".format(cmd))
        self._ctrl.sendall(_cola_pack(cmd))
        resp = _cola_recv(self._ctrl)
        self._info("  K->C  {}".format(resp))
        return resp

    def _cola_wait(self, cmd: str, timeout: float = 5.0) -> str:
        self._info("  C->K  {}".format(cmd))
        self._ctrl.sendall(_cola_pack(cmd))
        resp = _cola_recv(self._ctrl, timeout=timeout)
        self._info("  K->C  {}".format(resp))
        return resp

    def _info(self, msg: str) -> None:
        self._on_log(msg)
