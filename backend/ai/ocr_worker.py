"""
============================================================
ocr_worker.py  —  PaddleOCR fon (background) navbat ishchisi
============================================================
Maqsad: YOLO bergan crop'lardan VIN ni YUQORI ANIQLIK bilan o'qish,
ishonch chegaralarini boshqarish, va live oqim lag bermasligi uchun
tezlikni nazorat qilish.

OCR engine: PaddleOCR (EasyOCR dan ko'chirildi). Tashqi interfeys
o'zgarmagan — OCRWorker, submit/start/stop/get_stats, on_result, va VIN
tekshirish funksiyalari avvalgidek. Faqat OCR engine almashtirildi.

Asoslar (lector652 VIN-OCR metodologiyasi):
  * OCR ALOHIDA fon threadida navbat (Queue) orqali — kamera FPS tushmaydi.
  * Anti-duplicate: bir xil VIN duplicate_window_sec ichida qayta yozilmaydi.
  * VIN validatsiya: 17 belgi, I/O/Q yo'q.

Aniqlik uchun:
  * Crop -> grayscale -> CLAHE -> kattalashtirish (upscale) -> yengil sharpen.
  * PaddleOCR (det+rec yoki rec-only) -> (box, text, conf) bo'laklar.
  * Bo'laklar bbox X bo'yicha CHAPDAN-O'NGGA tartiblanadi (to'g'ri ketma-ketlik).
  * VIN charset (A-Z0-9, I/O/Q yo'q) post-filter orqali (normalize_vin).
  * Per-fragment va umumiy ishonch chegaralari.

Tezlik / nazorat:
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

from ..config import OCR
from ..logger import log

# on_result(vin: str, confidence: float, crop_bgr: np.ndarray)
ResultCallback = Callable[[str, float, np.ndarray], None]

# Standart avtomobil VIN strukturasi: aniq 17 belgi, alfanumerik,
# I, O, Q HARFLARI YO'Q (ISO 3779). Regex bilan qat'iy tekshiramiz.
_VIN_RE = re.compile(r"^[A-HJ-NPR-Z0-9]{17}$")


def normalize_vin(text: str) -> str:
    """OCR matnini tozalaydi: bo'shliq/belgilarni olib tashlaydi, katta harf, A-Z0-9."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def verify_vin(text: str) -> Tuple[bool, str]:
    """
    Standart VIN tekshiruvi.
      1. Bo'shliq va begona belgilarni tozalaydi (normalize).
      2. Qat'iy regex: ^[A-HJ-NPR-Z0-9]{17}$ (17 belgi, I/O/Q yo'q).
    Qaytadi: (ok, tozalangan_vin)
    """
    cleaned = normalize_vin(text)
    return bool(_VIN_RE.match(cleaned)), cleaned


def is_valid_vin(vin: str) -> bool:
    """Qat'iy VIN tekshiruvi (regex asosida)."""
    return bool(_VIN_RE.match(normalize_vin(vin)))


def _fragment_left_x(box) -> float:
    """OCR bbox (4 nuqta) dan eng chap X koordinatani oladi (tartiblash uchun)."""
    try:
        return float(min(pt[0] for pt in box))
    except Exception:
        return 0.0


# Synthetic box (rec-only rejim uchun — bitta qator, tartib ahamiyatsiz)
_FULL_BOX = [[0, 0], [1, 0], [1, 1], [0, 1]]


class _PaddleEngine:
    """
    PaddleOCR wrapper. read(img) -> [(box, text, conf), ...] — EasyOCR bilan
    AYNAN bir xil chiqish shakli, shu sababdan OCRWorker mantig'i o'zgarmaydi.

    Versiyalararo bardoshli: init va natija parsing eski (2.x) va yangi (3.x)
    PaddleOCR API larini qo'llab-quvvatlaydi.
    """

    def __init__(self) -> None:
        from paddleocr import PaddleOCR
        self.det = bool(OCR.paddle_det)
        gpu = OCRWorker._resolve_gpu()

        # PaddleOCR konstruktori versiyalar bo'yicha farq qiladi — bardoshli init
        common = dict(lang=OCR.lang)
        try:
            # 2.x API: use_angle_cls + use_gpu + show_log
            self._ocr = PaddleOCR(
                use_angle_cls=OCR.use_angle_cls,
                use_gpu=gpu,
                drop_score=OCR.drop_score,
                rec_batch_num=OCR.rec_batch_num,
                det_limit_side_len=OCR.det_limit_side_len,
                show_log=False,
                **common,
            )
        except TypeError:
            # 3.x API: use_textline_orientation + device (use_gpu/show_log olib tashlangan)
            self._ocr = PaddleOCR(
                use_textline_orientation=OCR.use_angle_cls,
                device=("gpu" if gpu else "cpu"),
                **common,
            )
        self.gpu = gpu

    def read(self, img: np.ndarray) -> List[Tuple[list, str, float]]:
        # PaddleOCR 3 kanalli (BGR) kutadi — grayscale bo'lsa o'tkazamiz
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        try:
            res = self._ocr.ocr(img, det=self.det, cls=OCR.use_angle_cls)
        except TypeError:
            # Yangi API .ocr() imzosi (det/cls argsiz)
            res = self._ocr.ocr(img)
        return self._parse(res)

    @staticmethod
    def _parse(res) -> List[Tuple[list, str, float]]:
        """PaddleOCR natijasini (box, text, conf) ro'yxatiga keltiradi."""
        out: List[Tuple[list, str, float]] = []
        if not res:
            return out
        page = res[0] if isinstance(res, (list, tuple)) and len(res) > 0 else res
        if not page:
            return out
        for line in page:
            try:
                # det+rec:  line = [box(4 nuqta), (text, score)]
                # rec-only: line = (text, score)
                if (isinstance(line, (list, tuple)) and len(line) == 2
                        and isinstance(line[0], (list, tuple)) and line[0]
                        and isinstance(line[0][0], (list, tuple))):
                    box = line[0]
                    text, score = line[1][0], line[1][1]
                else:
                    text, score = line[0], line[1]
                    box = _FULL_BOX
                out.append((box, str(text), float(score)))
            except Exception:
                continue
        return out


class OCRWorker:
    """EasyOCR ni fon threadida navbat orqali ishlatadi (aniqlik + nazorat)."""

    def __init__(self, on_result: ResultCallback) -> None:
        self.on_result = on_result
        self._queue: "queue.Queue" = queue.Queue(maxsize=OCR.queue_maxsize)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._engine: Optional[_PaddleEngine] = None   # PaddleOCR engine (lazy)

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
        log.info("OCR worker ishga tushdi (PaddleOCR, fon navbat).")

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
        """Bitta crop (orqaga moslik) — ko'p-kadrli ish sifatida o'raladi."""
        self.submit_frames([crop_bgr])

    def submit_frames(self, crops: list) -> None:
        """
        Bitta plastinka hodisasi uchun ENG YAXSHI croplar ro'yxati (multi-frame
        fusion). OCR ularda ovoz berib yagona VIN chiqaradi.
        """
        if not self._enabled or not crops:
            return
        try:
            self._queue.put_nowait(crops)
        except queue.Full:
            try:
                self._queue.get_nowait()         # eng eskini tashlaymiz
                self._queue.put_nowait(crops)
                with self._stats_lock:
                    self._dropped += 1
            except queue.Empty:
                pass

    # ===============================================================
    # PaddleOCR engine (lazy)
    # ===============================================================
    @staticmethod
    def _resolve_gpu() -> bool:
        """
        GPU so'ralgan bo'lsa, CUDA haqiqatan mavjudligini tekshiradi.
        Ubuntu+NVIDIA serverda True; CUDA yo'q bo'lsa xavfsiz CPU ga qaytadi.
        """
        if not OCR.use_gpu:
            return False
        try:
            import torch
            if torch.cuda.is_available():
                return True
            log.warning("OCR.use_gpu=True, lekin CUDA topilmadi — PaddleOCR CPU da ishlaydi.")
            return False
        except Exception:
            return False

    def _ensure_reader(self) -> bool:
        if self._engine is not None:
            return True
        try:
            log.info("PaddleOCR modeli yuklanmoqda... (birinchi marta sekin/yuklab oladi).")
            self._engine = _PaddleEngine()
            log.info(f"PaddleOCR tayyor (gpu={self._engine.gpu}, det={self._engine.det}, "
                     f"lang={OCR.lang}).")
            return True
        except Exception as exc:
            log.error(f"PaddleOCR yuklanmadi: {exc}")
            return False

    def _loop(self) -> None:
        if not self._ensure_reader():
            self._running = False
            return
        while self._running:
            try:
                job = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if job is None:                      # stop sentinel
                break
            # Throttle: ketma-ket OCR orasidagi minimal vaqt
            if OCR.min_interval_sec > 0:
                dt = time.monotonic() - self._last_run
                if dt < OCR.min_interval_sec:
                    continue                     # bu ishni o'tkazib yuboramiz (lag himoyasi)
            self._last_run = time.monotonic()
            self._process_job(job)

    # ===============================================================
    # Preprocess — aniqlik uchun (kattalashtirish + CLAHE + sharpen)
    # ===============================================================
    def _preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr

        # Kichik etched belgilarni kattalashtirish (recognizer aniqligi oshadi)
        if OCR.upscale_height and gray.shape[0] < OCR.upscale_height:
            scale = OCR.upscale_height / float(gray.shape[0])
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Bilateral — chekkalarni saqlab shovqin/blurni kamaytiradi (etched metal)
        if OCR.bilateral:
            gray = cv2.bilateralFilter(gray, OCR.bilateral_d,
                                       OCR.bilateral_sigma, OCR.bilateral_sigma)

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
    def _read_vin(self, crop: np.ndarray) -> Tuple[str, float]:
        """Bitta crop/variantni OCR qiladi -> (vin, avg_conf). Yon ta'sirsiz."""
        img = self._preprocess(crop)
        results = self._engine.read(img)
        frags = [(box, txt, float(conf)) for (box, txt, conf) in results
                 if float(conf) >= OCR.min_char_confidence and txt.strip()]
        frags.sort(key=lambda r: _fragment_left_x(r[0]))
        vin = normalize_vin("".join(f[1] for f in frags))
        conf = float(np.mean([f[2] for f in frags])) if frags else 0.0
        return vin, conf

    def _variants(self, crop: np.ndarray):
        """Crop variantlari: asl -> deskew -> ±burilish (retry uchun)."""
        yield crop
        if not OCR.retry_enabled:
            return
        from .crop_quality import deskew, rotate
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        yield deskew(gray)
        for ang in OCR.retry_rotations:
            yield rotate(gray, float(ang))

    def _read_with_retry(self, crop: np.ndarray, budget: int) -> Tuple[str, float, int]:
        """
        Crop uchun variantlarni ketma-ket sinaydi (budget chegarasida).
        Yaroqli VIN topilsa darhol qaytaradi; aks holda eng yuqori ishonchli
        nomzodni qaytaradi. -> (vin, conf, used_attempts)
        """
        used = 0
        best = ("", 0.0)
        for var in self._variants(crop):
            if used >= budget:
                break
            try:
                vin, conf = self._read_vin(var)
            except Exception as exc:
                log.error(f"PaddleOCR xatosi: {exc}")
                used += 1
                continue
            used += 1
            if vin and conf > best[1]:
                best = (vin, conf)
            if is_valid_vin(vin):
                return vin, conf, used
        return best[0], best[1], used

    @staticmethod
    def _fuse(candidates: List[Tuple[str, float, np.ndarray]]) -> Tuple[str, float, np.ndarray]:
        """
        Ko'p kadr natijalarini birlashtirish: KO'PCHILIK OVOZI; tenglikda — eng
        yuqori jami ishonch. Qaytadi: (vin, conf, eng_yaxshi_crop).
        """
        from collections import Counter
        votes = Counter(v for v, _, _ in candidates)
        top_count = max(votes.values())
        tied = [v for v, c in votes.items() if c == top_count]
        if len(tied) > 1:
            winner = max(tied, key=lambda vv: sum(cf for v, cf, _ in candidates if v == vv))
        else:
            winner = tied[0]
        entries = [(cf, cr) for v, cf, cr in candidates if v == winner]
        conf, crop = max(entries, key=lambda e: e[0])
        return winner, conf, crop

    def _process_job(self, crops: list) -> None:
        """
        Bitta plastinka hodisasi: ENG YAXSHI croplarda OCR + retry + fusion.
        OCR chaqiruvlari max_ocr_attempts bilan cheklangan (lag himoyasi).
        """
        t0 = time.perf_counter()
        candidates: List[Tuple[str, float, np.ndarray]] = []
        attempts = 0
        last_read = ""
        for crop in crops:
            if attempts >= OCR.max_ocr_attempts:
                break
            budget = OCR.max_ocr_attempts - attempts
            vin, conf, used = self._read_with_retry(crop, budget)
            attempts += used
            if vin:
                last_read = vin
            if vin and is_valid_vin(vin):
                candidates.append((vin, conf, crop))

        dt_ms = (time.perf_counter() - t0) * 1000.0

        with self._stats_lock:
            self._processed += 1
            self._last_ms = dt_ms
            self._ms_hist.append(dt_ms)
            self._last_vin_read = candidates[0][0] if candidates else last_read

        # --- Hech qaysi kadrdan yaroqli VIN chiqmadi ---
        if not candidates:
            with self._stats_lock:
                self._rejected += 1
            log.info(f"OCR: {len(crops)} kadrdan yaroqli VIN topilmadi "
                     f"('{last_read}', {attempts} urinish, {dt_ms:.0f} ms).")
            return

        # --- Fusion: ko'pchilik ovozi ---
        vin, conf, best_crop = self._fuse(candidates)
        with self._stats_lock:
            self._last_conf = conf

        if conf < self._min_conf:
            with self._stats_lock:
                self._rejected += 1
            log.info(f"OCR: ishonch past ({conf:.2f}<{self._min_conf:.2f}) — '{vin}' rad etildi.")
            return

        # --- Anti-duplicate ---
        if self._is_duplicate(vin):
            log.info(f"DUPLICATE: '{vin}' yaqinda o'qilgan — e'tiborsiz qoldirildi.")
            return

        with self._stats_lock:
            self._accepted += 1
        log.info(f"VIN aniqlandi: {vin} (conf={conf:.2f}, {len(candidates)}/{len(crops)} "
                 f"kadr rozi, {attempts} urinish, {dt_ms:.0f} ms)")
        try:
            self.on_result(vin, conf, best_crop)
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
