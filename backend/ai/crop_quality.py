"""
============================================================
crop_quality.py  —  Crop sifati baholash + deskew (yengil, OpenCV)
============================================================
YOLO bergan ROI croplarini OCR uchun saralash va to'g'rilash:

  * sharpness()  — Laplacian dispersiyasi (blur o'lchovi; yuqori = aniqroq).
  * brightness_score() — yorug'lik balansi (0..1).
  * tilt_angle()/deskew() — minAreaRect orqali qiyalikni aniqlab to'g'rilash.
  * quality_score() — yagona ball: conf + sharpness + brightness + size − tilt.

Maqsad: ko'p kadrdan ENG YAXSHISINI tanlash va qiya/blur kadrlarni e'tiborsiz
qoldirish. Og'ir model YO'Q — faqat OpenCV (real-time uchun arzon).
"""
from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np


def _to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def sharpness(img: np.ndarray) -> float:
    """Laplacian dispersiyasi — blur qancha kam bo'lsa, shuncha yuqori."""
    gray = _to_gray(img)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_score(img: np.ndarray) -> float:
    """O'rtacha yorug'lik ideal (~125) ga qanchalik yaqinligini 0..1 da qaytaradi."""
    gray = _to_gray(img)
    mean = float(gray.mean())
    return max(0.0, 1.0 - abs(mean - 125.0) / 125.0)


def tilt_angle(img: np.ndarray) -> float:
    """
    Matn qiyaligini (gradus) baholaydi: Otsu bilan binarize -> minAreaRect.
    [-45, 45] oralig'iga normallashtiriladi. Matn topilmasa 0.
    """
    gray = _to_gray(img)
    try:
        thr = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thr > 0))
        if coords.shape[0] < 50:
            return 0.0
        angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
        if angle < -45:
            angle = 90.0 + angle
        elif angle > 45:
            angle = angle - 90.0
        return float(angle)
    except Exception:
        return 0.0


def rotate(img: np.ndarray, angle: float) -> np.ndarray:
    """Markaz atrofida burish (chekkalar takrorlanadi — qora chiziq yo'q)."""
    if abs(angle) < 0.1:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def deskew(img: np.ndarray, max_angle: float = 30.0) -> np.ndarray:
    """
    Matnni gorizontalga to'g'rilaydi. Faqat kichik (|angle|<=max_angle) qiyalik
    tuzatiladi — 90° aylanishlardan saqlanish uchun.
    """
    angle = tilt_angle(img)
    if abs(angle) < 1.0 or abs(angle) > max_angle:
        return img
    return rotate(img, angle)


def quality_score(crop_bgr: np.ndarray, conf: float) -> Tuple[float, Dict[str, float]]:
    """
    Crop sifatining yagona balli (croplarni saralash uchun). Tarkibi:
        0.40*conf + 0.30*sharpness + 0.15*brightness + 0.15*size − tilt_penalty
    Qaytadi: (score, metrics_dict)
    """
    gray = _to_gray(crop_bgr)
    sh = sharpness(gray)
    sh_n = min(sh / 300.0, 1.0)              # ~300 dan yuqorisi "aniq" deb olinadi
    br = brightness_score(gray)
    h, w = gray.shape[:2]
    sz_n = min((h * w) / (120.0 * 40.0), 1.0)   # taxminiy plastinka o'lchamiga nisbatan
    ang = abs(tilt_angle(gray))
    tilt_pen = min(ang / 45.0, 1.0) * 0.25
    score = 0.40 * float(conf) + 0.30 * sh_n + 0.15 * br + 0.15 * sz_n - tilt_pen
    return float(score), {"sharp": sh, "bright": br, "tilt": ang, "size": float(h * w)}
