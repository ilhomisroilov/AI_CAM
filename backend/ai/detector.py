"""
============================================================
detector.py  —  YOLOv8n VIN plastinka aniqlovchi
============================================================
Vazifa:
  * Har bir kadrda metall VIN plastinkasini aniqlash (yolov8n).
  * Live stream uchun bbox (kuzatuv ramkasi) chizish.
  * Yuqori ishonch bilan aniqlangan ROI ni kesib berish (OCR uchun).

Model: o'qitilgan best.pt
(runs/detect/vin_plate_model/weights/best.pt — config.TRAINED_MODEL_PATH).
Agar bu fayl topilmasa, dastur ishdan to'xtamaydi — standart yolov8n.pt ga
qaytadi va ogohlantiradi. Modelni o'qitish bo'yicha TRAINING_GUIDE.md ga qarang.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..config import DETECTION
from ..logger import log

# MES rang palitrasi (BGR) — bbox uchun teal aksent
_TEAL_BGR = (156, 188, 26)        # #1ABC9C -> BGR
_BLUE_BGR = (136, 60, 31)         # #1F3C88 -> BGR


class PlateDetector:
    """YOLOv8n asosida VIN plastinka detektori."""

    def __init__(self) -> None:
        self.model = None
        self.device = DETECTION.device      # _load_model uni aniqlashtiradi
        self._load_model()

    def _resolve_device(self) -> str:
        """
        device ni aniqlaydi:
          "auto"  -> NVIDIA GPU bo'lsa "cuda:0", aks holda "cpu"
          "cuda*" -> CUDA mavjud bo'lmasa "cpu" ga qaytadi
        """
        dev = str(DETECTION.device).lower()
        try:
            import torch
            cuda_ok = torch.cuda.is_available()
        except Exception:
            return "cpu"

        if dev == "auto":
            if cuda_ok:
                log.info("device=auto -> CUDA (NVIDIA GPU) topildi: cuda:0")
                return "cuda:0"
            log.info("device=auto -> GPU topilmadi, CPU ishlatiladi.")
            return "cpu"

        if dev.startswith("cuda") and not cuda_ok:
            log.warning("CUDA mavjud emas — CPU ishlatiladi.")
            return "cpu"
        return DETECTION.device

    @staticmethod
    def _make_torch_load_safe() -> None:
        """
        torch >= 2.6 da `torch.load` default `weights_only=True` bo'ldi —
        bu Ultralytics checkpoint (.pt) yuklashni buzadi:
            "WeightsUnpickler error / Unsupported global ..."
        Bu — YOLO yuklanmaslik sababining eng keng tarqalgani.

        Biz O'ZIMIZNING ishonchli modelimizni yuklayapmiz, shuning uchun:
          1) Ultralytics klasslarini safe-globals ro'yxatiga qo'shamiz (toza yo'l)
          2) Zaxira: torch.load ni weights_only=False ga patch qilamiz.
        """
        try:
            import torch
        except Exception as exc:
            log.error(f"PyTorch import qilinmadi: {exc}")
            return

        # 1) Ultralytics + torch.nn klasslarini ishonchli deb belgilash
        try:
            import torch.serialization as ts
            safe = []
            try:
                from ultralytics.nn.tasks import DetectionModel
                safe.append(DetectionModel)
            except Exception:
                pass
            try:
                import torch.nn as nn
                safe += [nn.Sequential, nn.ModuleList]
            except Exception:
                pass
            if safe and hasattr(ts, "add_safe_globals"):
                ts.add_safe_globals(safe)
        except Exception:
            pass

        # 2) Zaxira: weights_only=False ni majburlash (faqat ishonchli fayllar uchun)
        if not getattr(torch.load, "_aicam_patched", False):
            _orig_load = torch.load

            def _patched_load(*args, **kwargs):
                kwargs.setdefault("weights_only", False)
                return _orig_load(*args, **kwargs)

            _patched_load._aicam_patched = True   # type: ignore[attr-defined]
            torch.load = _patched_load            # type: ignore[assignment]
            log.info("torch.load weights_only=False ga sozlandi (Ultralytics mosligi).")

    def _load_model(self) -> None:
        """Modelni yuklaydi. Maxsus model yo'q bo'lsa standart yolov8n ga qaytadi."""
        # MUHIM: YOLO import/yuklashdan OLDIN torch.load ni mos qilamiz
        self._make_torch_load_safe()

        try:
            from ultralytics import YOLO        # kech import — ishga tushish tezligini saqlaydi
        except Exception as exc:
            log.error(f"ultralytics import qilinmadi: {exc}. YOLO o'chirilgan. "
                      f"Tekshiring: pip install -r requirements.txt")
            return

        path = DETECTION.model_path
        if not os.path.exists(path):
            log.warning(
                f"O'qitilgan model topilmadi: {path}. "
                f"Standart 'yolov8n.pt' yuklanmoqda (faqat sinov uchun). "
                f"best.pt yo'lini tekshiring yoki TRAINING_GUIDE.md bo'yicha model o'qiting."
            )
            path = "yolov8n.pt"

        self.device = self._resolve_device()
        try:
            self.model = YOLO(path)
            try:
                self.model.to(self.device)
            except Exception as exc:
                log.warning(f"'{self.device}' ga o'tkazib bo'lmadi: {exc} — CPU ga qaytildi.")
                self.device = "cpu"
                self.model.to("cpu")
            log.info(f"YOLO model yuklandi: {path} (device={self.device})")
        except Exception as exc:
            # Aniq sababni ko'rsatamiz (weights_only, mos kelmaydigan torch va h.k.)
            log.error(f"YOLO model yuklanmadi: {type(exc).__name__}: {exc}")
            log.error("Maslahat: 'pip install -U ultralytics' (yangi torch bilan moslik) "
                      "yoki best.pt fayli butunligini tekshiring.")
            self.model = None

    @property
    def ready(self) -> bool:
        return self.model is not None

    # ---------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Kadrda plastinkalarni aniqlaydi.
        Qaytaradi: [(x1, y1, x2, y2, conf), ...]
        """
        if self.model is None:
            return []
        try:
            results = self.model.predict(
                frame,
                conf=DETECTION.conf_threshold,
                iou=DETECTION.iou_threshold,
                verbose=False,
                device=self.device,
            )
        except Exception as exc:
            log.error(f"YOLO predict xatosi: {exc}")
            return []

        dets: List[Tuple[int, int, int, int, float]] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                dets.append((x1, y1, x2, y2, conf))
        return dets

    # ---------------------------------------------------------------
    @staticmethod
    def draw_boxes(frame: np.ndarray, dets: List[Tuple[int, int, int, int, float]]) -> np.ndarray:
        """Live stream uchun bbox + ishonch yorlig'ini chizadi (real-time skeleton)."""
        out = frame.copy()
        for (x1, y1, x2, y2, conf) in dets:
            color = _TEAL_BGR if conf >= DETECTION.ocr_trigger_conf else _BLUE_BGR
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"VIN {conf * 100:.0f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(out, label, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return out

    # ---------------------------------------------------------------
    @staticmethod
    def crop_roi(frame: np.ndarray,
                 det: Tuple[int, int, int, int, float]) -> Optional[np.ndarray]:
        """ROI ni padding bilan kesib oladi (OCR uchun)."""
        x1, y1, x2, y2, _ = det
        p = DETECTION.crop_padding
        h, w = frame.shape[:2]
        x1 = max(0, x1 - p); y1 = max(0, y1 - p)
        x2 = min(w, x2 + p); y2 = min(h, y2 + p)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2].copy()

    @staticmethod
    def crop_with_margin(frame: np.ndarray,
                         det: Tuple[int, int, int, int, float],
                         frac: float) -> Optional[np.ndarray]:
        """
        ROI ni NISBIY margin (frac) bilan kengaytirib kesadi — YOLO box ba'zan
        belgilarni kesib qo'yadi, shuning uchun OCR uchun atrofga joy qo'shamiz.
        frac=0.15 -> har tomondan box o'lchamining 15% i qo'shiladi.
        """
        x1, y1, x2, y2, _ = det
        bw = x2 - x1
        bh = y2 - y1
        mx = int(bw * frac)
        my = int(bh * frac)
        h, w = frame.shape[:2]
        x1 = max(0, x1 - mx); y1 = max(0, y1 - my)
        x2 = min(w, x2 + mx); y2 = min(h, y2 + my)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2].copy()

    @staticmethod
    def best_detection(dets: List[Tuple[int, int, int, int, float]]
                       ) -> Optional[Tuple[int, int, int, int, float]]:
        """Eng yuqori ishonchli aniqlovni qaytaradi (OCR trigger uchun)."""
        if not dets:
            return None
        best = max(dets, key=lambda d: d[4])
        return best if best[4] >= DETECTION.ocr_trigger_conf else None
