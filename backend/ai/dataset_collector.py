"""
============================================================
dataset_collector.py  —  Avtomatik dataset yig'ish (Data Loop)
============================================================
Maqsad: dastlabki 10 ta rasmdan datasetni o'stirish.

YOLO past ishonch bilan (conf > collect_conf, masalan 0.40) plastinka
aniqlasa:
  * ASL (kesilmagan) kadr  -> dataset/collected_raw/vin_capture_YYYYMMDD_HHMMSS.jpg
  * mos bounding box'lar    -> ... .txt  (YOLO format: "cls cx cy w h", normalizatsiya)

Bu auto-labeling: keyin qo'lda ko'rib chiqib, modelni qayta o'qitish uchun
ishlatiladi.

MUHIM: barcha disk yozish ALOHIDA fon threadida (navbat orqali) bajariladi —
live stream FPS si tushmaydi. Navbat to'lsa eng eski element tashlanadi.
Bundan tashqari `min_interval_sec` throttle: sekundiga ~1 marta saqlanadi
(40 fps da diskni bosmaslik uchun).
"""
from __future__ import annotations

import os
import queue
import threading
import time
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..logger import log

# Detection turi: (x1, y1, x2, y2, conf)
Detection = Tuple[int, int, int, int, float]


class DatasetCollector:
    """Aniqlangan kadrlarni va YOLO yorliqlarini fon threadida saqlaydi."""

    def __init__(
        self,
        out_dir: str,
        collect_conf: float = 0.40,
        class_id: int = 0,
        min_interval_sec: float = 1.0,
        queue_max: int = 20,
        jpeg_quality: int = 95,
    ) -> None:
        self.out_dir = out_dir
        self.collect_conf = collect_conf
        self.class_id = class_id
        self.min_interval_sec = min_interval_sec
        self.jpeg_quality = jpeg_quality

        os.makedirs(self.out_dir, exist_ok=True)

        # (frame_kopiyasi, dets) navbati — fon thread yozadi
        self._queue: "queue.Queue" = queue.Queue(maxsize=queue_max)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._last_save = 0.0           # throttle uchun (monotonic)
        self.saved_count = 0

    # ---------------------------------------------------------------
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, name="dataset-collector", daemon=True
        )
        self._thread.start()
        log.info(f"Dataset auto-collection yoqildi -> {self.out_dir} (conf>{self.collect_conf})")

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        try:
            self._queue.put_nowait(None)        # stop sentinel
        except queue.Full:
            pass
        log.info(f"Dataset auto-collection to'xtatildi (jami saqlangan: {self.saved_count}).")

    # ---------------------------------------------------------------
    def maybe_collect(self, frame: np.ndarray, dets: List[Detection]) -> None:
        """
        Stream loop shu metodni chaqiradi. TEZ ishlaydi:
          - throttle (sekundiga 1 marta) va conf filtri shu yerda,
          - haqiqiy disk yozish fon threadda.
        frame: ASL kesilmagan kadr (BGR).
        """
        if not self._running or frame is None:
            return

        # Faqat collect_conf dan yuqori box'lar
        good = [d for d in dets if d[4] >= self.collect_conf]
        if not good:
            return

        # Throttle: oxirgi saqlashdan beri yetarli vaqt o'tdimi?
        now = time.monotonic()
        if (now - self._last_save) < self.min_interval_sec:
            return
        self._last_save = now

        # Navbatga faqat kopiya qo'yamiz (asl frame keyin o'zgarishi mumkin)
        item = (frame.copy(), good, frame.shape[1], frame.shape[0])
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            # Navbat to'lsa — eng eskini tashlab, yangisini qo'yamiz (FPS muhim)
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(item)
            except queue.Empty:
                pass

    # ---------------------------------------------------------------
    def _loop(self) -> None:
        while self._running:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if item is None:                    # stop sentinel
                break
            frame, dets, w, h = item
            self._write(frame, dets, w, h)

    def _write(self, frame: np.ndarray, dets: List[Detection], w: int, h: int) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"vin_capture_{ts}"
        img_path = os.path.join(self.out_dir, base + ".jpg")
        txt_path = os.path.join(self.out_dir, base + ".txt")

        # Bir sekundda bir nechta saqlash bo'lib qolsa, ustiga yozmaslik uchun
        # mikrosoniya qo'shamiz (throttle odatda buni oldini oladi).
        if os.path.exists(img_path):
            ts2 = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            base = f"vin_capture_{ts2}"
            img_path = os.path.join(self.out_dir, base + ".jpg")
            txt_path = os.path.join(self.out_dir, base + ".txt")

        try:
            # ASL kadrni saqlash (kesilmagan, xom orientatsiya)
            ok = cv2.imwrite(img_path, frame,
                             [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if not ok:
                log.error(f"Dataset: rasm saqlanmadi {img_path}")
                return

            # YOLO format yorliqlar: "cls cx cy w h" (0..1 normalizatsiya)
            lines = []
            for (x1, y1, x2, y2, _conf) in dets:
                cx = ((x1 + x2) / 2.0) / w
                cy = ((y1 + y2) / 2.0) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                # 0..1 oralig'iga qisish (xavfsizlik)
                cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
                bw, bh = min(max(bw, 0.0), 1.0), min(max(bh, 0.0), 1.0)
                lines.append(f"{self.class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

            self.saved_count += 1
            log.info(f"Dataset saqlandi #{self.saved_count}: {base}.jpg (+{len(lines)} box)")
        except Exception as exc:
            log.error(f"Dataset yozuv xatosi: {exc}")
