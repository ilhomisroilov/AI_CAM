"""
============================================================
benchmark_ocr.py  —  EasyOCR vs PaddleOCR benchmark (real numbers)
============================================================
Bu skript IKKALA OCR engine ni bir XIL crop'lar to'plamida ishlatib,
KECHIKISH (latency), XOTIRA (memory) va ANIQLIK (accuracy) ni o'lchaydi.
Maqsad: taxminiy emas, o'z ma'lumotlaringizdagi HAQIQIY raqamlar.

Ishlatish:
    python tools/benchmark_ocr.py --crops data/crops --runs 3
    python tools/benchmark_ocr.py --crops some_dir --gt-from-filename

Ground-truth (ixtiyoriy, aniqlikni o'lchash uchun):
    * --gt-from-filename: fayl nomidagi VIN (masalan "1HGCM82633A004352_*.jpg")
      to'g'ri javob deb olinadi (AI_CAM crop'larini shu nomda saqlaydi).
    * yoki <crop>.txt fayl ichidagi matn.

Preprocessing (CLAHE+upscale+sharpen) ikkala engine uchun BIR XIL qo'llanadi,
shunda taqqoslash adolatli bo'ladi.

Eslatma: PaddleOCR o'rnatilmagan bo'lsa, faqat mavjud engine o'lchanadi.
"""
from __future__ import annotations

import argparse
import gc
import os
import re
import statistics
import time
from glob import glob
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception as e:  # pragma: no cover
    raise SystemExit(f"opencv kerak: {e}")

_VIN_RE = re.compile(r"[A-HJ-NPR-Z0-9]{17}")


# ------------------------------------------------------------------ utils
def normalize(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def preprocess(crop_bgr: np.ndarray, upscale_h: int = 96) -> np.ndarray:
    """AI_CAM bilan bir xil preprocess: gray -> upscale -> CLAHE -> unsharp."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
    if upscale_h and gray.shape[0] < upscale_h:
        s = upscale_h / float(gray.shape[0])
        gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    gray = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    return gray


def ground_truth(path: str, from_filename: bool) -> Optional[str]:
    txt = os.path.splitext(path)[0] + ".txt"
    if os.path.exists(txt):
        m = _VIN_RE.search(normalize(open(txt, encoding="utf-8").read()))
        if m:
            return m.group(0)
    if from_filename:
        m = _VIN_RE.search(normalize(os.path.basename(path)))
        if m:
            return m.group(0)
    return None


def peak_rss_mb() -> float:
    """Joriy jarayon peak RSS (MB). Platforma-bardosh."""
    try:
        import resource  # Linux/Mac
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return ru / 1024.0 if os.name != "nt" else ru / (1024.0 * 1024.0)
    except Exception:
        try:
            import psutil
            return psutil.Process().memory_info().rss / (1024.0 * 1024.0)
        except Exception:
            return float("nan")


# ------------------------------------------------------------------ engines
class EasyOCREngine:
    name = "EasyOCR"

    def __init__(self, gpu: bool):
        import easyocr
        self.r = easyocr.Reader(["en"], gpu=gpu)

    def read(self, img) -> str:
        res = self.r.readtext(img, detail=1, allowlist="ABCDEFGHJKLMNPRSTUVWXYZ0123456789",
                              paragraph=False)
        res.sort(key=lambda x: min(p[0] for p in x[0]))
        return normalize("".join(t for _, t, _ in res))


class PaddleEngine:
    name = "PaddleOCR"

    def __init__(self, gpu: bool, det: bool = True):
        from paddleocr import PaddleOCR
        self.det = det
        try:
            self.o = PaddleOCR(use_angle_cls=False, use_gpu=gpu, show_log=False, lang="en")
        except TypeError:
            self.o = PaddleOCR(use_textline_orientation=False,
                               device=("gpu" if gpu else "cpu"), lang="en")

    def read(self, img) -> str:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        try:
            res = self.o.ocr(img, det=self.det, cls=False)
        except TypeError:
            res = self.o.ocr(img)
        frags: List[Tuple[float, str]] = []
        page = res[0] if res else None
        for line in (page or []):
            try:
                if (isinstance(line, (list, tuple)) and len(line) == 2
                        and isinstance(line[0], (list, tuple))
                        and line[0] and isinstance(line[0][0], (list, tuple))):
                    x = min(p[0] for p in line[0]); frags.append((x, line[1][0]))
                else:
                    frags.append((0.0, line[0]))
            except Exception:
                continue
        frags.sort(key=lambda f: f[0])
        return normalize("".join(t for _, t in frags))


# ------------------------------------------------------------------ run
def bench(engine, imgs: List[np.ndarray], gts: List[Optional[str]], runs: int) -> dict:
    # warmup
    engine.read(imgs[0])
    lat: List[float] = []
    correct = total = 0
    for _ in range(runs):
        for img, gt in zip(imgs, gts):
            t0 = time.perf_counter()
            pred = engine.read(img)
            lat.append((time.perf_counter() - t0) * 1000.0)
            if gt is not None:
                total += 1
                correct += int(pred == gt)
    lat.sort()
    return {
        "engine": engine.name,
        "n_infer": len(lat),
        "p50_ms": round(statistics.median(lat), 1),
        "p95_ms": round(lat[int(len(lat) * 0.95) - 1], 1),
        "mean_ms": round(statistics.mean(lat), 1),
        "acc": (round(100.0 * correct / total, 1) if total else None),
        "acc_n": total,
        "peak_mb": round(peak_rss_mb(), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default="data/crops", help="crop rasmlar papkasi")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--gt-from-filename", action="store_true")
    ap.add_argument("--paddle-rec-only", action="store_true",
                    help="PaddleOCR ni rec-only (det=False) rejimida sinash")
    args = ap.parse_args()

    paths = sorted(sum([glob(os.path.join(args.crops, e))
                        for e in ("*.jpg", "*.jpeg", "*.png", "*.bmp")], []))
    if not paths:
        raise SystemExit(f"Crop topilmadi: {args.crops}")

    imgs, gts = [], []
    for p in paths:
        im = cv2.imread(p)
        if im is None:
            continue
        imgs.append(preprocess(im))
        gts.append(ground_truth(p, args.gt_from_filename))
    n_gt = sum(1 for g in gts if g)
    print(f"Crop: {len(imgs)} ta | ground-truth: {n_gt} ta | runs: {args.runs} | gpu: {args.gpu}\n")

    rows = []
    for factory in (lambda: EasyOCREngine(args.gpu),
                    lambda: PaddleEngine(args.gpu, det=not args.paddle_rec_only)):
        try:
            eng = factory()
        except Exception as e:
            print(f"[skip] {e}")
            continue
        rows.append(bench(eng, imgs, gts, args.runs))
        del eng; gc.collect()

    print(f"{'engine':<12}{'p50 ms':>9}{'p95 ms':>9}{'mean ms':>9}{'acc %':>8}{'peak MB':>10}")
    print("-" * 57)
    for r in rows:
        acc = "-" if r["acc"] is None else r["acc"]
        print(f"{r['engine']:<12}{r['p50_ms']:>9}{r['p95_ms']:>9}{r['mean_ms']:>9}{str(acc):>8}{r['peak_mb']:>10}")


if __name__ == "__main__":
    main()
