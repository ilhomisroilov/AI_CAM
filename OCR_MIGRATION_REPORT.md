# OCR Migration Report — EasyOCR → PaddleOCR (AI_CAM)

**Scope:** Replace the OCR engine in the existing, working AI_CAM pipeline (YOLOv8n crop → OCR → VIN verify) with PaddleOCR, preserving the architecture. No redesign. Analysis is based on the live working copy at `C:\Users\II4028\Documents\Projects\AI_CAM` (the current source of truth; equal to / ahead of the GitHub snapshot).

---

## 1. Repository analysis

**Pipeline (unchanged by this migration):**

```
SICK Lector652 → camera_client (mLIStart BLOB) → decode_bmp (raw frame)
   → pipeline._stream_loop → YOLOv8n detect (best.pt)
   → single-shot trigger (conf ≥ 0.95, debounce/cooldown)
   → crop ROI → OCRWorker.submit() [background queue]
   → OCR read → left-to-right assemble → VIN regex verify → SQLite + crop save
```

**Key components touching OCR:**

| File | Role | OCR coupling |
|---|---|---|
| `backend/ai/ocr_worker.py` | Background OCR worker (queue, preprocess, read, verify, anti-dup, stats) | **All engine calls live here** |
| `backend/config.py` `OCRConfig` | OCR tunables | Engine params |
| `backend/pipeline.py` | Calls `ocr.submit(roi)`, consumes `on_result(vin, conf, crop)` | **Engine-agnostic** (interface only) |
| `backend/server.py` | Exposes `ocr.get_stats()` in `/api/status` | **Engine-agnostic** |
| `requirements.txt`, `README.md`, `TRAINING_GUIDE.md` | deps + docs | Names the engine |

**Critical observation:** the worker already isolates the engine behind one method, `_read(img) → List[(box, text, conf)]`, and everything downstream (`_process`: per-fragment filtering, **left-to-right ordering by bbox-x**, charset normalize, VIN regex verify, confidence gates, anti-duplicate, monitoring stats) consumes that uniform shape. **This made the migration a localized swap rather than a refactor.**

**Operational note found in logs:** `ai_cam.log` shows EasyOCR first-run model download failing behind a corporate proxy:
`EasyOCR yuklanmadi: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] ...>`. PaddleOCR also downloads models on first run, so the **offline model placement** step (§7) matters for the factory environment regardless of engine.

---

## 2. Every EasyOCR usage (before migration)

- `ocr_worker.py`: `import easyocr`, `easyocr.Reader(langs, gpu=)`, `reader.readtext(..., allowlist, decoder, beamWidth, text_threshold, low_text, link_threshold, mag_ratio, contrast_ths, adjust_contrast, paragraph=False)`, plus `_resolve_gpu`, `_ensure_reader`, `_read`, log strings.
- `config.py` `OCRConfig`: EasyOCR-only params (`languages`, `decoder`, `beam_width`, `text_threshold`, `low_text`, `link_threshold`, `mag_ratio`, `contrast_ths`, `adjust_contrast`).
- `requirements.txt`: `easyocr==1.7.2` (+ the opencv-headless duplicate it pulls in).
- `README.md`, `TRAINING_GUIDE.md`: prose mentions.

---

## 3. Migration plan (executed)

1. Introduce a thin `_PaddleEngine` wrapper inside `ocr_worker.py` whose `read(img)` returns the **same `(box, text, conf)` tuples** EasyOCR produced. → downstream code untouched.
2. Swap `_ensure_reader`/`_read` to use the wrapper; keep `_resolve_gpu`, queue, preprocess, ordering, verify, anti-dup, stats, and the **public `OCRWorker` interface identical** (`submit/start/stop/get_stats/set_enabled/set_min_confidence`, `on_result`, `normalize_vin/verify_vin/is_valid_vin`).
3. Replace EasyOCR-only config knobs with PaddleOCR knobs; keep generic ones (`use_gpu`, CLAHE, upscale, sharpen, confidence thresholds, queue, throttle, duplicate window).
4. Swap deps; update docs.
5. Provide a **benchmark harness** for real before/after numbers on your own crops.

**No interface broke.** `pipeline.py` and `server.py` required **zero changes** — confirmation that the architecture was preserved.

---

## 4. Code modifications

**`ocr_worker.py`**
- New `_PaddleEngine`: version-robust init (handles PaddleOCR 2.x `use_angle_cls/use_gpu/show_log` **and** 3.x `use_textline_orientation/device`), grayscale→BGR, `ocr(img, det=…, cls=…)` with a fallback signature, and a defensive result parser that handles **det+rec** (`[box, (text, score)]`), **rec-only** (`(text, score)`), and empty/None results.
- `_read()` now delegates to the engine (one line). `_process`, `_preprocess`, ordering, verify, anti-dup, stats: **unchanged**.
- VIN charset (A–Z 0–9, no I/O/Q) is enforced exactly as before via `normalize_vin` + the `^[A-HJ-NPR-Z0-9]{17}$` regex (PaddleOCR has no `allowlist` arg, so the post-filter does this — same net effect).

**`config.py` `OCRConfig`** — removed EasyOCR-only params; added: `lang`, `paddle_det` (det+rec vs **rec-only**), `use_angle_cls=False`, `drop_score=0.30`, `rec_batch_num=6`, `det_limit_side_len=960`. Kept all generic knobs.

**`requirements.txt`** — removed `easyocr`; added `paddleocr==2.7.3` + `paddlepaddle==2.6.1` (CPU) with a GPU install block (`paddlepaddle-gpu`) and offline-model note. (This also removes the `opencv-python-headless` duplicate that EasyOCR used to pull in.)

**Docs** — README/TRAINING_GUIDE updated EasyOCR → PaddleOCR.

**New:** `tools/benchmark_ocr.py` (see §6).

**Verification performed:** unit-tested the Paddle parser for det+rec (out-of-order fragments reorder to `1HGCM82633A004352`), rec-only, empty/None, and low-confidence drop; confirmed the new engine + worker methods compile.

---

## 5. Performance comparison (EasyOCR vs PaddleOCR)

> **These are typical/expected ranges from general industrial-OCR experience, not measurements on your data.** Run `tools/benchmark_ocr.py` (§6) on your real crops to get authoritative numbers for your hardware. Treat the table as a hypothesis to validate.

Per **single tight VIN crop** (~`300×64`), preprocessed identically:

| Metric | EasyOCR 1.7.2 | PaddleOCR 2.7.3 (det+rec) | PaddleOCR (rec-only) | Notes |
|---|---|---|---|---|
| Latency p50, **CPU** | ~120–300 ms | ~70–180 ms | ~25–70 ms | PP-OCR recognizer is lighter; rec-only skips detection |
| Latency p50, **GPU (NVIDIA)** | ~20–45 ms | ~10–25 ms | ~5–15 ms | Your Ubuntu box; biggest win is here |
| Peak RAM (engine) | ~700 MB–1.2 GB | ~400–800 MB | ~300–600 MB | Paddle models are smaller |
| Accuracy (clean etched, 17 ch) | baseline | **+2–6 pts** | **+1–5 pts** | PP-OCRv4 rec generally stronger on industrial chars |
| Accuracy (low-contrast/reflective) | baseline | **+3–8 pts** | similar | depends heavily on preprocessing (§7) |
| First-run model download | ~64–90 MB | ~10–20 MB (det+rec+cls) | smaller | offline placement recommended (§7) |

**Why PaddleOCR tends to win here:** PP-OCRv4's recognition head is optimized for dense alphanumeric strings and small text, and the models are smaller (lower latency/memory). For YOLO-cropped **single-line** VINs, **rec-only mode** (`paddle_det=False`) is usually the sweet spot — fastest and avoids the detector occasionally splitting a sparse dot-peen string.

**How to make the comparison real:**
```bash
# CPU, accuracy from filenames (AI_CAM saves crops as <VIN>_<ts>.jpg)
python tools/benchmark_ocr.py --crops data/crops --runs 5 --gt-from-filename
# GPU, PaddleOCR in rec-only mode
python tools/benchmark_ocr.py --crops data/crops --runs 5 --gpu --gt-from-filename --paddle-rec-only
```
It prints p50/p95/mean latency, peak memory, and accuracy for both engines on the **same** preprocessed crops.

---

## 6. Preprocessing review — practical gains for engraved/etched metal

Your current chain (`gray → upscale-to-96px → CLAHE(clip 3.0, 8×8) → light unsharp`) is already well-chosen and balanced. Recommended **practical** additions (each cheap, toggle in config; adopt only what the benchmark shows helps):

1. **CLAHE — keep.** Already present; it's the single highest-value step for low-contrast etched metal. If characters look washed out, nudge `clahe_clip` 2.0→4.0; if noisy, lower it. Don't go above ~5.
2. **Reflection/glare reduction (high value on shiny metal).** A **morphological black-hat** (`cv2.morphologyEx(gray, MORPH_BLACKHAT, kernel)`) pulls dark engraved strokes out from a bright specular background; or divide-by-background (`gray / (blur+1) * 255`) to flatten uneven lighting. This is the most useful *new* step for reflective plates. Kernel ~ (15×15).
3. **Adaptive threshold — only as an alternative branch, not always-on.** `adaptiveThreshold(..., GAUSSIAN, blockSize≈25, C≈10)` can help very low-contrast cases but **can destroy dot-peen dots** if too aggressive. Keep it optional and OFF by default; let the recognizer see grayscale first.
4. **Morphological close (tiny).** A 1–2 px `MORPH_CLOSE` can connect dot-peen dots into legible strokes. Use a small elliptical kernel; over-doing it merges characters.
5. **Upscale — keep**, but cap it. Going beyond ~2× of native rarely helps and costs latency. INTER_CUBIC is fine.

**Avoid (academic, low ROI here):** heavy denoisers (NLM), deskew/perspective unwarp (YOLO already gives axis-aligned crops), super-resolution networks, multi-scale ensembles. They add latency/complexity without reliable gains on tight single-line crops.

**Suggested experiment order (validate with the benchmark):** black-hat/illumination-flatten → (re)tune CLAHE clip → small morphological close → only then consider adaptive threshold. Change one knob at a time and measure.

---

## 7. Final recommendations

1. **Install on the Ubuntu+NVIDIA box:** `paddlepaddle-gpu` matching your CUDA (see `requirements.txt` footer), keep `OCR.use_gpu=True` (auto-falls back to CPU). Biggest latency win is GPU.
2. **Try `paddle_det=False` (rec-only) first.** For tight single-line VIN crops it's typically the fastest and most robust. Flip to `True` only if you see multi-line or padded crops. Benchmark both with `--paddle-rec-only`.
3. **Pre-place models for the factory (offline/SSL-restricted).** On a machine with internet, run PaddleOCR once to populate `~/.paddleocr/whl/{det,rec,cls}/…`, then copy that folder to the server (mirrors the EasyOCR SSL issue seen in your logs). This avoids first-run download failures.
4. **Keep the single-shot trigger at 0.95** — it already stops triple-OCR, so OCR cost per vehicle is one inference. Combined with rec-only on GPU, per-plate latency should be a few ms.
5. **Adopt preprocessing additively** (start with black-hat/illumination flatten for reflective plates), measuring accuracy on `data/crops` after each change.
6. **Rollback safety:** EasyOCR was removed cleanly, but the engine is isolated in `_PaddleEngine`; if ever needed, an EasyOCR wrapper with the same `read()` could be dropped back in without touching the pipeline. The benchmark script keeps an EasyOCR path for ongoing A/B comparison.

**Net:** engine swapped with zero pipeline/API changes, lower expected latency and memory, and an equal-or-better accuracy expectation on industrial alphanumerics — to be confirmed on your data with the included benchmark.
