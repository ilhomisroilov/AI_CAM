# YOLOv8n Training Guide — VIN Plate Detection

A practical, step-by-step tutorial to train `yolov8n` to detect the **metal VIN plate** on car bodies coming down your conveyor. The goal is a single-class detector (`vin_plate`) whose crops feed the PaddleOCR stage.

> You only need YOLO to **find and box the plate** — not to read it. Keep the task simple (one class) and the model stays fast and accurate on the Mini PC.

---

## 1. Collect images

The fastest way to grow your dataset is the built-in **auto-collection** feature: when processing is running, every frame with a YOLO detection above `collect_conf` (0.40) is saved automatically to `dataset/collected_raw/` as a `.jpg` plus a matching YOLO-format `.txt` label (see section 1b). You can also capture deliberately:

- Aim for **300–800 images** to start; 1500+ for production robustness.
- Capture **real variety**: different lighting, slightly different body positions, day/night shifts, dirty vs. clean plates, glare, motion blur.
- Save them at the camera's native resolution. Your Lector652 effective image size was `800×440` in the capture — keep that aspect.
- Frames are stored and processed in their **raw, original orientation** — no rotation is applied anywhere in the pipeline. Annotate images exactly as the camera produces them.

Put manually collected raw images in `dataset/images_raw/`; auto-collected ones land in `dataset/collected_raw/`.

### 1b. Auto-collection (auto-labeling)

While the camera is connected and **Start Processing** is active, the background stream thread saves, at most once per second (throttled so the live stream FPS is unaffected):

- `dataset/collected_raw/vin_capture_YYYYMMDD_HHMMSS.jpg` — the original, uncropped frame.
- `dataset/collected_raw/vin_capture_YYYYMMDD_HHMMSS.txt` — one line per detected box in YOLO format: `0 cx cy w h` (normalized 0–1, class `0` = `vin_plate`).

These are **pre-labels** produced by the current model — review/correct them in Roboflow or labelImg before retraining. This is the data-loop that lets you bootstrap from your initial 10 images.

---

## 2. Annotate (label the bounding boxes)

Use **[Roboflow](https://roboflow.com)** (web, easiest) or **[labelImg](https://github.com/HumanSignal/labelImg)** (offline).

Rules:

- One class only: `vin_plate`.
- Draw the box **tight around the etched VIN region** — include a little surrounding metal so OCR has context, but don't include unrelated stampings.
- Be consistent. If sometimes the plate is partially occluded, still box the visible part.
- Export in **YOLO format** (one `.txt` per image: `class cx cy w h`, normalized 0–1).

### labelImg quick start

```bash
pip install labelImg
labelImg dataset/images_raw    # set format to "YOLO" in the left toolbar
```

---

## 3. Organize the dataset

Split roughly **80% train / 20% val**:

```
dataset/
├── images/
│   ├── train/   img001.jpg ...
│   └── val/     img120.jpg ...
└── labels/
    ├── train/   img001.txt ...
    └── val/     img120.txt ...
```

Create **`dataset/data.yaml`**:

```yaml
path: ./dataset
train: images/train
val: images/val
nc: 1
names: ["vin_plate"]
```

---

## 4. Train

Install and train (GPU strongly recommended for training; CPU works but slow):

```bash
pip install ultralytics

yolo detect train \
  model=yolov8n.pt \
  data=dataset/data.yaml \
  epochs=120 \
  imgsz=640 \
  batch=16 \
  patience=30 \
  name=vin_plate_model
```

Tips for a small metal-plate dataset:

- `model=yolov8n.pt` starts from COCO weights (transfer learning) — far better than scratch.
- If you see overfitting, lower `epochs` or add more data; YOLO's built-in augmentation (mosaic, HSV, flips) already helps a lot.
- **Disable vertical flips** if plate orientation always matters — they're off by default for detection, which is correct here.
- Watch `box_loss` and `mAP50` in the console / `runs/detect/vin_plate_model/`.

---

## 5. Validate & test

```bash
# Metrics on the val set
yolo detect val model=runs/detect/vin_plate_model/weights/best.pt data=dataset/data.yaml

# Try it on a few new images
yolo detect predict model=runs/detect/vin_plate_model/weights/best.pt source=some_new_frame.jpg
```

Target **mAP50 ≳ 0.95** for a single, well-lit class like this. If lower, the usual fix is *more varied data*, not more epochs.

---

## 6. Deploy into AI_CAM

No copying needed — the app reads the trained weights **directly** from the training output path. `backend/config.py` is wired to:

```python
TRAINED_MODEL_PATH = BASE_DIR / "runs" / "detect" / "vin_plate_model" / "weights" / "best.pt"
DETECTION.model_path = str(TRAINED_MODEL_PATH)   # runs/detect/vin_plate_model/weights/best.pt
DETECTION.conf_threshold = 0.40      # low base threshold (also feeds dataset auto-collection)
DETECTION.ocr_trigger_conf = 0.70    # only crop+OCR above this
DETECTION.device = "cpu"             # or "cuda:0" if GPU on the Mini PC
```

So if your `yolo detect train` run used `name=vin_plate_model`, the weights at
`runs/detect/vin_plate_model/weights/best.pt` are loaded automatically. (If you used a
different `name=`, either rename the run folder or update `TRAINED_MODEL_PATH`.)

Restart `python run.py` — the dashboard will show **YOLO model: Ready** and draw teal boxes when confidence clears the OCR trigger.

> If `best.pt` isn't found at that path, the app **falls back to stock `yolov8n.pt`** so the UI still runs — but it won't reliably find VIN plates until the trained weights are in place.

---

## 7. Improving accuracy toward the 99% OCR target

YOLO accuracy and OCR accuracy are separate. For the **99% read target** on dot-peen VINs:

1. **Lighting is everything.** Your own notes concluded **raking light (10–30° grazing angle)** beats IR/UV for etched/dot-peen metal — it casts shadows in the engraving. Install that first; it does more than any software tweak.
2. **Tighten the crop** (`crop_padding`) so OCR sees mostly characters.
3. Keep preprocessing **light** (CLAHE only) — heavy thresholding destroyed dot-peen dots in past tests.
4. Collect misreads from the `crops/` folder and **fine-tune** PaddleOCR on them if needed.
