"""
============================================================
config.py  —  AI_CAM markaziy sozlamalari
============================================================
Barcha o'zgaruvchan parametrlar shu yerda. Ishlab chiqarishda
faqat shu faylni tahrirlash kifoya (IP, portlar, chegaralar).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --- Loyiha papka manzillari ---
BASE_DIR = Path(__file__).resolve().parent.parent      # .../AI_CAM
DATA_DIR = BASE_DIR / "data"
CROPS_DIR = DATA_DIR / "crops"                          # kesilgan VIN rasmlari
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"
DB_PATH = DATA_DIR / "ai_cam.db"
# Avtomatik dataset yig'ish papkasi (Data Loop)
DATASET_DIR = BASE_DIR / "dataset" / "collected_raw"
# O'qitilgan YOLOv8n model (training natijasi). BASE_DIR ga nisbatan absolyut
# yo'l — cwd qanday bo'lishidan qat'i nazar topiladi.
TRAINED_MODEL_PATH = BASE_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"

# Papkalar mavjudligini ta'minlash
for _d in (DATA_DIR, CROPS_DIR, LOGS_DIR, MODELS_DIR, DATASET_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass
class CameraConfig:
    """SICK LECTOR652 ulanish sozlamalari (lector652 loyihasidan)."""
    # Eslatma: IP endi hardcode QILINMAYDI. Standart qiymat faqat
    # UI dagi input maydoni uchun (foydalanuvchi /dashboard da o'zgartiradi).
    ip: str = "192.168.1.10"
    # CoLa-A (ASCII) control porti. Lector 65x: 2111=CoLa-A, 2112=CoLa-B
    cola_port: int = 2111
    # BLOB (rasm) streaming porti — foydalanuvchi talabi bo'yicha 2113 ga
    # qat'iy bog'langan (autodetect O'CHIRILGAN).
    blob_port: int = 2113
    # False -> GetBlobClientConfig bilan port almashtirilmaydi, doim 2113.
    blob_autodetect: bool = False
    # CoLa-A CheckPassword (ishlaydigan lector652 koddan). Ba'zi kameralar
    # talab qilmaydi — xato bo'lsa ulanish baribir davom etadi.
    password: str = "A89A6E74"

    recv_timeout: float = 3.0          # socket recv timeout (s)
    reconnect_delay: float = 2.0       # uzilganda qayta ulanish kechikishi (s)
    # Trigger pulse: gateon -> kutish -> gateoff
    trigger_pulse_off_delay: float = 0.15   # 150 ms (Wireshark dagidek)
    capture_timeout: float = 5.0       # rasm kelishini kutish (s)
    # Kadr xom (original) holatda qayta ishlanadi — orientatsiya allaqachon to'g'ri.


@dataclass
class DetectionConfig:
    """YOLOv8n plastinka aniqlash sozlamalari."""
    # O'qitilgan model: runs/detect/vin_plate_model/weights/best.pt
    model_path: str = str(TRAINED_MODEL_PATH)
    # Past chegara: dataset yig'ish uchun nomzod box'larni ham ko'rsatadi (>0.40).
    conf_threshold: float = 0.40       # YOLO bazaviy aniqlash chegarasi (dataset collect)
    iou_threshold: float = 0.45
    # OCR FAQAT shu ishonchdan yuqorida ishga tushadi — talab bo'yicha 0.95 (95%)
    ocr_trigger_conf: float = 0.95
    # "auto" -> NVIDIA GPU bo'lsa cuda:0, aks holda cpu. Yoki "cuda:0"/"cpu".
    device: str = "auto"
    crop_padding: int = 8              # ROI atrofiga qo'shimcha piksel

    # --- Single-shot trigger (debounce / state-lock) ---
    # Bitta plastinka kadrda bir necha kadr turishi mumkin — OCR FAQAT BIR MARTA
    # ishga tushadi. Plastinka kadrdan ketib (shu qadar kadr ko'rinmay) qaytsa,
    # yangi avtomobil sifatida qayta trigger bo'ladi.
    ocr_rearm_absent_frames: int = 3   # plastinka shu qadar kadr ko'rinmasa qayta yoqiladi
    ocr_cooldown_sec: float = 2.0      # ketma-ket triggerlar orasidagi minimal vaqt (xavfsizlik)

    # --- Multi-frame fusion + gating ---
    # OCR faqat plastinka shu qadar YUQORI-ISHONCH (>=0.95) kadrida ko'ringach
    # ishga tushadi (2-frame tasdiqlash). Har kadrda crop sifati baholanib
    # buferga yig'iladi; ENG YAXSHI fusion_k crop OCR ga (ovoz berish) yuboriladi.
    ocr_confirm_frames: int = 2        # OCR dan oldin kerakli yuqori-ishonch kadrlar
    ocr_crop_margin_frac: float = 0.15  # ROI ni shu nisbatda kengaytirish (belgilar kesilmasin)
    min_crop_quality: float = 0.45     # eng yaxshi crop sifati shundan past bo'lsa — kutamiz
    event_buffer_max: int = 10         # bitta hodisa uchun saqlanadigan eng yaxshi croplar soni
    fusion_k: int = 3                  # OCR ovoz berishga yuboriladigan top croplar soni

    # --- Avtomatik dataset yig'ish (Data Loop) ---
    collect_enabled: bool = True
    collect_conf: float = 0.40         # shu conf dan yuqori aniqlovlar saqlanadi
    collect_min_interval_sec: float = 1.0   # throttle: ~1 rasm/sekund (FPS himoyasi)
    collect_class_id: int = 0          # YOLO yorliq sinfi (0 = vin_plate)


@dataclass
class OCRConfig:
    """PaddleOCR sozlamalari (engraved/etched metal VIN uchun, aniqlik + tezlik)."""
    lang: str = "en"
    # True -> PaddleOCR GPU (NVIDIA/CUDA) bilan ishlaydi. Ubuntu+NVIDIA serverda
    # tavsiya etiladi. CUDA topilmasa avtomatik CPU ga qaytadi (xavfsiz).
    use_gpu: bool = True

    # --- Preprocess (aniqlik uchun) ---
    apply_clahe: bool = True           # yengil CLAHE (kontrast)
    clahe_clip: float = 3.0
    clahe_grid: int = 8
    upscale_height: int = 96           # crop ni shu balandlikka kattalashtirish
                                       # (0 = o'chiq). Kichik etched belgilar uchun muhim.
    sharpen: bool = True               # yengil unsharp mask (chekka aniqligi)

    # --- PaddleOCR parametrlari ---
    # paddle_det=True -> detektor+recognizer (EasyOCR kabi, ko'p qatorli/paddingli
    #   croplar uchun xavfsiz). False -> faqat recognizer (rec-only): YOLO bergan
    #   bitta qatorli tor crop uchun TEZROQ va ko'pincha aniqroq.
    paddle_det: bool = True
    use_angle_cls: bool = False        # burchak klassifikatori (metall matn tik -> o'chiq, tez)
    drop_score: float = 0.30           # PaddleOCR shu balldan past natijalarni tashlaydi
    rec_batch_num: int = 6             # recognizer batch (tezlik)
    det_limit_side_len: int = 960      # detektor maksimal tomon uzunligi (det rejimida)

    # --- Ishonch chegaralari ---
    min_char_confidence: float = 0.30  # har bir bo'lak (fragment) uchun minimal ishonch
    min_confidence: float = 0.45       # umumiy (o'rtacha) ishonch chegarasi

    # --- Denoise (engraved metal uchun) ---
    bilateral: bool = True             # chekkalarni saqlab shovqinni kamaytirish
    bilateral_d: int = 5
    bilateral_sigma: float = 50.0

    # --- Retry + fusion (fail handling) ---
    retry_enabled: bool = True         # OCR muvaffaqiyatsiz bo'lsa variantlarni sinash
    retry_rotations: tuple = (-7.0, 7.0)   # qayta urinishda buriladigan gradlar (±)
    max_ocr_attempts: int = 5          # bitta hodisa uchun jami OCR chaqiruvlari chegarasi

    # --- Tezlik / lag nazorati ---
    queue_maxsize: int = 50            # OCR navbati hajmi (orqaga bosim oldini olish)
    min_interval_sec: float = 0.0      # ketma-ket OCR yugurishlari orasidagi minimal
                                       # vaqt (0 = o'chiq; navbat allaqachon cheklaydi)

    # Anti-duplicate: bir xil VIN shu oraliqda qayta yozilmaydi
    duplicate_window_sec: float = 90.0


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    mjpeg_fps: int = 25                # live stream maksimal FPS
    jpeg_quality: int = 80             # MJPEG kodlash sifati


@dataclass
class AuthConfig:
    """Yengil frontend autentifikatsiya (zavod ichki tarmog'i uchun)."""
    enabled: bool = True
    username: str = "admin"
    password: str = "vin_factory2026"   # zavod standart paroli — kerak bo'lsa o'zgartiring
    cookie_name: str = "ai_cam_session"
    session_ttl_sec: int = 12 * 3600    # 12 soat — keyin qayta login


# --- Global yagona nusxalar ---
CAMERA = CameraConfig()
DETECTION = DetectionConfig()
OCR = OCRConfig()
SERVER = ServerConfig()
AUTH = AuthConfig()

# VIN format qoidasi: 17 belgi, I/O/Q harflari yo'q
VIN_LENGTH = 17
VIN_INVALID_CHARS = set("IOQ")
VIN_ALLOWED = set("ABCDEFGHJKLMNPRSTUVWXYZ0123456789")
