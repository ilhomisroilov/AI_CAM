"""
============================================================
vin_rules.py  —  Model-aware VIN strukturasi qoidalari
============================================================
QY va BL7M plastinka modellari uchun 17-belgili VIN tuzilishi.
Bu modul FAQAT bilim bazasi (qoidalar) — u belgini O'ZGARTIRMAYDI yoki
O'YLAB TOPMAYDI. Post-processor (vin_postprocess.py) bu qoidalardan
NOMZODLARNI BAHOLASH/SARALASH uchun foydalanadi.

Pozitsiyalar (1-asosli):
  1-3   : N S T   (har ikkala model)
  4     : QY=F,  BL7M=H
  5     : QY={A,B,C},  BL7M={A,B,C,D}    (transmissiya)
  6     : QY=8,  BL7M=4
  7     : QY=1,  BL7M=1
  8     : QY=4,  BL7M={G,A}
  9     : {A,E} (har ikkala)
  10    : model yili kodi  {S,T,V,W}     (kengaytiriladi)
  11    : J
  12-17 : faqat raqam 0-9
"""
from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Tuple

VIN_LENGTH = 17
MODELS = ("QY", "BL7M")
DIGITS = frozenset("0123456789")

# Model yili kodlari — KENGAYTIRILADIGAN (yangi yillar shu yerga qo'shiladi)
YEAR_CODES: Dict[str, int] = {
    "S": 2025,
    "T": 2026,
    "V": 2027,
    "W": 2028,
}
YEAR_CHARS = frozenset(YEAR_CODES.keys())


def _f(chars: str) -> FrozenSet[str]:
    return frozenset(chars)


# Har model uchun 17 ta pozitsiyaning ruxsat etilgan belgilar to'plami.
# (1-asosli pozitsiya -> RULES[model][pos-1])
RULES: Dict[str, List[FrozenSet[str]]] = {
    "QY": [
        _f("N"), _f("S"), _f("T"),          # 1-3
        _f("F"),                            # 4
        _f("ABC"),                          # 5
        _f("8"),                            # 6
        _f("1"),                            # 7
        _f("4"),                            # 8
        _f("AE"),                           # 9
        YEAR_CHARS,                         # 10
        _f("J"),                            # 11
        DIGITS, DIGITS, DIGITS, DIGITS, DIGITS, DIGITS,  # 12-17
    ],
    "BL7M": [
        _f("N"), _f("S"), _f("T"),          # 1-3
        _f("H"),                            # 4
        _f("ABCD"),                         # 5
        _f("4"),                            # 6
        _f("1"),                            # 7
        _f("GA"),                           # 8
        _f("AE"),                           # 9
        YEAR_CHARS,                         # 10
        _f("J"),                            # 11
        DIGITS, DIGITS, DIGITS, DIGITS, DIGITS, DIGITS,  # 12-17
    ],
}

# 12-17 pozitsiyalar — raqamli (harf qattiq jazolanadi). Tezkor tekshiruv uchun.
NUMERIC_POSITIONS = frozenset(range(11, 17))   # 0-asosli 11..16 = pozitsiya 12..17
# Eng qattiq cheklangan (heavy-penalty) pozitsiyalar: 1-3 (NST) va 12-17 (raqam)
HEAVY_POSITIONS = frozenset([0, 1, 2]) | NUMERIC_POSITIONS


def is_supported_model(model: Optional[str]) -> bool:
    return model in RULES


def allowed_at(model: str, pos0: int) -> FrozenSet[str]:
    """0-asosli pozitsiya uchun ruxsat etilgan belgilar to'plami."""
    return RULES[model][pos0]


def is_allowed(model: str, pos0: int, ch: str) -> bool:
    return ch in RULES[model][pos0]


def year_from_code(code: str) -> Optional[int]:
    """Pozitsiya-10 kodidan model yilini qaytaradi (yoki None)."""
    return YEAR_CODES.get(code)


def register_year(code: str, year: int) -> None:
    """Yangi model yili kodini qo'shadi (kengaytirish uchun)."""
    YEAR_CODES[code] = year
    global YEAR_CHARS
    YEAR_CHARS = frozenset(YEAR_CODES.keys())
    for m in RULES:
        RULES[m][9] = YEAR_CHARS      # pozitsiya 10


def validate(model: str, vin: str) -> Tuple[bool, List[bool]]:
    """
    VIN ni model qoidalariga solishtiradi.
    Qaytadi: (toliq_mos, har_pozitsiya_mosligi[17]).
    """
    if model not in RULES or len(vin) != VIN_LENGTH:
        return False, []
    flags = [is_allowed(model, i, ch) for i, ch in enumerate(vin)]
    return all(flags), flags


def compliance_fraction(model: str, vin: str) -> float:
    """Qoidaga mos pozitsiyalar ulushi (0..1)."""
    ok, flags = validate(model, vin)
    if not flags:
        return 0.0
    return sum(flags) / float(len(flags))
