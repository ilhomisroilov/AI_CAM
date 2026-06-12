"""
============================================================
vin_postprocess.py  —  VIN-aware OCR post-processing (candidate ranking)
============================================================
Maqsad: OCR natijasini O'ZGARTIRMASDAN, eng EHTIMOLLI VIN nomzodini
tanlash. Tizim belgini O'YLAB TOPMAYDI / MAJBURAN ALMASHTIRMAYDI — u faqat
OCR KO'RGAN belgilar (va ularning vizual chalkashliklari) orasidan VIN
strukturasiga ko'ra SARALAYDI.

Kafolat ("never invent"):
  Har pozitsiya uchun nomzodlar = { OCR belgisi } ∪ { OCR belgisining
  ma'lum chalkashlik variantlari }. Kutilgan strukturaviy belgi FAQAT shu
  to'plamda bo'lsa tanlanishi mumkin. Aks holda OCR belgisi saqlanadi
  (jazo bilan belgilanadi). Xom OCR natijasi HAR DOIM saqlanadi.

Ball:  char_score = w_visual*visual + (struct_bonus | -struct_penalty)
       FinalScore = 0.5*mean(chosen_visual) + 0.5*compliance_fraction
Global ketma-ketlik bali = pozitsiyalar bo'yicha additiv (strukturaviy
qoidalar pozitsiyaga bog'liq -> per-position argmax = global optimum).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import vin_rules

# --- OCR vizual chalkashliklar (ikki tomonlama) ---
# 1↔I, 1↔T, S↔5, O↔0, B↔8, G↔6, A↔4, E↔F
CONFUSIONS: Dict[str, Tuple[str, ...]] = {
    "1": ("I", "T"),
    "I": ("1",),
    "T": ("1",),
    "S": ("5",),
    "5": ("S",),
    "O": ("0",),
    "0": ("O",),
    "B": ("8",),
    "8": ("B",),
    "G": ("6",),
    "6": ("G",),
    "A": ("4",),
    "4": ("A",),
    "E": ("F",),
    "F": ("E",),
}


@dataclass
class PositionDecision:
    pos: int                 # 1-asosli pozitsiya
    raw_char: str
    chosen_char: str
    visual: float
    allowed: bool            # tanlangan belgi strukturaga mosmi
    changed: bool            # OCR belgisidan farq qiladimi (re-rank natijasi)


@dataclass
class VinResult:
    raw_vin: str             # xom OCR (audit uchun saqlanadi)
    validated_vin: str       # strukturaga ko'ra saralangan natija
    model: Optional[str]
    final_score: float       # 0..1
    compliance: float        # qoidaga mos pozitsiyalar ulushi 0..1
    fully_compliant: bool
    decisions: List[PositionDecision] = field(default_factory=list)
    note: str = ""


def _candidates(ch: str, visual: float, confusion_prior: float) -> Dict[str, float]:
    """OCR belgisi + uning chalkashlik variantlari (past vizual ball bilan)."""
    cands: Dict[str, float] = {ch: visual}
    for alt in CONFUSIONS.get(ch, ()):  # faqat ma'lum chalkashliklar
        v = visual * confusion_prior
        if v > cands.get(alt, -1.0):
            cands[alt] = v
    return cands


def postprocess(
    raw_vin: str,
    model: Optional[str],
    char_confidences: Optional[List[float]] = None,
    overall_conf: float = 0.0,
    *,
    confusion_prior: float = 0.6,
    w_visual: float = 1.0,
    struct_bonus: float = 0.5,
    struct_penalty: float = 0.5,
) -> VinResult:
    """
    Xom OCR natijasini model qoidalariga ko'ra saralaydi.

    char_confidences: agar mavjud bo'lsa, har belgi uchun vizual ishonch
      (PaddleOCR yuqori-darajali API buni bermaydi -> overall_conf ishlatiladi).
    """
    raw = "".join(c for c in raw_vin.upper() if c.isalnum())

    # Model qo'llab-quvvatlanmasa yoki uzunlik 17 emas -> validatsiya qilmaymiz
    if not vin_rules.is_supported_model(model) or len(raw) != vin_rules.VIN_LENGTH:
        return VinResult(
            raw_vin=raw, validated_vin=raw, model=model,
            final_score=float(overall_conf), compliance=0.0, fully_compliant=False,
            note=("noma'lum model" if not vin_rules.is_supported_model(model)
                  else f"uzunlik {len(raw)} != 17 — validatsiya o'tkazilmadi"),
        )

    confs = char_confidences if (char_confidences and len(char_confidences) == len(raw)) \
        else [overall_conf] * len(raw)

    chosen: List[str] = []
    chosen_visual: List[float] = []
    decisions: List[PositionDecision] = []

    for i, ch in enumerate(raw):
        cands = _candidates(ch, float(confs[i]), confusion_prior)
        best_char, best_score, best_vis = ch, -1e9, float(confs[i])
        for cand, vis in cands.items():
            struct = struct_bonus if vin_rules.is_allowed(model, i, cand) else -struct_penalty
            score = w_visual * vis + struct
            if score > best_score:
                best_char, best_score, best_vis = cand, score, vis
        chosen.append(best_char)
        chosen_visual.append(best_vis)
        decisions.append(PositionDecision(
            pos=i + 1, raw_char=ch, chosen_char=best_char, visual=best_vis,
            allowed=vin_rules.is_allowed(model, i, best_char),
            changed=(best_char != ch),
        ))

    validated = "".join(chosen)
    fully, _flags = vin_rules.validate(model, validated)
    compliance = vin_rules.compliance_fraction(model, validated)
    mean_vis = sum(chosen_visual) / len(chosen_visual) if chosen_visual else 0.0
    final = 0.5 * mean_vis + 0.5 * compliance

    return VinResult(
        raw_vin=raw, validated_vin=validated, model=model,
        final_score=float(final), compliance=float(compliance), fully_compliant=fully,
        decisions=decisions,
    )
