"""
Unit tests — VIN rules + VIN-aware post-processing (QY / BL7M).
Run:  python -m pytest tests/test_vin_rules.py -q
   or: python tests/test_vin_rules.py   (asserts, no pytest needed)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.ai import vin_rules as R
from backend.ai.vin_postprocess import postprocess, _candidates, CONFUSIONS

QY_VALID = "NSTFA814ATJ123456"      # spec example (QY)
BL7M_VALID = "NSTHB41GATJ987654"    # valid under stated BL7M rules


# ---------------- structure rules ----------------
def test_qy_valid_vin_compliant():
    ok, flags = R.validate("QY", QY_VALID)
    assert ok and all(flags)

def test_bl7m_valid_vin_compliant():
    ok, _ = R.validate("BL7M", BL7M_VALID)
    assert ok

def test_positions_1_3_are_NST_both_models():
    for m in ("QY", "BL7M"):
        assert R.is_allowed(m, 0, "N") and R.is_allowed(m, 1, "S") and R.is_allowed(m, 2, "T")
        assert not R.is_allowed(m, 0, "M")   # N only
        assert not R.is_allowed(m, 1, "5")   # S only
        assert not R.is_allowed(m, 2, "1")   # T only

def test_position4_model_specific():
    assert R.is_allowed("QY", 3, "F") and not R.is_allowed("QY", 3, "H")
    assert R.is_allowed("BL7M", 3, "H") and not R.is_allowed("BL7M", 3, "F")

def test_position5_transmission():
    assert R.allowed_at("QY", 4) == frozenset("ABC")
    assert R.allowed_at("BL7M", 4) == frozenset("ABCD")
    assert not R.is_allowed("QY", 4, "D")        # D only for BL7M
    assert R.is_allowed("BL7M", 4, "D")

def test_positions_6_7_8():
    assert R.is_allowed("QY", 5, "8") and not R.is_allowed("QY", 5, "4")     # pos6
    assert R.is_allowed("BL7M", 5, "4") and not R.is_allowed("BL7M", 5, "8")
    assert R.is_allowed("QY", 6, "1") and R.is_allowed("BL7M", 6, "1")        # pos7
    assert R.is_allowed("QY", 7, "4") and not R.is_allowed("QY", 7, "A")      # pos8 QY=4
    assert R.allowed_at("BL7M", 7) == frozenset("GA")                         # pos8 BL7M={G,A}

def test_position9_and_11():
    for m in ("QY", "BL7M"):
        assert R.allowed_at(m, 8) == frozenset("AE")     # pos9
        assert R.is_allowed(m, 10, "J") and not R.is_allowed(m, 10, "1")  # pos11=J

def test_positions_12_17_digits_only():
    for m in ("QY", "BL7M"):
        for p in range(11, 17):
            assert R.is_allowed(m, p, "0") and R.is_allowed(m, p, "9")
            assert not R.is_allowed(m, p, "A")   # letters not allowed

def test_year_mapping_and_extension():
    assert R.year_from_code("S") == 2025 and R.year_from_code("T") == 2026
    assert R.year_from_code("V") == 2027 and R.year_from_code("W") == 2028
    assert R.year_from_code("Z") is None
    R.register_year("X", 2029)
    assert R.year_from_code("X") == 2029 and R.is_allowed("QY", 9, "X")


# ---------------- confusion map ----------------
def test_confusion_pairs_present():
    for a, b in [("1", "I"), ("1", "T"), ("S", "5"), ("O", "0"),
                 ("B", "8"), ("G", "6"), ("A", "4"), ("E", "F")]:
        assert b in CONFUSIONS.get(a, ()) or a in CONFUSIONS.get(b, ())


# ---------------- post-processing (ranking, never invent) ----------------
def test_ns1_corrected_to_nst_via_confusion_and_rule():
    # spec example: NS1FA814ATJ123456 -> NSTFA814ATJ123456
    r = postprocess("NS1FA814ATJ123456", "QY", overall_conf=0.9)
    assert r.validated_vin == QY_VALID
    assert r.fully_compliant and r.final_score > 0.7
    assert r.raw_vin == "NS1FA814ATJ123456"     # raw preserved for audit

def test_never_invent_keeps_unconfusable_char():
    # 'M' at pos1 has NO confusion to 'N' -> must stay 'M' (faithful), penalized
    r = postprocess("MSTFA814ATJ123456", "QY", overall_conf=0.9)
    assert r.validated_vin[0] == "M"
    assert not r.fully_compliant
    assert r.decisions[0].changed is False and r.decisions[0].allowed is False

def test_digits_positions_letter_with_confusion_corrected():
    # pos12 'B' -> '8' (B<->8 confusion + digit rule)
    raw = "NSTFA814ATJ" + "B23456"
    r = postprocess(raw, "QY", overall_conf=0.9)
    assert r.validated_vin[11] == "8" and r.fully_compliant

def test_digits_positions_letter_without_digit_confusion_kept():
    # pos12 'J' has no digit confusion -> stays 'J' (faithful), not compliant
    raw = "NSTFA814ATJ" + "J23456"
    r = postprocess(raw, "QY", overall_conf=0.9)
    assert r.validated_vin[11] == "J" and not r.fully_compliant

def test_S5_confusion_at_pos2():
    r = postprocess("N5TFA814ATJ123456", "QY", overall_conf=0.9)
    assert r.validated_vin[1] == "S" and r.fully_compliant

def test_model_consistency_pos4():
    # raw F at pos4: compliant for QY, NOT for BL7M (needs H, F->E only, no invent)
    assert postprocess(QY_VALID, "QY", overall_conf=0.9).fully_compliant
    r_bl = postprocess(QY_VALID, "BL7M", overall_conf=0.9)
    assert r_bl.validated_vin[3] in ("F", "E") and not r_bl.fully_compliant

def test_bl7m_pos8_4_reranked_to_A():
    # spec DB example NSTHB414ESJ987654: pos8 '4' -> 'A' under stated BL7M rule
    r = postprocess("NSTHB414ESJ987654", "BL7M", overall_conf=0.9)
    assert r.validated_vin[7] == "A" and r.fully_compliant

def test_unknown_model_or_bad_length_no_validation():
    r = postprocess("NSTFA814ATJ123456", "ZZZ", overall_conf=0.8)
    assert r.validated_vin == r.raw_vin and not r.fully_compliant
    r2 = postprocess("SHORT", "QY", overall_conf=0.8)
    assert r2.validated_vin == "SHORT" and r2.compliance == 0.0


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn(); passed += 1; print(f"  PASS {fn.__name__}")
        except Exception:
            print(f"  FAIL {fn.__name__}"); traceback.print_exc()
    print(f"\n{passed}/{len(fns)} tests passed")
    sys.exit(0 if passed == len(fns) else 1)
