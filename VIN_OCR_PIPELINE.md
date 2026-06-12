# VIN-Aware OCR Pipeline (QY / BL7M)

How AI_CAM turns raw OCR into the **most probable VIN** without ever inventing characters, using the known VIN structure as a *validation, ranking, and error-correction* layer.

> Core guarantee: **the raw OCR result is always preserved and stored. Structure only re-ranks among characters the OCR could plausibly have produced — it never fabricates the expected character.**

---

## 1. Flow

```
YOLO detect (class → model: QY | BL7M)
   → multi-frame event buffer (best crops, §pipeline)
   → PaddleOCR read (per crop + retry variants: deskew, ±rotate)
   → raw VIN string + mean confidence
   → VIN-aware post-process  ─────────────┐
        per position:                     │  vin_postprocess.postprocess()
          candidates = {OCR char} ∪ {visual-confusion alts}
          score = w_visual·visual + (struct_bonus | −struct_penalty)
          choose argmax  (never adds the "expected" char unless OCR-plausible)
   → validated VIN + final_score + per-position audit
   → multi-frame fusion: majority vote on validated VIN (tie → highest score)
   → accept if len==17 and final_score ≥ min_final_score
   → store: detected_vin (validated), raw_vin (audit), model, confidence
```

Modules: `backend/ai/vin_rules.py` (structure knowledge), `backend/ai/vin_postprocess.py` (candidate ranking + scoring), integrated in `backend/ai/ocr_worker.py`, threaded through `backend/pipeline.py`, persisted by `backend/database/db.py`.

---

## 2. VIN structure (per model) — `vin_rules.py`

17 characters. `RULES[model][position-1]` is the allowed-character set.

| Pos | QY | BL7M | Meaning |
|----|----|------|---------|
| 1–3 | `N S T` | `N S T` | fixed prefix |
| 4 | `F` | `H` | model marker |
| 5 | `A B C` | `A B C D` | transmission |
| 6 | `8` | `4` | fixed |
| 7 | `1` | `1` | fixed |
| 8 | `4` | `G A` | fixed / set |
| 9 | `A E` | `A E` | — |
| 10 | `S T V W` | `S T V W` | model year (extensible) |
| 11 | `J` | `J` | fixed |
| 12–17 | `0–9` | `0–9` | numeric only |

**Year mapping** (extensible via `register_year`): `S→2025, T→2026, V→2027, W→2028`.

`validate(model, vin) → (fully_compliant, per_position_flags)`; `compliance_fraction()` returns the share of compliant positions.

---

## 3. Candidate ranking & scoring — `vin_postprocess.py`

**Confusion map** (bidirectional, from the spec): `1↔I, 1↔T, S↔5, O↔0, B↔8, G↔6, A↔4, E↔F`.

For each position *i*:
- `candidates = { ocr_char : visual }  ∪  { conf_alt : visual·confusion_prior }` — **only** the OCR char and its known visual confusions. Nothing else is ever introduced.
- For each candidate `c`: `score = w_visual·visual + (struct_bonus if allowed else −struct_penalty)`.
- Pick the highest-scoring candidate.

**Sequence scoring** is additive over positions; because the structural rules are per-position, per-position argmax equals the global-sequence optimum.

```
FinalScore = 0.5·mean(chosen_visual) + 0.5·compliance_fraction
```

**Never-invent in practice:**
- `NS1FA814ATJ123456` (QY) → `NSTFA814ATJ123456`: position 3 `1`→`T` because `T` is a known confusion of `1` *and* the rule requires `T`. ✔
- `MSTFA814ATJ123456` (QY) → stays `M…`: `M` has **no** confusion to `N`, so the system does **not** force `N`; it keeps `M`, flags it non-compliant. ✔ (faithful)
- Positions 12–17: a letter with a digit-confusion (`B`→`8`) is corrected; a letter without one (`J`) is kept and penalized. ✔

Result object carries `raw_vin`, `validated_vin`, `model`, `final_score`, `compliance`, `fully_compliant`, and a per-position `decisions[]` list (raw char, chosen char, visual, allowed, changed) for full auditability.

---

## 4. Multi-frame fusion — `ocr_worker.py`

Each plate event submits the top-K best crops + the model. For each crop, OCR is tried with retry variants (original → deskew → ±rotation) within a bounded attempt budget. Every read is post-processed; **accepted** validated VINs (len 17, score ≥ `min_final_score`) become candidates. The final VIN is the **majority vote** across frames on the *validated* string (tie-break: highest total score). The winning entry keeps its **raw** OCR for the DB.

---

## 5. Model source (QY / BL7M)

The model comes from the YOLO detection class (`detector._model_of`). The current `best.pt` is **single-class** (`vin_plate`), so unknown class names fall back to `VIN.default_model` (configurable). **When YOLO is retrained as a 2-class detector (`QY`, `BL7M`), the class name maps to the model automatically** via `VIN.class_to_model` — no code change.

---

## 6. Database

`vin_records` columns: `id, timestamp, detected_vin (validated), raw_vin (audit), model (QY|BL7M), confidence (final_score), image_path`.

- **SQLite** (current): `init_db()` runs an idempotent migration — existing databases get `raw_vin` and `model` added automatically (old rows keep `NULL`).
- **PostgreSQL** (if/when you migrate): 
  ```sql
  ALTER TABLE vin_records ADD COLUMN IF NOT EXISTS raw_vin VARCHAR(32);
  ALTER TABLE vin_records ADD COLUMN IF NOT EXISTS model   VARCHAR(8);
  ```

History UI shows **Model**, **VIN (validated)**, **Raw OCR** (highlighted when it differs), and **Score**; CSV/Excel export includes all columns.

---

## 7. Configuration (`config.py` → `VIN`)

`enabled`, `default_model`, `class_to_model`, `confusion_prior` (0.6), `w_visual` (1.0), `struct_bonus` (0.5), `struct_penalty` (0.5), `min_final_score` (0.55). Year codes live in `vin_rules.YEAR_CODES` (extend with `register_year`).

---

## 8. Tests — `tests/test_vin_rules.py`

18 unit tests covering every position rule (QY & BL7M), the confusion map, the `NS1→NST` correction, the never-invent guarantee, digit-position handling, model-specific position 4, the `pos8 4→A` BL7M case, year mapping + extension, and unknown-model/bad-length fallthrough. Run: `python tests/test_vin_rules.py` (or `pytest -q`).

---

## Known spec notes
- The spec's example row `NSTHB414ESJ987654 | BL7M` has `4` at position 8, but the stated BL7M rule is position 8 ∈ {G, A}. Following the **stated rule**, the engine re-ranks `4`→`A` (a known confusion) → `NSTHB41AESJ987654`. If `4` is actually valid at position 8, update `RULES["BL7M"][7]`.
- True per-character Top-N would require hooking PaddleOCR's recognizer logits; the current engine derives candidates from the confusion map + structure (handles all spec examples). `postprocess()` already accepts a `char_confidences` list for when per-char scores become available.
