# 02 — Validation Methodology and Experiment Ladder

## 1. Unit of validation: the S1 entity

The metric is a macro average over S1 entities, so the fold unit is the S1 entity **with its entire match set**. A random pair split is **REJECTED** for three reasons. It puts sibling pairs of one S1 on both sides. It lets the decision layer (top-k, margin, per-source caps), which operates on an S1's candidate set as a whole, see half an entity. And it cannot compute macro F0.5 at all.

### Primary scheme
- **Repeated stratified GroupKFold**: 5 folds × 3 seeds (15 evaluations), grouped by S1 ID.
- Stratify on `country × match-count bucket {0,1,2,3+} × source-mix {none, S2, S3, both}`.
- Candidate pool for a validation S1 = the **entire** train S2 ∪ S3 pool, including records matched to training-fold S1s. This is realistic: test S1s see a pool full of other entities' records and distractors. Removing them would inflate precision.
- Report mean ± std and the **worst fold**. A change is accepted only if the mean improves **and** the worst fold does not regress beyond noise (paired comparison on identical folds).

### Leakage controls
| Risk | Control |
|---|---|
| Model sees the validation S1's own positive pairs | pairs are generated per S1; the fold split is on S1 before any pair exists |
| Train-derived frequency stats (token IDF, name frequency) computed with validation labels | stats computed **only from unlabeled record text** (all S1/S2/S3 records, no GT). Label-free ⇒ no label leakage. At inference, recompute on the test pool (transductive, uses only provided test records, no labels) |
| Target-like features ("how many S1 did this S2 match in training") | **forbidden** — label leakage and meaningless on test |
| Near-duplicate S1s across folds (S1 is "deduplicated", but check) | forensics dup-after-normalisation count. If > 0, group near-duplicate S1s into one fold group |
| Threshold tuned on the same predictions it is reported on | **nested**: thresholds chosen on OOF predictions of folds ≠ k, then applied to fold k. Also report the "optimistic" number and the gap between them |
| Pool-size shift train → test | check candidates/S1 and density features on test vs train. Features that are raw counts get normalised (rank or ratio) or dropped |

### France / open-country robustness (France has no labels, so this is proxy evidence only)
1. **Leave-one-country-out**: train on US and evaluate on India, and the reverse. A feature set whose LOCO drop is much larger than an ablated, country-agnostic variant is flagged as country-overfit.
2. **Country-agnostic ablation**: disable every country-specific rule (PIN regex, ZIP regex, Indian suffix list) and measure the CV cost. If the cost is small, prefer the agnostic version.
3. **Test-distribution diagnostics (unsupervised)**: on test, compare France against US/India for candidates/S1, max-score distribution, predicted-singleton rate and predicted matches/S1. A France predicted-singleton rate far from the train singleton rates means miscalibration. It is not proof, but it is an alarm.
4. **Synthetic perturbation suite** (Phase 11): accent strip/add, apostrophes, `rue`/`r.`, `SARL`/`SAS` suffix drop, 5-digit CP removal. It uses only operators observed in the train noise profile plus Unicode operators that the problem statement names (transliteration).

## 2. Metrics reported for every experiment
Macro F0.5 (primary). Singleton accuracy. Non-singleton macro F0.5. Rate of non-singletons predicted empty. Micro pair precision/recall. **Blocking recall** (pair and entity-level: fraction of S1 whose full truth set ⊆ candidates). Candidates/S1 at mean, p50, p90, p95, p99 and max. Reduction ratio. Runtime. Peak memory. Everything is broken down by country and by source (S2 vs S3).

Evaluator: `src/metric.py`, unit-tested against the official worked example (0.714) and the singleton rules.

## 3. Experiment ladder (each rung must beat the previous one on nested CV to survive)

| EXP | Rung | Purpose / hypothesis to falsify |
|---|---|---|
| 000 | Forensics | numbers in 01_DATA_FORENSICS; no model |
| 001 | Baseline A: exact normalised name+address | precision ceiling of a trivial rule; singleton floor |
| 002 | Baseline B: rapidfuzz name/address thresholds | how far deterministic fuzzy gets |
| 003 | Baseline C: char-TF-IDF retrieval + threshold | retrieval-only ceiling |
| 010 | Blocking study | union of channels; recall vs size curve; **every missed positive categorised** |
| 020 | Baseline D: feature bank + logistic regression | linear reference (sklearn, baseline only, A4) |
| 021 | Baseline E: feature bank + GBDT (LightGBM MIT / XGBoost Apache-2.0) | candidate final scorer |
| 030 | Entity-level decision layer | top-1/top-k, margin, per-source handling, exclusivity (only if A6 holds) |
| 040 | Dedicated singleton gate | whether an entity-level "has any match" model beats a pure pairwise threshold |
| 050 | Robustness | LOCO, country-agnostic ablation, seed stability |
| 060 | Adversarial edge-case suite | Phase 11 list; each case is a test with an expected behaviour |
| 070 | Final pipeline | clean-room run → TSVs → validator → subset invariant → zip |

**Blocking policy:** candidate_pairs.tsv = the exact set scored by the final model. If the decision layer adds a filter (e.g. top-k), that filter is applied **before** writing candidates or it counts as part of the model. Resolve this explicitly in EXP_030.

## 4. Submission budget (≤ 5/day, ≤ 15 total, window closes 2026-09-27 23:59 IST)
| Day | Max use | Rule |
|---|---|---|
| 1 | 2–3 | S1 = best validated baseline (sanity check for format + CV↔LB correlation). S2 = GBDT. |
| 2 | 3–4 | Only hypotheses with a CV gain > 1 fold-std. |
| 3 | 2–3 | Final candidates. Hold ≥ 1 in reserve. |
Never submit threshold sweeps. The final choice is made on **CV**, with LB as a secondary check.

## 5. Model/licence policy (A4)
Final scorer: LightGBM (MIT) or XGBoost 3.4.1 (Apache-2.0, already installed). String metrics: rapidfuzz 3.14.6 (MIT, installed). Normalisation: stdlib `unicodedata` (no GPL `unidecode`). scikit-learn (BSD-3) is used only for baselines and CV utilities. Any library that performs network calls at runtime is rejected. No pretrained embedding model unless it beats the GBDT on nested CV **and** passes the licence audit.
