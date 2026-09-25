# 00 — Official Rules and Constraints

Sources (all read in full, 2026-09-25):

| Tag | File |
|---|---|
| **PS** | `6ab5628d5a817_amazon_ml_challenge_problem_statement.pdf` (7 content pages + 1 blank) |
| **GL** | `6ab56657b4f1a_guidelines_and_key_instructions_amazon_ml_challenge_2026.pdf` (2 pages) |
| **TR** | `task.txt` — transcript of the challenge video (`6ab509c5b7036_ml_challenge_2026_video.mp4`, 00:00–05:35) |

Referenced but **NOT supplied**: `dataset/train/*`, `dataset/test/*`, `utils/validate_submission.py`, `Documentation_template.md`, the "student_resource/" directory, the "Prep before you start" blog, the Google Form. None exist on this machine (searched `~/Downloads`, `~/Desktop`, `~/Documents`, depth 5).

---

## 1. Hard submission constraints

| # | Rule | Source |
|---|---|---|
| H1 | All files are tab-separated `.tsv`; submissions must be tab-separated too. | PS p1 |
| H2 | `matching_results.tsv` columns exactly `source1_entity_id`, `matched_entity_ids`. | PS p3 |
| H3 | ID lists are comma-separated, **no quoting**, no spaces shown in example. | PS p3 |
| H4 | Every test S1 entity has **exactly one row**. Missing entities → rejection. | PS p3, p5 (C3) |
| H5 | Singletons: `matched_entity_ids` **empty**. | PS p3 |
| H6 | No duplicate IDs within a list; no duplicate `source1_entity_id` rows → rejection. | PS p3, p5 (C4) |
| H7 | Lists may only contain S2/S3 IDs that **exist in the test set**. Self-matches to S1 → rejection. | PS p3, p5 (C2) |
| H8 | Submissions failing validation are **not evaluated**; valid ones show `SCORED`. | PS p5 (C1) |
| H9 | Leaderboard upload = `matching_results.tsv` only. | PS p3, p6; TR 03:49 |

## 2. Evaluation constraints

| # | Rule | Source |
|---|---|---|
| E1 | Metric: F_β, β = 0.5. `F0.5 = 1.25·P·R / (0.25·P + R)`. | PS p6; TR 04:27 |
| E2 | **Macro** average: F0.5 per S1 entity, averaged over all S1 entities in the eval set. | PS p6 |
| E3 | Singleton (no true matches): predicted empty → 1.0; any prediction → 0.0. | PS p6; TR 05:08 |
| E4 | Worked example: pred {a,b,c}, truth {a,c} → P=2/3, R=1, F=0.714. | PS p6 |
| E5 | Public LB = subset of test; private LB = remaining portion; full test predicted in both cases. | PS p6 |
| E6 | **Not specified:** score when truth non-empty and prediction empty (P undefined). Standard convention and the only sane reading is 0.0. Our evaluator uses 0.0. | inferred — flagged A7 |

## 3. Data constraints

| # | Rule | Source |
|---|---|---|
| D1 | Source files have columns `entity_id, business_name, business_address, country`. | PS p1 |
| D2 | Source is identified by ID prefix `S1-`/`S2-`/`S3-` and file; no source column. | PS p2 |
| D3 | S1 is the **deduplicated reference source**. | PS p1, p2; TR 01:21 |
| D4 | An S1 entity may match **zero, one or many** S2/S3 records. | PS p1; TR 01:21 |
| D5 | GT: `source1_entity_id`, `matched_entity_ids` (comma list, empty for no match). One row per S1. | PS p2; TR 03:23 |
| D6 | Train countries: US, India. Test adds **France** (not in train). | PS p1–2 |
| D7 | `country` is an **open set**: do not hard-code, filter or one-hot to {US, India}; every France entity must be in the submission. | PS p2 |
| D8 | Expected noise — names: abbreviations (Corp/Corporation, Pvt/Private, Ltd/Limited), legal suffix inconsistency, DBA/trade names, `&` vs "and", word-order transposition, typos, transliterations. | PS p1–2 |
| D9 | Expected noise — addresses: abbreviations (Rd/Road, St/Street), transliteration variants, missing components (no PIN, no state), landmark references ("Near SBI ATM"), municipal numbering formats, component reordering, partial addresses. | PS p1–2 |
| D10 | Fields deliberately limited to name + address. | TR 00:51 |
| D11 | No GT for test; hold out a validation split from train and score with F0.5 yourself. | PS p3 |

## 4. Fair-play restrictions

| # | Rule | Source |
|---|---|---|
| F1 | **Strictly prohibited**: external databases, APIs or services to look up business identities or resolve entities — commercial ER APIs, government registries, geocoding APIs, any external data augmentation from internet sources. | PS p7; TR 05:35 |
| F2 | Any evidence of external lookup → **immediate disqualification**. All code/pipelines reviewed. | PS p7 |
| F3 | "Designed to test ML skills using only the provided training data." | PS p7 |
| F4 | No cheating/plagiarism; no multiple IDs; no simultaneous logins; desktop/laptop only. | GL p1–2 |

## 5. Model restrictions

| # | Rule | Source |
|---|---|---|
| M1 | "Final model should be a **MIT/Apache 2.0 License** model and **up to 8 Billion parameters**." | PS p5 (C5) |

Our interpretation (flagged A4): take the strict reading. The final matching classifier will be implemented with an MIT- or Apache-2.0-licensed library. **LightGBM (MIT)** and **XGBoost (Apache-2.0)** qualify. scikit-learn is BSD-3-Clause, so it is used only for utilities and baselines, not the final scorer, unless the organisers clarify. rapidfuzz (MIT) is used for string metrics. Any pretrained embedding model must be MIT/Apache-2.0 **and** ≤ 8B params **and** loaded offline with its weights pinned.

## 6. Candidate-generation requirements

| # | Rule | Source |
|---|---|---|
| C1 | `candidate_pairs.tsv` columns `source1_entity_id`, `candidate_entity_ids`. | PS p4 |
| C2 | It must be the **last** filtering stage: exactly the set the matching model runs inference over, not an earlier blocking pass. | PS p3–4 |
| C3 | One row per S1; empty when no candidates; S2/S3 IDs only; no duplicates in a list. | PS p4 |
| C4 | Final matches should be a **subset** of candidates. A violation "signals a pipeline bug — the validator **warns**". | PS p4 |
| C5 | Not scored. Used to audit blocking (recall ceiling, **reduction ratio**) and to verify the pipeline. | PS p4; TR 04:04 |
| C6 | Blocking described as a cheap key built from name **and** address; records may group through name or through address; blocking favours recall. Descriptive, not mandatory. | TR 01:49–02:43 |

## 7. Output requirements

| # | Rule | Source |
|---|---|---|
| O1 | Both TSVs go in `output/` of the final package. | PS p3 |
| O2 | Run `python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test` from `student_resource/`. PASS → exit 0; otherwise issues are listed and it exits 1. Stdlib only; does not compute score. | PS p4; TR 04:27 |

## 8. Final package / reproducibility

| # | Rule | Source |
|---|---|---|
| R1 | Single zip `<team_name>_submission.zip` with `output/{matching_results,candidate_pairs}.tsv`, `code/business_entity_resolution/{src/,README.md,requirements.txt}`, `Documentation_template.md`. | PS p5 |
| R2 | Code must be self-contained and runnable. "Anyone should be able to regenerate both output files from the training/test data using only what is in this folder." Versions pinned. | PS p5 |
| R3 | Methodology doc must cover: methodology, candidate generation/blocking, model architecture and feature engineering, other relevant info. `.md` or `.pdf`. | PS p5, p7; GL p1–2 |
| R4 | Source code for experiments, training and inference, "with proper comments describing the functions". | GL p1 |
| R5 | Maintain version history of all submissions; final source code may be requested later. | GL p1 |
| R6 | Top teams' packages are reviewed in detail (fair-play and license) before final rankings. | PS p5, p7; TR 04:27 |

## 9. Leaderboard and submission limits

| # | Rule | Source |
|---|---|---|
| L1 | Window: **25 Sep 2026 00:00 IST → 27 Sep 2026 23:59 IST.** | GL p1 |
| L2 | **Max 5 submissions per day**, for 3 days (≤ 15 total). The submit button is disabled afterwards. | GL p1 |
| L3 | Live public leaderboard; the final leaderboard is revealed after the challenge. | GL p1 |
| L4 | Top 100 teams are announced after artefact submission, LB score and eligibility checks. | GL p1 |

---

## 10. Ambiguities and apparent contradictions (NOT silently resolved)

| ID | Issue | Sources in tension | Working position until clarified |
|---|---|---|---|
| **A1** | **Methodology length**: PS says "no page limit — prioritise clarity and technical depth"; GL says a "1-2-page document". | PS p7 vs GL p1 | Write the full template for the zip (PS governs the package). Also produce a 1–2 page executive summary so both are satisfied. |
| **A2** | **What decides the ranking**: PS says "final decision will be based on the private leaderboard"; GL says "evaluation and shortlisting will be based on performance across both leaderboards". | PS p6 vs GL p1 | Optimise private LB (the held-out, unbiased half) and never trade private robustness for public-LB gain. The public score still matters for shortlisting, so no deliberately sandbagged public submissions. |
| **A3** | **Subset rule severity**: "should be a subset… the validator *warns*" versus a hard rejection. | PS p4 | Treat it as a hard invariant; we assert it in code. |
| **A4** | **Scope of "Final model… MIT/Apache 2.0, ≤ 8B"**: does it cover libraries (scikit-learn = BSD-3), or only pretrained model weights? | PS p5 | Strict reading: final scorer in LightGBM (MIT) or XGBoost (Apache-2.0). scikit-learn only for non-final utilities. Ask via the Google Form. |
| **A5** | **Package self-containment vs data**: "regenerate… from the training/test data using only what is in this folder", yet the data is not listed inside `code/`. | PS p5 | README takes `--data-dir` pointing at `dataset/`. The code must never assume absolute paths or cached artefacts. |
| **A6** | **Can one S2/S3 record match multiple S1s?** S1 is "deduplicated", which suggests each S2/S3 record maps to at most one S1, but this is never stated. | PS p1 | **NOT PROVEN.** Measure it on train GT. If it holds, a global one-to-many assignment constraint becomes a strong precision tool. |
| **A7** | **Score when truth ≠ ∅ and prediction = ∅** is not stated (precision is 0/0). | PS p6 | 0.0 (recall = 0 ⇒ F = 0 under any convention). |
| **A8** | **Public/private split mechanism** (random? stratified by country? is France in both halves?) is not stated. | PS p6 | Assume a random split over S1 entities, with France present in both halves. France failures therefore hit private too. |
| **A9** | **Daily submission quota rollover** (does unused quota carry over?) is not stated. | GL p1 | Assume no rollover. |
| **A10** | **Country label in test** may be spelled differently ("France" vs "FR") or be missing or noisy per record. | PS p2 | Treat `country` as an untrusted string. Never gate candidate generation on exact country equality without evidence. |
| **A11** | **Pretrained model download** (e.g. multilingual MiniLM, Apache-2.0) is not an "entity lookup", but it is an internet artefact. Is it "external data augmentation"? | PS p5 vs p7 | Allowed only if weights are pinned, licensed MIT/Apache, used offline, and learn nothing from web data about specific entities. Default: **avoid**. Justify only if it beats the tree model in entity-level CV. |
| **A12** | **Transliteration without external lookup**: `unidecode` is GPL-2.0 (a licence risk if it counts as part of the "model"). `unicodedata` NFKD accent stripping is stdlib. | PS p5 | Use stdlib `unicodedata` only. |
| **A13** | GL asks for "source code used for **experiments**" as well as training and inference. | GL p1 | Keep `experiments/` scripts inside the package. |

## 11. Time budget (hard reality)

Today is **2026-09-25** and the window closes on **2026-09-27 23:59 IST** (≈ 3 days). The quota is 5 submissions/day, and the data is not yet on disk. Every hour without data is lost forensics time.
