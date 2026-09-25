# 01 — Data Forensics

## STATUS: BLOCKED — dataset not present

Inventory of `Business Entity Resolution Challenge/` as of 2026-09-25:

| File | Size | Role |
|---|---|---|
| `6ab5628d5a817_amazon_ml_challenge_problem_statement.pdf` | 173 KB | Source PS |
| `6ab56657b4f1a_guidelines_and_key_instructions_amazon_ml_challenge_2026.pdf` | 71 KB | Source GL |
| `task.txt` | 6 KB | Source TR (video transcript) |

**Missing:** `dataset/train/train_source{1,2,3}.tsv`, `dataset/train/train_ground_truth.tsv`, `dataset/test/test_source{1,2,3}.tsv`, `utils/validate_submission.py`, `Documentation_template.md`. I searched `~/Downloads`, `~/Desktop` and `~/Documents` to depth 5. Only the video (`~/Downloads/6ab509c5b7036_ml_challenge_2026_video.mp4`) is there.

**No statistic in this file is filled in.** Every number below is `TBD` until `src/forensics.py` runs on real data. I refuse to write anecdotes in place of measurements.

## Tooling ready

`code/business_entity_resolution/src/forensics.py --data-dir <dataset> --out 01_DATA_FORENSICS_generated.md`

It has been smoke-tested on a synthetic 3-source fixture (scratchpad only, never used for modelling). It reads with `sep="\t"`, `quoting=QUOTE_NONE`, `dtype=str` and `keep_default_na=False`, so a business literally named "NA" or an empty address is not coerced to NaN.

### Checks it computes (maps to Phase 1 list)

| Phase-1 item | Implemented as |
|---|---|
| record counts S1/S2/S3, train & test | row counts per file |
| duplicate IDs / duplicate rows | `entity_id.duplicated`, full-row dup excl. ID, dup after normalisation |
| missing / blank / whitespace-only | empty, whitespace-only, leading/trailing ws, multi-space, literal `nan/null/none/NA/-` |
| malformed Unicode | mojibake signatures (`Ã.`, `â€`, `Â.`, U+FFFD), control chars, non-ASCII rate |
| punctuation, `&`/and, apostrophes, hyphens | per-field rates |
| legal suffixes | regex over IN/US/FR suffix inventory (pvt, ltd, llp, llc, inc, corp, sarl, sas, eurl…) |
| repeated tokens | `\b(\w+) \1\b` |
| landmark addresses | near / opp / behind / beside / next to … |
| PIN/ZIP patterns | 6-digit (IN PIN), 5-digit (US ZIP **and** FR code postal: same shape, a collision risk), ZIP+4, no-digit addresses |
| country distributions | per file |
| same-name/different-address, same-address/different-name | collision counts + largest group within each file |
| near-identical / duplicated noisy records | dup after normalisation |
| GT distribution 0/1/2/3+ | histogram + full histogram |
| S2-only / S3-only / both | source-mix counts; (#S2, #S3) joint table |
| one S1 → many from one source | count of S1 with ≥2 from S2, ≥2 from S3 |
| **S2/S3 record claimed by >1 S1** | tests whether "deduplicated S1" implies exclusivity (A6) |
| never-matched S2/S3 records | distractor rate in the pool |
| per-country singleton rate | by S1 country |
| positive-pair noise profile | exact-name rate, exact-addr rate, token-Jaccard p10/median, **numeric-conflict rate among true positives**, country-label agreement across a true pair |
| train/test leakage | shared IDs, test records whose normalised (name, addr) already occurs in train |
| samples | 25 random positive pairs, 15 random singletons, 4 rows per country per file (France visible from the test files) |

### Checks that need a human pass after the script (cannot be automated honestly)

1. Read about 200 positive pairs and categorise the noise mechanism (abbreviation, transliteration, landmark, reorder, DBA, typo, component drop) into a frequency table.
2. Read every France test record sample: accents, `rue/av./bd`, `SARL/SAS/EURL`, 5-digit CP placement ("75001 Paris" vs trailing), CEDEX, `bis/ter` numbering.
3. Inspect the largest same-address/different-name groups (malls, business parks, co-working spaces) as the false-merge zone.
4. Inspect the largest same-name/different-address groups (chains, branches) as the other false-merge zone.
5. Diff the noise profile of S2 vs S3 (source-specific vendors per TR 00:30).

## Decisions gated on these numbers

| Measurement | Decision it drives |
|---|---|
| singleton rate (overall, by country) | prior for the empty-list decision; size of the payoff from singleton precision |
| S2/S3 claimed by >1 S1 | whether a global exclusivity/assignment constraint is legal (A6) |
| multi-match-from-same-source rate | whether "top-1 per source" capping costs recall |
| numeric-conflict rate among positives | whether a digit conflict can be a hard veto or only a soft feature |
| country agreement across positives | whether country may be a blocking key at all |
| positives with name Jaccard 0 or address Jaccard 0 | which blocking channel is indispensable |
| test record reuse from train | leakage and whether train-memorisation features are poison |
