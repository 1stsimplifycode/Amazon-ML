"""Phase-1 data forensics. Produces a markdown report of evidence (no modelling).

Usage:
    python src/forensics.py --data-dir <path/to/dataset> --out 01_DATA_FORENSICS_generated.md

Reads with explicit sep='\\t', quoting disabled, all columns as str, keep_default_na=False
so blank strings are never silently turned into NaN or merged with the literal "NA".
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import unicodedata
from collections import Counter, defaultdict

import warnings

import pandas as pd

warnings.filterwarnings("ignore", message="This pattern is interpreted as a regular expression")

COLS = ["entity_id", "business_name", "business_address", "country"]
MOJIBAKE = re.compile(r"(Ã.|â€|Â.|�)")
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
LANDMARK = re.compile(r"\b(near|nr|opp|opposite|behind|beside|next to|in front of|adjacent|close to|landmark)\b", re.I)
LEGAL = re.compile(r"\b(pvt|private|ltd|limited|llp|llc|inc|incorporated|corp|corporation|co|company|plc|"
                   r"sarl|sas|sa|eurl|sasu|sci|gmbh|opc|lp|pllc|pc|dba)\b\.?", re.I)
NUM = re.compile(r"\d+")
PIN6 = r"(?<!\d)\d{6}(?!\d)"
DIG5 = r"(?<!\d)\d{5}(?!\d)"
ZIP4 = r"\d{5}-\d{4}"
REPEAT = r"\b(\w+) \1\b"
APOS = "['’]"
NODIGIT = r"[0-9]"
DIGITS_ONLY = r"[0-9 ]+"


def load(path: str) -> pd.DataFrame:
    """Load a TSV as raw strings; never coerce blanks/'NA' to NaN."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False,
                       quoting=csv.QUOTE_NONE, encoding="utf-8")


def norm(s: str) -> str:
    """Forensic-only normaliser: NFKD, strip accents, lowercase, punctuation->space."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower().replace("&", " and ")
    return " ".join(re.sub(r"[^\w]+", " ", s).split())


def toks(s: str) -> set:
    return set(norm(s).split())


def jacc(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def pct(x, n):
    return f"{x} ({100 * x / n:.2f}%)" if n else f"{x}"


def quant(series) -> str:
    q = pd.Series(series).quantile([0, .5, .9, .95, .99, 1]).round(3).tolist()
    return f"min {q[0]} / p50 {q[1]} / p90 {q[2]} / p95 {q[3]} / p99 {q[4]} / max {q[5]}"


def audit_source(name: str, df: pd.DataFrame, prefix: str, L: list) -> None:
    """Per-file structural audit: schema, IDs, duplicates, blanks, unicode, patterns."""
    n = len(df)
    L.append(f"\n### {name}  (rows = {n})\n")
    L.append(f"- columns: `{list(df.columns)}` {'OK' if list(df.columns) == COLS else '**SCHEMA MISMATCH**'}")
    if list(df.columns) != COLS:
        return
    L.append(f"- IDs with wrong prefix (≠ `{prefix}`): {pct((~df.entity_id.str.startswith(prefix)).sum(), n)}")
    L.append(f"- duplicate entity_id: {pct(df.entity_id.duplicated().sum(), n)}")
    L.append(f"- exact duplicate rows (excl. id): {pct(df[COLS[1:]].duplicated().sum(), n)}")
    nn = df.business_name.map(norm); na = df.business_address.map(norm)
    L.append(f"- duplicate (norm name, norm addr) excl. id: {pct(pd.DataFrame({'a': nn, 'b': na}).duplicated().sum(), n)}")
    for c in COLS[1:]:
        v = df[c]
        L.append(f"- `{c}`: empty {pct((v == '').sum(), n)}, whitespace-only {pct(((v != '') & (v.str.strip() == '')).sum(), n)}, "
                 f"leading/trailing ws {pct((v != v.str.strip()).sum(), n)}, multi-space {pct(v.str.contains('  ').sum(), n)}, "
                 f"non-ASCII {pct(v.map(lambda s: any(ord(ch) > 127 for ch in s)).sum(), n)}, "
                 f"mojibake {pct(v.str.contains(MOJIBAKE).sum(), n)}, control chars {pct(v.str.contains(CONTROL).sum(), n)}, "
                 f"literal 'nan/null/none/na' {pct(v.str.strip().str.lower().isin(['nan', 'null', 'none', 'na', 'n/a', '-']).sum(), n)}")
    L.append(f"- name length chars: {quant(df.business_name.str.len())}")
    L.append(f"- addr length chars: {quant(df.business_address.str.len())}")
    L.append(f"- name tokens: {quant(nn.str.split().map(len))}; addr tokens: {quant(na.str.split().map(len))}")
    L.append(f"- 1-char names: {pct((nn.str.len() <= 1).sum(), n)}; all-digit names: {pct(nn.str.fullmatch(DIGITS_ONLY).fillna(False).sum(), n)}")
    L.append(f"- names with legal-suffix token: {pct(df.business_name.str.contains(LEGAL).sum(), n)}")
    L.append(f"- names with '&': {pct(df.business_name.str.contains('&', regex=False).sum(), n)}; with ' and ': {pct(df.business_name.str.lower().str.contains(' and ', regex=False).sum(), n)}")
    L.append(f"- names with apostrophe: {pct(df.business_name.str.contains(APOS).sum(), n)}; hyphen: {pct(df.business_name.str.contains('-', regex=False).sum(), n)}")
    L.append(f"- addresses with landmark phrase: {pct(df.business_address.str.contains(LANDMARK).sum(), n)}")
    L.append(f"- addresses with 6-digit number (IN PIN-like): {pct(df.business_address.str.contains(PIN6).sum(), n)}; "
             f"5-digit (US ZIP / FR CP-like): {pct(df.business_address.str.contains(DIG5).sum(), n)}; "
             f"ZIP+4: {pct(df.business_address.str.contains(ZIP4).sum(), n)}; no digits at all: {pct((~df.business_address.str.contains(NODIGIT)).sum(), n)}")
    L.append(f"- country values: `{dict(Counter(df.country).most_common(12))}`")
    # repeated tokens within a name ("acme acme")
    L.append(f"- names with an immediately repeated token: {pct(nn.str.contains(REPEAT).sum(), n)}")
    # collisions inside the file
    by_name = defaultdict(set); by_addr = defaultdict(set)
    for a, b in zip(nn, na):
        by_name[a].add(b); by_addr[b].add(a)
    L.append(f"- same norm-name with >1 distinct norm-address: {sum(len(v) > 1 for v in by_name.values())} names "
             f"(largest {max((len(v) for v in by_name.values()), default=0)})")
    L.append(f"- same norm-address with >1 distinct norm-name: {sum(len(v) > 1 and k != '' for k, v in by_addr.items())} addresses "
             f"(largest {max((len(v) for k, v in by_addr.items() if k), default=0)})")
    top_names = Counter(nn).most_common(10)
    L.append(f"- most frequent norm names: `{top_names}`")
    tokc = Counter(t for s in nn for t in s.split())
    L.append(f"- most frequent name tokens: `{tokc.most_common(25)}`")
    L.append(f"- sample rows by country:")
    for c, g in df.groupby("country"):
        for _, r in g.sample(min(4, len(g)), random_state=0).iterrows():
            L.append(f"    - [{c}] `{r.entity_id}` | `{r.business_name}` | `{r.business_address}`")


def audit_gt(gt: pd.DataFrame, s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame, L: list) -> None:
    """Ground-truth audit: match-count distribution, source mix, many-to-one, pair similarity."""
    L.append("\n## Ground truth\n")
    L.append(f"- columns `{list(gt.columns)}`; rows {len(gt)}; duplicate S1 rows {gt.source1_entity_id.duplicated().sum()}")
    s1ids, s2ids, s3ids = set(s1.entity_id), set(s2.entity_id), set(s3.entity_id)
    g1 = set(gt.source1_entity_id)
    L.append(f"- S1 in source file but missing from GT: {len(s1ids - g1)}; GT S1 not in source file: {len(g1 - s1ids)}")
    m = {r.source1_entity_id: [x.strip() for x in r.matched_entity_ids.split(",") if x.strip()] for r in gt.itertuples()}
    dup_in_list = sum(len(v) != len(set(v)) for v in m.values())
    allm = [x for v in m.values() for x in v]
    L.append(f"- lists with duplicate ids: {dup_in_list}; matched ids not in S2/S3 files: {sum(x not in s2ids | s3ids for x in allm)}")
    n = len(m)
    cnt = Counter(min(len(v), 3) for v in m.values())
    L.append(f"- match-count distribution: 0 → {pct(cnt[0], n)}, 1 → {pct(cnt[1], n)}, 2 → {pct(cnt[2], n)}, 3+ → {pct(cnt[3], n)}")
    L.append(f"- full size histogram: `{sorted(Counter(len(v) for v in m.values()).items())}`")
    mix = Counter()
    per_src_max = Counter()
    for v in m.values():
        a = sum(x.startswith("S2-") for x in v); b = sum(x.startswith("S3-") for x in v)
        mix["none" if not v else "S2 only" if b == 0 else "S3 only" if a == 0 else "both"] += 1
        per_src_max[(min(a, 3), min(b, 3))] += 1
    L.append(f"- source mix: `{dict(mix)}`")
    L.append(f"- (#S2, #S3) per S1 (capped 3): `{sorted(per_src_max.items())}`")
    L.append(f"- S1 with ≥2 matches from the SAME source: S2 {sum(sum(x.startswith('S2-') for x in v) >= 2 for v in m.values())}, "
             f"S3 {sum(sum(x.startswith('S3-') for x in v) >= 2 for v in m.values())}")
    owner = Counter(allm)
    L.append(f"- **S2/S3 records claimed by >1 S1 (tests A6)**: {sum(c > 1 for c in owner.values())} (max {max(owner.values(), default=0)})")
    L.append(f"- S2 records never matched: {pct(len(s2ids - set(allm)), len(s2ids))}; S3 never matched: {pct(len(s3ids - set(allm)), len(s3ids))}")
    # singleton rate by country
    c1 = dict(zip(s1.entity_id, s1.country))
    by_c = defaultdict(list)
    for k, v in m.items():
        by_c[c1.get(k, "?")].append(len(v))
    for c, v in by_c.items():
        L.append(f"- country {c}: n={len(v)}, singleton rate {sum(x == 0 for x in v) / len(v):.3f}, mean matches {sum(v) / len(v):.3f}")
    # positive-pair similarity profile
    rec = {r.entity_id: r for r in pd.concat([s1, s2, s3]).itertuples()}
    rows = []
    for k, v in m.items():
        a = rec[k]
        for x in v:
            b = rec.get(x)
            if b is None:
                continue
            na_, nb_ = norm(a.business_name), norm(b.business_name)
            aa, ab = norm(a.business_address), norm(b.business_address)
            da, db = set(NUM.findall(a.business_address)), set(NUM.findall(b.business_address))
            rows.append(dict(src=x[:2], country=a.country, country_eq=a.country == b.country,
                             name_eq=na_ == nb_, addr_eq=aa == ab,
                             name_j=jacc(set(na_.split()), set(nb_.split())),
                             addr_j=jacc(set(aa.split()), set(ab.split())),
                             num_conflict=bool(da and db and not (da & db)),
                             num_any_overlap=bool(da & db)))
    P = pd.DataFrame(rows)
    if len(P):
        L.append(f"\n### Positive-pair profile (n={len(P)})\n")
        L.append(f"- country label equal across pair: {P.country_eq.mean():.4f}  ← if <1, country is NOT a safe blocking key")
        L.append(P.groupby(["src", "country"]).agg(n=("name_eq", "size"), name_exact=("name_eq", "mean"),
                                                   addr_exact=("addr_eq", "mean"), name_j_p10=("name_j", lambda s: s.quantile(.1)),
                                                   name_j_med=("name_j", "median"), addr_j_p10=("addr_j", lambda s: s.quantile(.1)),
                                                   addr_j_med=("addr_j", "median"), num_conflict=("num_conflict", "mean"),
                                                   num_overlap=("num_any_overlap", "mean")).round(3).to_string().join([chr(10)+"```"+chr(10), chr(10)+"```"]))
        L.append(f"\n- positives with name Jaccard == 0: {pct((P.name_j == 0).sum(), len(P))}; addr Jaccard == 0: {pct((P.addr_j == 0).sum(), len(P))}; both == 0: {pct(((P.name_j == 0) & (P.addr_j == 0)).sum(), len(P))}")
        L.append(f"- positives with a numeric conflict (both have numbers, none shared): {pct(P.num_conflict.sum(), len(P))}  ← how dangerous is a hard digit-veto?")
    # sample positives
    L.append("\n### 25 random positive pairs\n")
    import random
    rnd = random.Random(0)
    pairs = [(k, x) for k, v in m.items() for x in v]
    for k, x in rnd.sample(pairs, min(25, len(pairs))):
        a, b = rec[k], rec.get(x)
        if b is not None:
            L.append(f"- `{a.business_name}` | `{a.business_address}` [{a.country}]  ⇄  `{b.business_name}` | `{b.business_address}` [{b.country}]")
    L.append("\n### 15 random singletons\n")
    sing = [k for k, v in m.items() if not v]
    for k in rnd.sample(sing, min(15, len(sing))):
        a = rec[k]
        L.append(f"- `{a.business_name}` | `{a.business_address}` [{a.country}]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="directory containing train/ and test/")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    L = ["# 01 — Data forensics (generated by src/forensics.py)\n"]
    data = {}
    for split in ["train", "test"]:
        L.append(f"\n## Split: {split}\n")
        for k in (1, 2, 3):
            p = os.path.join(a.data_dir, split, f"{split}_source{k}.tsv")
            if not os.path.exists(p):
                L.append(f"- **MISSING FILE** `{p}`"); continue
            df = load(p); data[(split, k)] = df
            audit_source(f"{split}_source{k}", df, f"S{k}-", L)
    # cross-split ID / record overlap (leakage between train and test)
    L.append("\n## Train/test overlap\n")
    for k in (1, 2, 3):
        if ("train", k) in data and ("test", k) in data:
            tr, te = data[("train", k)], data[("test", k)]
            ktr = set(zip(tr.business_name.map(norm), tr.business_address.map(norm)))
            kte = list(zip(te.business_name.map(norm), te.business_address.map(norm)))
            L.append(f"- source{k}: shared entity_ids {len(set(tr.entity_id) & set(te.entity_id))}; "
                     f"test records whose (norm name, norm addr) appear in train: {pct(sum(x in ktr for x in kte), len(kte))}")
    gtp = os.path.join(a.data_dir, "train", "train_ground_truth.tsv")
    if os.path.exists(gtp) and all(("train", k) in data for k in (1, 2, 3)):
        audit_gt(load(gtp), data[("train", 1)], data[("train", 2)], data[("train", 3)], L)
    else:
        L.append("\n**Ground truth missing — GT audit skipped.**")
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
