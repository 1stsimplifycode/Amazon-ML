"""02 noise profile: quantified noise rates on true pairs vs negative candidates.

usage: python noise_profile.py <dataset_dir> <out_md> [n_per_cell]
Negatives: (a) random same-country pool record ("easy"), (b) the best-looking
non-matching pool record sharing the S1's rarest name token ("hard"). Both come
from train only.
"""
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

from io_utils import load_gt_pairs, load_source
from normalize import LEGAL, addr_forms, has_indic, name_forms

data, out = sys.argv[1], sys.argv[2]
NPC = int(sys.argv[3]) if len(sys.argv) > 3 else 40000
rng = np.random.RandomState(11)

gt, pairs = load_gt_pairs(data)
s1 = load_source(data, "train", 1)
pool = pd.concat([load_source(data, "train", 2), load_source(data, "train", 3)], ignore_index=True)
s1i = s1.set_index("entity_id"); pooli = pool.set_index("entity_id")
matched = set(pairs.m.tolist())
pairs["src"] = pairs.m.str[:2]
pairs["country"] = s1i.country.reindex(pairs.s1).values

samples = []
for (src, c), g in pairs.groupby(["src", "country"]):
    g = g.sample(min(NPC, len(g)), random_state=rng)
    samples.append(g.assign(label="pos"))
    # easy negatives: random same-country, same-source pool record that is not this S1's match
    cand = pool[(pool.country == c) & pool.entity_id.str.startswith(src)].entity_id
    neg = cand.sample(len(g), random_state=rng, replace=True).values
    samples.append(pd.DataFrame({"s1": g.s1.values, "m": neg, "src": src, "country": c, "label": "neg_random"}))
S = pd.concat(samples, ignore_index=True)
S = S[~((S.label != "pos") & S.m.isin(matched) & (S.m.map(dict(zip(pairs.m, pairs.s1))) == S.s1))]

# hard negatives: same-country pool record sharing the S1's rarest core-name token, not a true match
print("building hard negatives", flush=True)
hs = S[S.label == "pos"].drop_duplicates("s1").sample(min(60000, S.s1.nunique()), random_state=rng)
tok_pool = pool[["entity_id", "business_name", "country"]].copy()
tok_pool["tok"] = tok_pool.business_name.str.lower().str.findall(r"[a-z0-9]{3,}")
ex = tok_pool[["entity_id", "country", "tok"]].explode("tok").dropna()
df_ = ex.tok.value_counts()
s1_small = s1i.loc[hs.s1.values]
rare = s1_small.business_name.str.lower().str.findall(r"[a-z0-9]{3,}").map(
    lambda ts: min(ts, key=lambda t: df_.get(t, 0) if df_.get(t, 0) > 1 else 1e18) if ts else None)
q = pd.DataFrame({"s1": hs.s1.values, "tok": rare.values, "country": s1_small.country.values, "src": hs.src.values}).dropna()
q = q[q.tok.map(df_).fillna(0).between(2, 3000)]  # a shared token that is neither unique nor generic
ex = ex[ex.tok.isin(set(q.tok))]
truth_of = dict(zip(pairs.m, pairs.s1))
j = q.merge(ex, on=["tok", "country"])
j = j[j.entity_id.str[:2] == j.src]
j = j[j.entity_id.map(truth_of).fillna("") != j.s1]
j = j.groupby("s1").head(1)
S = pd.concat([S, pd.DataFrame({"s1": j.s1.values, "m": j.entity_id.values, "src": j.src.values,
                                "country": j.country.values, "label": "neg_hard"})], ignore_index=True)
del ex, tok_pool

A = s1i.reindex(S.s1); B = pooli.reindex(S.m)
na = [name_forms(x) for x in A.business_name.tolist()]; nb = [name_forms(x) for x in B.business_name.tolist()]
aa = [addr_forms(x) for x in A.business_address.tolist()]; ab = [addr_forms(x) for x in B.business_address.tolist()]
ra, rb = A.business_name.tolist(), B.business_name.tolist()
xa, xb = A.business_address.tolist(), B.business_address.tolist()

ABBR_PAIRS = [("private", "pvt"), ("limited", "ltd"), ("corporation", "corp"), ("company", "co"), ("incorporated", "inc")]
ST_PAIRS = [("street", "st"), ("road", "rd"), ("avenue", "ave"), ("drive", "dr"), ("lane", "ln"), ("court", "ct"),
            ("boulevard", "blvd"), ("place", "pl"), ("rue", "r"), ("saint", "st")]
STATE_TAIL = re.compile(r",\s*([^,]+)$")


def words(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def feats(i):
    n1, n2, a1, a2 = na[i], nb[i], aa[i], ab[i]
    t1, t2 = n1["n_tok"].split(), n2["n_tok"].split()
    c1, c2 = set(n1["n_core"].split()), set(n2["n_core"].split())
    w1, w2 = set(words(ra[i])), set(words(rb[i]))
    typo = 0
    for t in c2 - c1:
        if len(t) >= 4 and any(Levenshtein.distance(t, u) <= 2 for u in c1 if abs(len(u) - len(t)) <= 2):
            typo = 1; break
    abbr = any((x in w1 and y in w2) or (y in w1 and x in w2) for x, y in ABBR_PAIRS)
    at1, at2 = a1["a_tok"].split(), a2["a_tok"].split()
    raw1, raw2 = set(words(xa[i])), set(words(xb[i]))
    stab = any((x in raw1 and y in raw2) or (y in raw1 and x in raw2) for x, y in ST_PAIRS)
    n_1, n_2 = a1["a_nums"].split(), a2["a_nums"].split()
    tail = STATE_TAIL.search(xa[i])
    return {
        "name_exact_raw": ra[i] == rb[i],
        "name_eq_casefold": ra[i].lower() == rb[i].lower(),
        "name_eq_after_fold": n1["n_fold"] == n2["n_fold"],
        "name_eq_after_punct": n1["n_tok"] == n2["n_tok"],
        "name_eq_core": n1["n_core"] == n2["n_core"],
        "name_eq_sorted_core": n1["n_sorted"] == n2["n_sorted"] and n1["n_core"] != n2["n_core"],
        "m_name_all_upper": rb[i].isupper(), "m_name_all_lower": rb[i].islower(),
        "punct_differs": re.sub(r"[\w\s]", "", ra[i]) != re.sub(r"[\w\s]", "", rb[i]),
        "junk_symbol_in_m": bool(re.search(r"(<<|>>|--|\*\*|##|\[|\(\()", rb[i])),
        "legal_differs": n1["n_legal"] != n2["n_legal"],
        "legal_only_in_s1": bool(n1["n_legal"]) and not n2["n_legal"],
        "legal_only_in_m": bool(n2["n_legal"]) and not n1["n_legal"],
        "abbr_vs_full_legal": abbr,
        "token_order_changed": sorted(t1) == sorted(t2) and t1 != t2,
        "core_token_dropped": len(c1 - c2) > 0,
        "core_token_added": len(c2 - c1) > 0,
        "typo_token": typo,
        "char_edit_le2_name": 0 < Levenshtein.distance(n1["n_tok"], n2["n_tok"]) <= 2,
        "m_name_indic": n2["n_indic"], "m_name_url": n2["n_url"],
        "m_name_accent": any(ord(c) > 127 and c.isalpha() for c in rb[i]) and not n2["n_indic"],
        "amp_vs_and": ("&" in ra[i]) != ("&" in rb[i]) and (("and" in w1) or ("and" in w2)),
        "name_has_digit_s1": any(c.isdigit() for c in ra[i]),
        "leet_digit_in_m_name": n2["n_tok"] != n2["n_digit"],
        "name_core_jaccard_lt_0.5": (len(c1 & c2) / max(1, len(c1 | c2))) < 0.5,
        "name_no_shared_core_token": not (c1 & c2),
        "m_addr_blank": a2["a_blank"],
        "addr_exact_raw": xa[i] == xb[i],
        "addr_eq_casefold": xa[i].lower() == xb[i].lower(),
        "addr_eq_norm_tokens": at1 == at2,
        "addr_reordered": sorted(at1) == sorted(at2) and at1 != at2,
        "street_abbr_diff": stab,
        "s1_has_postal": bool(a1["a_postal"]), "m_has_postal": bool(a2["a_postal"]),
        "postal_missing_in_m": bool(a1["a_postal"]) and not a2["a_postal"] and not a2["a_blank"],
        "postal_conflict": bool(a1["a_postal"]) and bool(a2["a_postal"]) and not set(a1["a_postal"].split()) & set(a2["a_postal"].split()),
        "num_deleted": bool(set(n_1) - set(n_2)) and len(n_2) < len(n_1) and not a2["a_blank"],
        "num_substituted": bool(n_1) and bool(n_2) and set(n_1) != set(n_2) and len(n_1) == len(n_2),
        "num_added": bool(set(n_2) - set(n_1)) and len(n_2) > len(n_1),
        "first_num_equal": bool(a1["a_first_num"]) and a1["a_first_num"] == a2["a_first_num"],
        "any_num_shared": bool(set(n_1) & set(n_2)),
        "landmark_in_m_not_s1": bool(a2["a_landmark"]) and not a1["a_landmark"],
        "landmark_in_s1": bool(a1["a_landmark"]),
        "m_addr_subset_of_s1": bool(at2) and set(at2) <= set(at1) and at1 != at2,
        "m_addr_indic": a2["a_indic"],
        "null_token_in_m_addr": bool(re.search(r"\bnull\b", xb[i], re.I)),
        "state_tail_omitted": bool(tail) and not a2["a_blank"] and tail.group(1).strip().lower() not in xb[i].lower(),
        "addr_token_jaccard_lt_0.5": (len(set(at1) & set(at2)) / max(1, len(set(at1) | set(at2)))) < 0.5,
    }


print("computing features on", len(S), flush=True)
F = pd.DataFrame([feats(i) for i in range(len(S))])
F = pd.concat([S.reset_index(drop=True), F], axis=1)

L = ["# 02 — Noise Profile (quantified)", "",
     f"Generated by `code/business_entity_resolution/src/noise_profile.py`. Sample: up to {NPC:,} true pairs per (source × country), "
     "the same number of random same-country same-source negatives (`neg_random`), and one hard negative per sampled S1 "
     "(`neg_hard`: a non-matching pool record that shares the S1's rarest name token, same country and source). "
     "All rates are shares of pairs in the cell. `m` is the S2/S3 side.", "",
     "Every row was measured on training data only. France has no labels, so its section is descriptive only (see bottom).", ""]
cols = [c for c in F.columns if c not in ("s1", "m", "src", "country", "label")]
F["cell"] = F.label + "|" + F.src + "|" + F.country
tab = F.groupby("cell")[cols].mean().T * 100
order = [c for lab in ("pos", "neg_hard", "neg_random") for c in sorted(tab.columns) if c.startswith(lab + "|")]
tab = tab[order]
L.append("| feature (% of pairs) | " + " | ".join(c.replace("|", " ") for c in tab.columns) + " |")
L.append("|---" * (len(tab.columns) + 1) + "|")
for r, row in tab.iterrows():
    L.append(f"| {r} | " + " | ".join(f"{v:.2f}" for v in row.values) + " |")
L.append("")
L.append("Cell sizes: " + ", ".join(f"{k}: {v:,}" for k, v in F.cell.value_counts().sort_index().items()))
L.append("")

# discriminative power: pos vs hard-neg gap
gap = (F[F.label == "pos"][cols].mean() - F[F.label == "neg_hard"][cols].mean()) * 100
L += ["## Most discriminative binary signals (pos − hard-neg, percentage points)", "", "| signal | pos % | hard-neg % | gap |", "|---|---|---|---|"]
for c in gap.abs().sort_values(ascending=False).index[:20]:
    L.append(f"| {c} | {F[F.label=='pos'][c].mean()*100:.2f} | {F[F.label=='neg_hard'][c].mean()*100:.2f} | {gap[c]:+.2f} |")
L.append("")

# which name tokens get inserted / dropped in true pairs (noise vocabulary)
ins, dro = Counter(), Counter()
P = F.index[F.label == "pos"]
for i in P[:80000]:
    c1, c2 = set(na[i]["n_core"].split()), set(nb[i]["n_core"].split())
    ins.update(c2 - c1); dro.update(c1 - c2)
L += ["## Noise vocabulary: tokens most often *inserted* into the S2/S3 name of a true pair", "",
      ", ".join(f"`{k}` {v}" for k, v in ins.most_common(40)), "",
      "## Tokens most often *dropped* from the S1 name in a true pair", "", ", ".join(f"`{k}` {v}" for k, v in dro.most_common(40)), ""]

# France descriptive profile (test only, no labels)
t1 = load_source(data, "test", 1); t2 = load_source(data, "test", 2)
fr1 = t1[t1.country == "France"].sample(20000, random_state=1); fr2 = t2[t2.country == "France"].sample(20000, random_state=1)
def desc(df, lab):
    n = [name_forms(x) for x in df.business_name]; a = [addr_forms(x) for x in df.business_address]
    return {"set": lab, "accent_in_name %": 100 * np.mean([any(ord(c) > 127 and c.isalpha() for c in x) for x in df.business_name]),
            "all-upper addr %": 100 * np.mean([x.isupper() for x in df.business_address]),
            "5-digit postal %": 100 * np.mean([bool(x["a_postal"]) for x in a]),
            "legal suffix present %": 100 * np.mean([bool(x["n_legal"]) for x in n]),
            "blank addr %": 100 * np.mean([x["a_blank"] for x in a]),
            "top legal": Counter(t for x in n for t in x["n_legal"].split()).most_common(8)}
L += ["## France (test, unlabeled): descriptive only", ""]
for d in (desc(fr1, "test S1 France"), desc(fr2, "test S2 France")):
    L.append("- " + "; ".join(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}" for k, v in d.items()))
tokc = Counter(t for x in fr2.business_address for t in re.findall(r"[A-Za-zÀ-ÿ\.]+", x))
L += ["- most frequent test-S2 France address tokens: " + ", ".join(f"`{k}` {v}" for k, v in tokc.most_common(40)), ""]
open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("done")
