"""Pair feature bank. Input: two aligned frames (S1 side `a`, pool side `b`).

String similarities use rapidfuzz.process.cpdist (vectorised, multithreaded).
Set/number features are computed in Python per pair (they are cheap).
All outputs are float32; NaN never appears (missing -> explicit flag + 0).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Jaro, Levenshtein, Indel, Prefix, Postfix

from normalize import LEGAL


def _cp(a, b, scorer, **kw):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32, **kw)


def _nospace(xs):
    return ["".join(x.split()) for x in xs]


COLS = ("n_jacc", "n_dice", "n_idf_jacc", "n_shared_maxidf", "n_a_only_idf", "n_b_only_idf", "n_ntok_a", "n_ntok_b",
        "n_legal_eq", "n_legal_conflict", "n_legal_missing", "n_digit_eq", "n_digit_conflict", "n_min_df",
        "a_jacc", "a_idf_jacc", "a_contain_b_in_a", "a_contain_a_in_b", "a_shared_maxidf",
        "num_jacc", "num_first_eq", "num_any_shared", "num_conflict", "num_a_missing", "num_b_extra",
        "num_rarest_shared", "num_a_count", "num_b_count", "postal_eq", "postal_conflict", "postal_partial",
        "postal_one_missing", "lm_overlap", "lm_b_only")


def _procs(pool) -> int:
    return getattr(pool, "_processes", 1) if pool is not None else 1


def _loop_cols(job) -> dict:
    """Per-pair set / IDF / number features for one row chunk (runs in a worker)."""
    (an, bn, aa, ba, an_l, bn_l, ad, bd, anum, bnum, apo, bpo, alm, blm), df_name, df_addr, N = job
    L = np.log
    def idf_n(t):
        return L((N + 1) / (df_name.get("N" + t, 0) + 1)) + 1
    def idf_a(t):
        return L((N + 1) / (df_addr.get("A" + t, 0) + 1)) + 1
    cols = {k: np.zeros(len(an), np.float32) for k in COLS}
    for i in range(len(an)):
        sa, sb = set(an[i].split()), set(bn[i].split())
        inter, uni = sa & sb, sa | sb
        if uni:
            cols["n_jacc"][i] = len(inter) / len(uni)
            cols["n_dice"][i] = 2 * len(inter) / (len(sa) + len(sb))
            wi = sum(idf_n(t) for t in inter); wu = sum(idf_n(t) for t in uni)
            cols["n_idf_jacc"][i] = wi / wu
            cols["n_shared_maxidf"][i] = max((idf_n(t) for t in inter), default=0)
            cols["n_a_only_idf"][i] = sum(idf_n(t) for t in sa - sb)
            cols["n_b_only_idf"][i] = sum(idf_n(t) for t in sb - sa)
        cols["n_min_df"][i] = min((df_name.get("N" + t, 0) for t in sa), default=0)
        cols["n_ntok_a"][i], cols["n_ntok_b"][i] = len(sa), len(sb)
        la, lb = set(an_l[i].split()), set(bn_l[i].split())
        cols["n_legal_eq"][i] = float(la == lb and bool(la))
        cols["n_legal_conflict"][i] = float(bool(la) and bool(lb) and not (la & lb))
        cols["n_legal_missing"][i] = float(bool(la) != bool(lb))
        da = {t for t in ad[i].split() if t.isdigit()}; db = {t for t in bd[i].split() if t.isdigit()}
        cols["n_digit_eq"][i] = float(bool(da) and da == db)
        cols["n_digit_conflict"][i] = float(bool(da) and bool(db) and not (da & db))
        # address tokens
        ta, tb = set(aa[i].split()), set(ba[i].split())
        if ta and tb:
            ai, au = ta & tb, ta | tb
            cols["a_jacc"][i] = len(ai) / len(au)
            cols["a_idf_jacc"][i] = sum(idf_a(t) for t in ai) / sum(idf_a(t) for t in au)
            cols["a_contain_b_in_a"][i] = len(ai) / len(tb)
            cols["a_contain_a_in_b"][i] = len(ai) / len(ta)
            cols["a_shared_maxidf"][i] = max((idf_a(t) for t in ai), default=0)
        na_ = [x.lstrip("0") or "0" for x in anum[i].split()]; nb_ = [x.lstrip("0") or "0" for x in bnum[i].split()]
        sna, snb = set(na_), set(nb_)
        cols["num_a_count"][i], cols["num_b_count"][i] = len(sna), len(snb)
        if sna and snb:
            cols["num_jacc"][i] = len(sna & snb) / len(sna | snb)
            cols["num_first_eq"][i] = float(na_[0] == nb_[0])
            cols["num_any_shared"][i] = float(bool(sna & snb))
            cols["num_conflict"][i] = float(not (sna & snb))
            r = max(sna, key=lambda x: (len(x), x))  # deterministic tie-break (set order is hash-seeded)
            cols["num_rarest_shared"][i] = float(r in snb)
        cols["num_a_missing"][i] = len(sna - snb)
        cols["num_b_extra"][i] = len(snb - sna)
        pa, pb = set(apo[i].split()), set(bpo[i].split())
        if pa and pb:
            cols["postal_eq"][i] = float(bool(pa & pb))
            cols["postal_conflict"][i] = float(not (pa & pb))
            cols["postal_partial"][i] = float(any(x[:3] == y[:3] for x in pa for y in pb))
        cols["postal_one_missing"][i] = float(bool(pa) != bool(pb))
        la_, lb_ = set(alm[i].split()), set(blm[i].split())
        cols["lm_overlap"][i] = float(bool(la_ & lb_))
        cols["lm_b_only"][i] = float(bool(lb_) and not la_)
    return cols


def pair_features(a: pd.DataFrame, b: pd.DataFrame, df_name: dict, df_addr: dict, n_docs: int,
                  pool=None) -> pd.DataFrame:
    F = {}
    an, bn = a.n_core.tolist(), b.n_core.tolist()
    at, bt = a.n_tok.tolist(), b.n_tok.tolist()
    aa, ba = a.a_tok.tolist(), b.a_tok.tolist()
    ar, br = a.name_raw.tolist(), b.name_raw.tolist()
    # ---------------- name: string similarities
    F["n_raw_eq"] = np.array([x == y for x, y in zip(ar, br)], np.float32)
    F["n_ci_eq"] = np.array([x.lower() == y.lower() for x, y in zip(ar, br)], np.float32)
    F["n_tok_eq"] = np.array([x == y for x, y in zip(at, bt)], np.float32)
    F["n_core_eq"] = np.array([x == y and x != "" for x, y in zip(an, bn)], np.float32)
    F["n_ratio"] = _cp(an, bn, fuzz.ratio)
    F["n_tsort"] = _cp(an, bn, fuzz.token_sort_ratio)
    F["n_tset"] = _cp(an, bn, fuzz.token_set_ratio)
    F["n_partial"] = _cp(an, bn, fuzz.partial_ratio)
    F["n_lev"] = _cp(an, bn, Levenshtein.normalized_similarity)
    F["n_jw"] = _cp(an, bn, JaroWinkler.similarity)
    F["n_jaro"] = _cp(an, bn, Jaro.similarity)
    F["n_tok_ratio"] = _cp(at, bt, fuzz.token_set_ratio)
    ans, bns = _nospace(an), _nospace(bn)
    F["n_nospace_ratio"] = _cp(ans, bns, fuzz.ratio)
    F["n_nospace_partial"] = _cp(ans, bns, fuzz.partial_ratio)
    F["n_prefix"] = _cp(ans, bns, Prefix.similarity)
    F["n_suffix"] = _cp(ans, bns, Postfix.similarity)
    F["n_len_a"] = np.array([len(x) for x in ans], np.float32)
    F["n_len_b"] = np.array([len(x) for x in bns], np.float32)
    F["n_raw_ratio"] = _cp([x.lower() for x in ar], [x.lower() for x in br], fuzz.ratio)
    # ---------------- address: string similarities
    F["a_ratio"] = _cp(aa, ba, fuzz.ratio)
    F["a_tsort"] = _cp(aa, ba, fuzz.token_sort_ratio)
    F["a_tset"] = _cp(aa, ba, fuzz.token_set_ratio)
    F["a_partial"] = _cp(aa, ba, fuzz.partial_ratio)
    F["a_eq"] = np.array([x == y and x != "" for x, y in zip(aa, ba)], np.float32)
    aal, bal = [" ".join(t for t in x.split() if not any(c.isdigit() for c in t)) for x in aa], \
               [" ".join(t for t in x.split() if not any(c.isdigit() for c in t)) for x in ba]
    F["a_alpha_tset"] = _cp(aal, bal, fuzz.token_set_ratio)
    F["a_blank_b"] = b.a_blank.to_numpy(np.float32)
    F["b_indic_name"] = b.n_indic.to_numpy(np.float32)
    F["b_url"] = b.n_url.to_numpy(np.float32)
    F["b_indic_addr"] = b.a_indic.to_numpy(np.float32)

    # ---------------- set / idf / number features (python loop, parallel over row chunks)
    lists = (an, bn, aa, ba, a.n_legal.tolist(), b.n_legal.tolist(), a.n_digit.tolist(), b.n_digit.tolist(),
             a.a_nums.tolist(), b.a_nums.tolist(), a.a_postal.tolist(), b.a_postal.tolist(),
             a.a_landmark.tolist(), b.a_landmark.tolist())
    step = max(1, -(-len(a) // (4 * _procs(pool))))
    jobs = []
    for s in range(0, len(a), step):
        part = tuple(x[s:s + step] for x in lists)
        ntok = {t for x in part[0] + part[1] for t in x.split()}
        atok = {t for x in part[2] + part[3] for t in x.split()}
        # only the DF entries this chunk touches travel to the worker
        jobs.append((part, {"N" + t: df_name.get("N" + t, 0) for t in ntok},
                     {"A" + t: df_addr.get("A" + t, 0) for t in atok}, float(n_docs)))
    parts = pool.map(_loop_cols, jobs) if pool is not None else [_loop_cols(j) for j in jobs]
    cols = {k: np.concatenate([p[k] for p in parts]) if parts else np.zeros(0, np.float32) for k in COLS}
    F.update(cols)
    # ---------------- explicit contradiction features
    F["x_name_eq_num_conflict"] = F["n_core_eq"] * F["num_conflict"]
    F["x_addr_high_name_low"] = ((F["a_tset"] >= 90) & (F["n_tset"] < 50)).astype(np.float32)
    F["x_generic_name_weak_addr"] = ((F["n_min_df"] > 200) & (F["a_tset"] < 60)).astype(np.float32)
    F["x_rare_name_strong_addr"] = ((F["n_shared_maxidf"] > 8) & (F["a_tset"] >= 80)).astype(np.float32)
    return pd.DataFrame(F)
