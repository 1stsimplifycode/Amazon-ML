"""Shared model pipeline pieces: frames, stage-1 fast features, context features."""
from __future__ import annotations

from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process

import keys
import translit_dict as td
from features import pair_features

FULL = ["entity_id", "country", "n_tok", "n_digit", "n_core", "n_legal", "n_url", "n_indic", "a_tok",
        "a_nums", "a_postal", "a_landmark", "a_blank", "a_indic", "name_raw"]


def _read(path):
    # Arrow-backed strings: ~5x less RAM than Python objects; features are
    # bit-identical (checked on a 20k-pair sample, all 82 columns)
    t = pq.read_table(path, columns=FULL)
    return t.to_pandas(types_mapper={pa.large_string(): pd.StringDtype("pyarrow"),
                                     pa.string(): pd.StringDtype("pyarrow")}.get)


def load_frames(work: str, split: str):
    mapping = td.load(f"{work}/translit_dict.json")
    s1 = _read(f"{work}/{split}_s1.parquet")
    pool = pd.concat([_read(f"{work}/{split}_s{k}.parquet") for k in (2, 3)], ignore_index=True)
    td.apply(pool, mapping); td.apply(s1, mapping)
    return s1, pool


def _arr(x):
    return x if isinstance(x, (pa.Array, pa.ChunkedArray)) else pa.array(x)


def _take(col: pd.Series, idx) -> list:
    """col[idx] as a Python list, gathered in Arrow first (never the whole column)."""
    return pc.take(_arr(col), pa.array(np.asarray(idx, np.int64))).to_pylist()


def index_pairs(cand, s1: pd.DataFrame, pool: pd.DataFrame):
    """Row positions of cand["s1"] in s1 and cand["m"] in pool (-1 if absent).

    `cand` may be a DataFrame or a pyarrow Table; lookups stay in Arrow, so 66M ID
    strings are never materialised as Python objects."""
    def pos(col, ids):
        r = pc.index_in(_arr(col), value_set=_arr(ids.entity_id))
        return r.fill_null(-1).to_numpy().astype(np.int64)
    return pos(cand["s1"], s1), pos(cand["m"], pool)


def _cp(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def group_stats(qi: np.ndarray, pi: np.ndarray, score: np.ndarray) -> dict:
    """Blocking-score context over the FULL candidate set (all folds; label-free)."""
    d = pd.DataFrame({"q": qi, "p": pi, "s": score})
    gq, gp = d.groupby("q", sort=False).s, d.groupby("p", sort=False).s
    return {"b_score_rel": (d.s / gq.transform("max")).to_numpy(np.float32),
            "b_ncand": gq.transform("size").to_numpy(np.float32),
            "b_m_nret": gp.transform("size").to_numpy(np.float32),
            "b_m_rel": (d.s / gp.transform("max")).to_numpy(np.float32)}


def stage1_features(cand: pd.DataFrame, s1: pd.DataFrame, pool: pd.DataFrame, ai, bi) -> pd.DataFrame:
    """Cheap features for blocked pairs: blocking stats + 5 fast string scores.

    `cand` must already hold the group_stats columns."""
    an, bn = _take(s1.n_core, ai), _take(pool.n_core, bi)
    aa, ba = _take(s1.a_tok, ai), _take(pool.a_tok, bi)
    F = pd.DataFrame({
        "b_score": cand.score.to_numpy(np.float32), "b_nshared": cand.nshared.to_numpy(np.float32),
        "b_rank": cand["rank"].to_numpy(np.float32), "b_rev": cand.rev.to_numpy(np.float32),
        "f_n_ratio": _cp(list(an), list(bn), fuzz.ratio),
        "f_n_tset": _cp(list(an), list(bn), fuzz.token_set_ratio),
        "f_n_nospace": _cp(["".join(x.split()) for x in an], ["".join(x.split()) for x in bn], fuzz.ratio),
        "f_a_tset": _cp(list(aa), list(ba), fuzz.token_set_ratio),
        "f_a_partial": _cp(list(aa), list(ba), fuzz.partial_ratio),
        "f_b_blank": pool.a_blank.to_numpy(np.float32)[bi],
        "f_b_indic": pool.n_indic.to_numpy(np.float32)[bi],
    })
    for c in ("b_score_rel", "b_ncand", "b_m_nret", "b_m_rel"):
        F[c] = cand[c].to_numpy(np.float32)
    return F


def context_features(df: pd.DataFrame, p: np.ndarray, prefix: str) -> pd.DataFrame:
    """Entity-level and pool-level competition features from a probability column.

    For pair (s, m): rank of p within s, p minus best p of s, second-best p of s,
    count of strong candidates of s, best p of m among OTHER S1 (exclusivity), and
    how many S1 compete for m with p > 0.5.
    """
    d = pd.DataFrame({"s1": df.s1.to_numpy(), "m": df.m.to_numpy(), "p": p})
    gs = d.groupby("s1", sort=False).p
    out = pd.DataFrame(index=d.index)
    out[prefix + "rank_s"] = gs.rank(ascending=False, method="first").to_numpy(np.float32)
    mx = gs.transform("max")
    out[prefix + "gap_s"] = (d.p - mx).to_numpy(np.float32)
    out[prefix + "max_s"] = mx.to_numpy(np.float32)
    out[prefix + "sum_s"] = gs.transform("sum").to_numpy(np.float32)
    out[prefix + "n_strong_s"] = (d.p > 0.5).groupby(d.s1, sort=False).transform("sum").to_numpy(np.float32)
    # best competitor for m: top-2 per m, then other = top1 if I am not top1 else top2
    gm = d.groupby("m", sort=False).p
    top1 = gm.transform("max")
    srt = d.sort_values(["m", "p"], ascending=[True, False])
    second = srt.groupby("m", sort=False).p.nth(1)
    sec_map = dict(zip(srt.loc[second.index, "m"].tolist(), second.tolist()))
    second_v = d.m.map(sec_map).fillna(0).to_numpy(np.float32)
    is_top = d.p.to_numpy() >= top1.to_numpy()
    other = np.where(is_top, second_v, top1.to_numpy())
    out[prefix + "m_other_best"] = other.astype(np.float32)
    out[prefix + "m_margin"] = (d.p.to_numpy() - other).astype(np.float32)
    out[prefix + "m_n_strong"] = (d.p > 0.5).groupby(d.m, sort=False).transform("sum").to_numpy(np.float32)
    return out


def stage2_features(cand, s1, pool, ai, bi, df_name, df_addr, n_docs, chunk=500000, procs=6):
    parts = []
    with Pool(procs) as workers:
        for s in range(0, len(cand), chunk):
            a = s1.iloc[ai[s:s + chunk]].reset_index(drop=True)
            b = pool.iloc[bi[s:s + chunk]].reset_index(drop=True)
            parts.append(pair_features(a, b, df_name, df_addr, n_docs, pool=workers))
    return pd.concat(parts, ignore_index=True)


def name_addr_df(s1, pool):
    """Label-free token DF (name 'N', address 'A') over all records of the split."""
    return keys.token_df([s1[["n_core", "a_tok", "a_nums"]], pool[["n_core", "a_tok", "a_nums"]]])
