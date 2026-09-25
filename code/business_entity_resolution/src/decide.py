"""Entity-level decision layer: pair probabilities -> one predicted set per S1.

Methods (all operate on a frame with columns s1, m, p):
  threshold  : keep pairs with p >= t
  exclusive  : before thresholding, keep each pool record only for its best S1
               (GT forensics: every S2/S3 record belongs to at most one S1)
  expected_f : per S1 choose the prefix of candidates (sorted by p) maximising the
               plug-in expected F0.5 E[F_k] ~= 1.25*sum_{i<=k} p_i / (0.25*sum_all p_i + k);
               the empty set scores prod(1 - p_i) (probability that S1 is a singleton).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

B2 = 0.25


def exclusive(df: pd.DataFrame, margin: float = 0.0) -> pd.DataFrame:
    """Keep a pool record only for the S1 with its highest p (ties: all kept)."""
    best = df.groupby("m").p.transform("max")
    return df[df.p >= best - margin]


def by_threshold(df: pd.DataFrame, t: float) -> pd.DataFrame:
    return df[df.p >= t]


def expected_f(df: pd.DataFrame, floor: float = 0.0, miss_mass: float = 0.0) -> pd.DataFrame:
    """Vectorised per-S1 prefix selection maximising plug-in expected F0.5.

    miss_mass adds expected true matches outside the candidate set (blocking misses)
    to the denominator. floor drops candidates with p < floor before optimising.
    """
    d = df[df.p >= floor].sort_values(["s1", "p"], ascending=[True, False]).copy()
    if d.empty:
        return d
    g = d.groupby("s1", sort=False)
    d["k"] = g.cumcount() + 1
    d["cum"] = g.p.cumsum()
    tot = g.p.transform("sum") + miss_mass
    d["ef"] = (1 + B2) * d.cum / (B2 * tot + d.k)
    logq = np.log1p(-np.clip(d.p.to_numpy(), 0, 1 - 1e-7))
    d["p_empty"] = np.exp(pd.Series(logq, index=d.index).groupby(d.s1).transform("sum"))
    best_k = d.loc[d.groupby("s1", sort=False).ef.idxmax(), ["s1", "k", "ef", "p_empty"]]
    best_k = best_k[best_k.ef > best_k.p_empty]
    d = d.merge(best_k[["s1", "k"]].rename(columns={"k": "kmax"}), on="s1")
    return d[d.k <= d.kmax][["s1", "m", "p"]]


def to_sets(sel: pd.DataFrame, all_s1) -> dict:
    out = {s: set() for s in all_s1}
    for s, m in zip(sel.s1.tolist(), sel.m.tolist()):
        out[s].add(m)
    return out
