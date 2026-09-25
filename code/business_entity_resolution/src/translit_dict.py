"""Learn romanised-Indic-token -> Latin-token dictionary from labelled pairs.

Leakage rule: fit ONLY on S1 entities of the dictionary fold (fold 0). Folds 1-4
are CV folds and never contribute. The same dictionary is used on test.

Alignment: a true pair (S1 Latin name, pool Indic name) with equal token counts is
aligned position-by-position. A mapping is kept when it has support >= min_support
distinct S1 entities and >= min_share of that romanised token's aligned occurrences.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

import pandas as pd

from normalize import LEGAL


def learn(s1: pd.DataFrame, pool: pd.DataFrame, pairs: pd.DataFrame, s1_ids: set,
          min_support: int = 3, min_share: float = 0.6) -> dict:
    p = pairs[pairs.s1.isin(s1_ids)]
    ind = pool[pool.n_indic.astype(bool)][["entity_id", "n_tok"]].rename(columns={"entity_id": "m", "n_tok": "m_tok"})
    p = p.merge(ind, on="m").merge(s1[["entity_id", "n_tok"]].rename(columns={"entity_id": "s1"}), on="s1")
    cnt: dict = defaultdict(Counter)
    for a, b in zip(p.m_tok.tolist(), p.n_tok.tolist()):
        ta, tb = a.split(), b.split()
        if len(ta) == len(tb):
            for x, y in zip(ta, tb):
                if x != y:
                    cnt[x][y] += 1
    out = {}
    for x, c in cnt.items():
        y, n = c.most_common(1)[0]
        if n >= min_support and n / sum(c.values()) >= min_share:
            out[x] = y
    return out


def core_of(tokens: list[str]) -> str:
    return " ".join(t for t in tokens if t not in LEGAL and t not in ("and", "the"))


def apply(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Rewrite n_tok / n_core for Indic-script records in place (returns df)."""
    m = df.n_indic.astype(bool).to_numpy()
    if not m.any():
        return df
    toks = [[mapping.get(t, t) for t in s.split()] for s in df.n_tok[m].tolist()]
    df.loc[m, "n_tok"] = [" ".join(t) for t in toks]
    df.loc[m, "n_core"] = [core_of(t) for t in toks]
    return df


def save(mapping: dict, path: str) -> None:
    json.dump(mapping, open(path, "w", encoding="utf-8"), ensure_ascii=False, sort_keys=True, indent=0)


def load(path: str) -> dict:
    return json.load(open(path, encoding="utf-8"))
