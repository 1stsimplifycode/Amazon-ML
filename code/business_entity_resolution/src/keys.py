"""Compound-key blocking (scalable, open-set).

Per record we pick the rarest few name tokens T, address numbers N and address
words A (rarity = document frequency inside the same country block) and emit
hashed keys:

  n:t        rare name token             b:t1|t2   name token pair
  x:t|n      name token x address number y:n|a     address number x address word
  z:t|a      name token x address word    m:n       address number
  a:a        address word                 q:name    whole space-free core name
  p:t4|n     4-char prefix of rarest name token x number (typo tolerance)

Only keys that occur in S1 AND in the pool with pool DF <= cap are kept, so the
sparse product stays near-linear. Score = sum of IDF over shared keys.
"""
from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.sparse as sp

_DIGITS = re.compile(r"\d+")
MASK = (1 << 62) - 1
_DF: dict = {}


def _init(df):
    global _DF
    _DF = df


def _rarest(tokens, k, prefix):
    u = list(dict.fromkeys(tokens))
    u.sort(key=lambda t: _DF.get(prefix + t, 1))
    return u[:k]


def record_keys(n_core: str, n_tok: str, a_tok: str, a_nums: str) -> list:
    nt = [t for t in n_core.split() if len(t) >= 2] or [t for t in n_tok.split() if len(t) >= 2]
    T = _rarest(nt, 3, "N")
    nums = [x.lstrip("0") or "0" for x in a_nums.split()]
    N = _rarest([x for x in nums if len(x) <= 7], 3, "D")
    atoks = a_tok.split()
    A = _rarest([t for t in atoks if len(t) >= 3 and not any(c.isdigit() for c in t)], 4, "A")
    k = ["n" + t for t in T]
    k += ["b" + a + "|" + b if a < b else "b" + b + "|" + a for i, a in enumerate(T) for b in T[i + 1:]]
    k += ["x" + t + "|" + n for t in T for n in N]
    k += ["y" + n + "|" + a for n in N for a in A[:2]]
    k += ["z" + t + "|" + a for t in T[:2] for a in A]
    k += ["m" + n for n in N] + ["a" + a for a in A]
    # number pairs (address-only identity when the name is a brand/typo/transliteration)
    k += ["w" + a + "|" + b if a < b else "w" + b + "|" + a for i, a in enumerate(N) for b in N[i + 1:]]
    # alphanumeric unit tokens such as 5c, 2a, g1, b3
    k += ["k" + t for t in atoks if any(c.isdigit() for c in t) and any(c.isalpha() for c in t)][:3]
    if nt:
        k.append("q" + "".join(sorted(nt)))
        k.append("q" + "".join(nt))
    # typo tolerance: 3-char prefixes of every rare name token x numbers / address words
    P3 = list(dict.fromkeys(t[:3] for t in T if len(t) >= 4))
    k += ["p" + p + "|" + n for p in P3 for n in N]
    k += ["r" + p + "|" + a for p in P3[:2] for a in A[:2]]
    return k


def _work(args):
    base, cols = args
    rows, hs = [], []
    for i, (c, t, a, n) in enumerate(zip(*cols)):
        for key in set(record_keys(c, t, a, n)):
            rows.append(base + i)
            hs.append(hash(key) & MASK)
    return np.asarray(rows, np.int32), np.asarray(hs, np.int64)


def token_df(frames, min_count: int = 1) -> dict:
    """DF of name tokens (prefix N), address numbers (D) and address words (A).

    Streams strings through a Counter (no exploded intermediate frames).
    min_count=2 drops singletons; lossless for _rarest, which defaults to 1, and
    it shrinks the dict that is pickled into every worker."""
    from collections import Counter
    cnt = Counter()
    for f in frames:
        for v in f["n_core"].tolist():
            cnt.update({"N" + t for t in v.split()})
        for v in f["a_tok"].tolist():
            cnt.update({"A" + t for t in v.split()})
        for v in f["a_nums"].tolist():
            cnt.update({"D" + (x.lstrip("0") or "0") for x in v.split()})
    return {t: v for t, v in cnt.items() if v >= min_count}


def hashed_keys(frame: pd.DataFrame, df: dict, procs: int = 5, chunk: int = 25000, only=None):
    """(row, key-hash) arrays. `only` (sorted unique hashes) drops every other key
    as each chunk arrives, so a large pool never materialises its full key list."""
    os.environ["PYTHONHASHSEED"] = "0"  # workers spawn with deterministic str hash
    cols = [frame[c].tolist() for c in ("n_core", "n_tok", "a_tok", "a_nums")]
    jobs = [(s, [c[s:s + chunk] for c in cols]) for s in range(0, len(frame), chunk)]
    del cols
    rs, hs = [], []
    with Pool(procs, initializer=_init, initargs=(df,)) as p:
        for r, h in p.imap(_work, jobs, chunksize=1):
            if only is not None:
                pos = np.minimum(np.searchsorted(only, h), len(only) - 1)
                keep = only[pos] == h
                r, h = r[keep], h[keep]
            rs.append(r); hs.append(h)
    return np.concatenate(rs), np.concatenate(hs)


def block(q_rows, q_h, nq, p_rows, p_h, npool, k: int, cap: int, chunk: int = 2000, threads: int = 6):
    """Return (q_idx, p_idx, score, n_shared, rank) of top-k pool records per query."""
    uq = np.unique(q_h)
    pos = np.minimum(np.searchsorted(uq, p_h), len(uq) - 1)
    keep = uq[pos] == p_h
    del pos
    p_rows, p_h = p_rows[keep], p_h[keep]
    uk, cnt = np.unique(p_h, return_counts=True)
    ok = cnt <= cap
    uk, cnt = uk[ok], cnt[ok]
    idf = np.log((npool + 1) / (cnt + 1)).astype(np.float32) + 1
    pc = np.searchsorted(uk, p_h); m = (pc < len(uk)) & (uk[np.minimum(pc, len(uk) - 1)] == p_h)
    P = sp.csr_matrix((np.ones(m.sum(), np.float32), (p_rows[m], pc[m])), shape=(npool, len(uk)))
    qc = np.searchsorted(uk, q_h); mq = (qc < len(uk)) & (uk[np.minimum(qc, len(uk) - 1)] == q_h)
    Q = sp.csr_matrix((idf[qc[mq]], (q_rows[mq], qc[mq])), shape=(nq, len(uk)))
    Qb = sp.csr_matrix((np.ones(mq.sum(), np.float32), (q_rows[mq], qc[mq])), shape=(nq, len(uk)))
    PT = P.T.tocsr()

    def one(s):
        C = (Q[s:s + chunk] @ PT).tocsr()
        Cn = (Qb[s:s + chunk] @ PT).tocsr()
        # identical sparsity pattern (idf > 0); sorted indices make entries align 1:1
        C.sort_indices(); Cn.sort_indices()
        assert np.array_equal(C.indices, Cn.indices)
        rows = np.repeat(np.arange(C.shape[0]), np.diff(C.indptr))
        order = np.lexsort((-C.data, rows))
        r, c, d, nsh = rows[order], C.indices[order], C.data[order], Cn.data[order]
        start = np.r_[0, np.cumsum(np.bincount(r, minlength=C.shape[0]))[:-1]]
        rank = np.arange(len(r), dtype=np.int64) - start[r]
        sel = rank < k
        return ((r[sel] + s).astype(np.int32), c[sel].astype(np.int32), d[sel],
                np.minimum(nsh[sel], 32767).astype(np.int16), rank[sel].astype(np.int16))

    # scipy sparse matmul and numpy sorts release the GIL, so threads parallelise
    with ThreadPoolExecutor(threads) as ex:
        parts = list(ex.map(one, range(0, nq, chunk)))
    return tuple(np.concatenate([p[j] for p in parts]) for j in range(5))
