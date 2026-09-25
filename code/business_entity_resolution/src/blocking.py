"""Candidate generation: IDF-weighted sparse retrieval, several views, union.

Views (each gives top-K pool records per S1, same country only; country is an
open-set string key, so an unseen label like "France" is simply its own block):
  name   : word tokens of the normalised name (+ 4-char token prefixes for typos)
  addr   : word tokens of the normalised address
  both   : name and address vectors concatenated (hybrid)
  chr    : char 3-grams of the space-free core name (typos, concatenations, URLs, transliteration)
Tokens whose document frequency exceeds `df_cap` are dropped from retrieval only
(they carry little identity and make the sparse product quadratic).
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd
import scipy.sparse as sp


def fold_of(ids: pd.Series, k: int = 5) -> np.ndarray:
    return np.fromiter((zlib.crc32(s.encode()) % k for s in ids), dtype=np.int8, count=len(ids))


def name_terms(s: str) -> list[str]:
    toks = s.split()
    return toks + ["~" + t[:4] for t in toks if len(t) >= 6]


def char_terms(s: str) -> list[str]:
    s = "".join(s.split())
    return [s[i:i + 3] for i in range(max(1, len(s) - 2))] if s else []


def tfidf(q_docs, p_docs, analyzer, df_cap: int, min_df: int = 1):
    """Fit vocabulary on the union; return L2-normalised IDF matrices (Q, P)."""
    # analysis-only (B01, rejected); lazy so the pipeline never imports sklearn (BSD-3)
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(analyzer=analyzer, binary=True, dtype=np.float32, min_df=min_df)
    X = cv.fit_transform(list(q_docs) + list(p_docs)).tocsc()
    df = np.diff(X.indptr)
    keep = np.where((df <= df_cap) & (df >= 1))[0]
    X = X[:, keep]
    idf = np.log((X.shape[0] + 1) / (df[keep] + 1)).astype(np.float32) + 1
    X = X.tocsr() @ sp.diags(idf)
    X = sp.csr_matrix(X)
    norms = np.sqrt(X.multiply(X).sum(axis=1)).A1
    norms[norms == 0] = 1
    X = sp.diags(1 / norms).astype(np.float32) @ X
    X = sp.csr_matrix(X)
    nq = len(q_docs)
    return X[:nq], X[nq:]


def topk_sparse(Q: sp.csr_matrix, P: sp.csr_matrix, k: int, chunk: int = 4000):
    """Yield (q_row, p_row, score) arrays of the top-k scores per Q row."""
    PT = P.T.tocsr()
    out_q, out_p, out_s = [], [], []
    for s in range(0, Q.shape[0], chunk):
        C = (Q[s:s + chunk] @ PT).tocsr()
        C.eliminate_zeros()
        rows = np.repeat(np.arange(C.shape[0]), np.diff(C.indptr))
        order = np.lexsort((-C.data, rows))
        r, c, d = rows[order], C.indices[order], C.data[order]
        start = np.r_[0, np.cumsum(np.bincount(r, minlength=C.shape[0]))[:-1]]
        rank = np.arange(len(r)) - start[r]
        m = rank < k
        out_q.append(r[m] + s); out_p.append(c[m]); out_s.append(d[m])
    return np.concatenate(out_q), np.concatenate(out_p), np.concatenate(out_s)
