"""Stage 2: candidate generation for a whole split.

usage: python build_candidates.py <work_dir> <split> [K] [cap]
Writes <work_dir>/cand_<split>_<country>.parquet per country and concatenates them
into <work_dir>/cand_<split>.parquet with columns s1, m, score, nshared, rank, rev.
rev=1 marks pairs found only by the reverse pass (blank-address pool record ->
top-3 S1 by name keys). Countries come from the data (open set) and each is run in
its own subprocess so peak memory is one country.
"""
import gc
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import keys
import translit_dict as td

COLS = ["entity_id", "country", "n_tok", "n_core", "a_tok", "a_nums", "n_indic", "a_blank"]


def load_country(work, split, k, c):
    return pq.read_table(f"{work}/{split}_s{k}.parquet", columns=COLS, filters=[("country", "=", c)]).to_pandas()


def run_country(work, split, c, K=30, cap=1000, K_rev=3):
    t = time.time()
    mapping = td.load(f"{work}/translit_dict.json")
    q = load_country(work, split, 1, c)
    p = pd.concat([load_country(work, split, k, c) for k in (2, 3)], ignore_index=True)
    td.apply(q, mapping); td.apply(p, mapping)
    q_ids, p_ids = q.entity_id.to_numpy(object), p.entity_id.to_numpy(object)
    blank = np.flatnonzero(p.a_blank.to_numpy(bool))
    nq, npool = len(q), len(p)
    if nq == 0 or npool == 0:
        pd.DataFrame(columns=["s1", "m", "score", "nshared", "rank", "rev"]).to_parquet(f"{work}/cand_{split}_{c}.parquet")
        return
    df = keys.token_df([q, p], min_count=2)
    qr, qh = keys.hashed_keys(q, df)
    del q; gc.collect()
    pr, ph = keys.hashed_keys(p, df, only=np.unique(qh))
    del p, df; gc.collect()
    print(c, "keys", len(qh), len(ph), f"{time.time()-t:.0f}s", flush=True)
    # reverse-pass keys first, so the big pool arrays can be dropped inside block()
    sel = np.isin(pr, blank)
    remap = np.full(npool, -1, np.int64); remap[blank] = np.arange(len(blank))
    br_rows, br_h = remap[pr[sel]].astype(np.int32), ph[sel]
    del sel, remap
    qi, pi, sc, ns, rk = keys.block(qr, qh, nq, pr, ph, npool, K, cap)
    del pr, ph; gc.collect()
    fwd = pd.DataFrame({"qi": qi, "pi": pi, "score": sc, "nshared": ns, "rank": rk})
    bi, s1i, bs, bn, _ = keys.block(br_rows, br_h, len(blank), qr, qh, nq, K_rev, cap)
    rev = pd.DataFrame({"qi": s1i, "pi": blank[bi], "score_rev": bs, "nshared_rev": bn})
    m = fwd.merge(rev, on=["qi", "pi"], how="outer")
    m["rev"] = m.score.isna().astype(np.int8)
    m["score"] = m.score.fillna(m.score_rev).astype(np.float32)
    m["nshared"] = m.nshared.fillna(m.nshared_rev).astype(np.int16)
    m["rank"] = m["rank"].fillna(K).astype(np.int16)
    m["s1"] = pd.array(q_ids[m.qi.to_numpy()], dtype="string[pyarrow]")
    m["m"] = pd.array(p_ids[m.pi.to_numpy()], dtype="string[pyarrow]")
    m = m[["s1", "m", "score", "nshared", "rank", "rev"]]
    m.to_parquet(f"{work}/cand_{split}_{c}.parquet", compression="zstd", index=False)
    print(c, "S1", nq, "pool", npool, "pairs", len(m), "rev-only", int(m.rev.sum()), f"{time.time()-t:.0f}s", flush=True)


def concat(work, split):
    """Stream per-country files into cand_<split>.parquet (the parent never holds them;
    it must stay small while the next country's subprocess peaks near 11 GB)."""
    w = None
    for c in countries(work, split):
        t = pq.read_table(f"{work}/cand_{split}_{c}.parquet")
        w = w or pq.ParquetWriter(f"{work}/cand_{split}.parquet", t.schema, compression="zstd")
        w.write_table(t)
        del t
    w.close()


def countries(work, split):
    cs = set()
    for k in (1, 2, 3):
        cs |= set(pq.read_table(f"{work}/{split}_s{k}.parquet", columns=["country"]).column(0).unique().to_pylist())
    return sorted(cs)


if __name__ == "__main__":
    work, split = sys.argv[1], sys.argv[2]
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    cap = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
    if len(sys.argv) > 5:
        run_country(work, split, sys.argv[5], K, cap)
        sys.exit(0)
    for c in countries(work, split):
        r = subprocess.run([sys.executable, "-u", __file__, work, split, str(K), str(cap), c])
        if r.returncode != 0:
            raise SystemExit(f"country {c} failed")
    concat(work, split)
    for c in countries(work, split):
        os.remove(f"{work}/cand_{split}_{c}.parquet")
    print("done", flush=True)
