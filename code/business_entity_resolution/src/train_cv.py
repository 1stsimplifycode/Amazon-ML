"""Entity-level CV of the full cascade on train.

usage: python train_cv.py <work_dir> <dataset_dir> <out_json> [n_keep]

Folds (crc32(S1 id) % 5):
  fold 0    : translit dictionary + stage-1 model training (never scored)
  folds 1-4 : stage-2 CV. Split A trains on {1,2} and scores {3,4}; split B the reverse.
Everything that touches labels is fit on training folds only. Blocking and DF
statistics are label-free. Pairs are held as integer row codes (qi into S1, pi into
the S2+S3 pool) to fit 66M candidates in RAM.
"""
import gc
import os
import zlib
import json
import sys
import time

import numpy as np
import psutil
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb

import decide
import metric
from blocking import fold_of
from io_utils import load_gt_pairs
from pipeline import (context_features, group_stats, index_pairs, load_frames, name_addr_df,
                      stage1_features, stage2_features)

T0 = time.time()
BIG = np.int64(1 << 32)


def log(*a):
    rss = psutil.Process().memory_info().rss / 1e9
    print(f"[{time.time()-T0:6.0f}s {rss:4.1f}GB]", *a, flush=True)


def xgb_fit(X, y, rounds=400, depth=8, seed=0):
    params = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", max_depth=depth,
                  eta=0.1, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, nthread=8, seed=seed,
                  max_bin=128)
    return xgb.train(params, xgb.DMatrix(X, label=y), rounds)


def xgb_pred(m, X):
    return m.predict(xgb.DMatrix(X))


def s1_feats(cand, idx, s1, pool, chunk=2_000_000):
    """stage1_features over rows idx of cand, in chunks (bounds the Python-string lists)."""
    qi, pi = cand.qi.to_numpy(), cand.pi.to_numpy()
    return pd.concat([stage1_features(cand.iloc[ii], s1, pool, qi[ii], pi[ii])
                      for ii in (idx[s:s + chunk] for s in range(0, len(idx), chunk))], ignore_index=True)


def evaluate(sel_q, sel_p, fold_q_idx, tq, tp):
    """Entity-level report over every S1 of the fold (zero-candidate S1 included).

    Works on integer row indices (qi into S1, pi into the pool), never ID strings."""
    ids = fold_q_idx.tolist()
    pred = {s: set() for s in ids}
    for q, p in zip(np.asarray(sel_q).tolist(), np.asarray(sel_p).tolist()):
        pred[q].add(p)
    tf = {s: set() for s in ids}
    m = np.isin(tq, fold_q_idx)
    for q, p in zip(tq[m].tolist(), tp[m].tolist()):
        tf[q].add(p)
    r = metric.report(pred, tf, ids)
    r["false_merges"] = sum(len(pred[s] - tf[s]) for s in ids)
    r["missed_links"] = sum(len(tf[s] - pred[s]) for s in ids)
    r["singleton_false_merge_entities"] = sum(1 for s in ids if not tf[s] and pred[s])
    return r


def main(work, data, out_json, n_keep=10):
    res = {}
    s1, pool = load_frames(work, "train")
    fold_s1 = fold_of(s1.entity_id)
    # SMOKE=N: keep 1/N of S1 entities (fold-independent hash) for a fast end-to-end check
    smoke = int(os.environ.get("SMOKE", "0"))
    use = (np.fromiter((zlib.crc32(s.encode()) // 5 % smoke == 0 for s in s1.entity_id.tolist()), bool, len(s1))
           if smoke else np.ones(len(s1), bool))
    _, pairs = load_gt_pairs(data)
    tq, tp = index_pairs(pairs, s1, pool)
    assert (tq >= 0).all() and (tp >= 0).all(), "ground-truth ID missing from frames"
    tq, tp = tq[use[tq]], tp[use[tq]]
    del pairs; gc.collect()
    log("frames", len(s1), len(pool), "true pairs", len(tq))
    tbl = pq.read_table(f"{work}/cand_train.parquet")
    qi, pi = index_pairs(tbl, s1, pool)
    assert (qi >= 0).all() and (pi >= 0).all(), "candidate ID missing from frames"
    keep_rows = use[qi]
    qi, pi = qi[keep_rows], pi[keep_rows]
    cand = tbl.drop(["s1", "m"]).filter(pa.array(keep_rows)).to_pandas()
    del tbl; gc.collect()
    pa.default_memory_pool().release_unused()
    cand["qi"] = qi.astype(np.int32); cand["pi"] = pi.astype(np.int32)
    tcode = np.sort(tq.astype(np.int64) * BIG + tp)
    code = cand.qi.to_numpy(np.int64) * BIG + cand.pi.to_numpy(np.int64)
    pos = np.minimum(np.searchsorted(tcode, code), len(tcode) - 1)
    cand["y"] = tcode[pos] == code
    cand["fold"] = fold_s1[cand.qi.to_numpy()]
    del code, pos; gc.collect()
    log("cand", len(cand), "positives", int(cand.y.sum()))
    tp_f = pd.Series(fold_s1[tq]).value_counts()
    # ---------------- blocking metrics per fold
    blk = {}
    n_all_pool = len(pool)
    for f in range(5):
        c = cand[cand.fold == f]
        nper = pd.Series(np.bincount(c.qi.to_numpy(), minlength=len(s1))[(fold_s1 == f) & use])
        blk[f] = dict(recall=float(c.y.sum() / tp_f[f]), mean=float(nper.mean()), median=float(nper.median()),
                      p90=float(nper.quantile(.9)), p95=float(nper.quantile(.95)), p99=float(nper.quantile(.99)),
                      max=int(nper.max()), zero_cand_s1=int((nper == 0).sum()),
                      rev_only_recall_gain=float(c[c.rev == 1].y.sum() / tp_f[f]),
                      reduction_ratio=float(1 - len(c) / (len(nper) * n_all_pool)))
    res["blocking"] = blk
    log("blocking", json.dumps(blk))
    # ---------------- stage 1: group stats over all folds, model on fold 0
    for k, v in group_stats(cand.qi.to_numpy(), cand.pi.to_numpy(), cand.score.to_numpy()).items():
        cand[k] = v
    p1 = np.zeros(len(cand), np.float32)
    f0 = np.flatnonzero(cand.fold.to_numpy() == 0)
    X0 = s1_feats(cand, f0, s1, pool)
    m1 = xgb_fit(X0, cand.y.to_numpy()[f0], rounds=200, depth=7)
    res["stage1_importance"] = sorted(m1.get_score(importance_type="gain").items(), key=lambda x: -x[1])
    del X0; gc.collect()
    for f in range(5):
        idx = np.flatnonzero(cand.fold.to_numpy() == f)
        p1[idx] = xgb_pred(m1, s1_feats(cand, idx, s1, pool))
        log("stage1 predicted fold", f)
    cand["p1"] = p1
    cand["r1"] = cand.groupby("qi", sort=False).p1.rank(ascending=False, method="first").astype(np.float32)
    keep = ((cand.r1 <= n_keep) & (cand.p1 >= 0.001)).to_numpy()
    st1 = {}
    for f in range(5):
        mf = (cand.fold == f).to_numpy()
        st1[f] = dict(recall_after_prune=float(cand.y[mf & keep].sum() / tp_f[f]),
                      pairs_per_s1=float((mf & keep).sum() / ((fold_s1 == f) & use).sum()))
    res["stage1"] = st1
    log("stage1", json.dumps(st1))
    # context features on the pruned set (competition across ALL folds, label-free)
    wk = cand[keep].reset_index(drop=True)
    del cand, p1; gc.collect()
    C1 = context_features(wk.rename(columns={"qi": "s1", "pi": "m"}), wk.p1.to_numpy(), "c1_")
    sel = (wk.fold >= 1).to_numpy()
    wk, C1 = wk[sel].reset_index(drop=True), C1[sel].reset_index(drop=True)
    df_tok = name_addr_df(s1, pool)
    F1 = s1_feats(wk, np.arange(len(wk)), s1, pool)
    F2 = stage2_features(wk, s1, pool, wk.qi.to_numpy(), wk.pi.to_numpy(), df_tok, df_tok, len(s1) + len(pool))
    del df_tok; gc.collect()
    X = pd.concat([F1, C1, F2], axis=1)
    X["p1"] = wk.p1.to_numpy(np.float32)
    del F1, F2; gc.collect()
    log("stage2 X", X.shape)
    y2 = wk.y.to_numpy()
    splits = {"A": ({1, 2}, {3, 4}), "B": ({3, 4}, {1, 2})}
    oof = np.zeros(len(X), np.float32)
    imp = {}
    for name, (trf, tef) in splits.items():
        trm = wk.fold.isin(trf).to_numpy(); tem = wk.fold.isin(tef).to_numpy()
        m2 = xgb_fit(X[trm], y2[trm], rounds=500, depth=8, seed=1)
        oof[tem] = xgb_pred(m2, X[tem])
        imp[name] = sorted(m2.get_score(importance_type="gain").items(), key=lambda x: -x[1])[:40]
        log("split", name, "trained")
    wk["p"] = oof
    res["importance"] = imp
    wk["exact"] = (X.n_tok_eq.to_numpy() * X.a_eq.to_numpy()) > 0
    wk["fuzzy"] = (X.n_tset.to_numpy() >= 90) & (X.a_tset.to_numpy() >= 80)
    wk.drop(columns=[c for c in wk.columns if c.startswith("b_")]).to_parquet(f"{work}/oof_train.parquet", compression="zstd", index=False)
    X.iloc[:0].to_parquet(f"{work}/feature_schema.parquet")
    # ---------------- entity-level evaluation per fold
    ev = {}
    for f in (1, 2, 3, 4):
        fq = np.flatnonzero((fold_s1 == f) & use)
        d = wk[wk.fold == f].rename(columns={"qi": "s1", "pi": "m"})
        r = {}
        e = d[d.exact]; r["exact_rule"] = evaluate(e.s1.to_numpy(), e.m.to_numpy(), fq, tq, tp)
        e = d[d.fuzzy]; r["fuzzy_rule"] = evaluate(e.s1.to_numpy(), e.m.to_numpy(), fq, tq, tp)
        base = d[["s1", "m", "p"]]
        for t in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9):
            e = decide.by_threshold(base, t); r[f"thr_{t}"] = evaluate(e.s1.to_numpy(), e.m.to_numpy(), fq, tq, tp)
            e = decide.by_threshold(decide.exclusive(base), t); r[f"excl_thr_{t}"] = evaluate(e.s1.to_numpy(), e.m.to_numpy(), fq, tq, tp)
        e = decide.expected_f(base); r["expected_f"] = evaluate(e.s1.to_numpy(), e.m.to_numpy(), fq, tq, tp)
        e = decide.expected_f(decide.exclusive(base)); r["excl_expected_f"] = evaluate(e.s1.to_numpy(), e.m.to_numpy(), fq, tq, tp)
        ev[f] = r
        log("fold", f, {k: round(v["macro_f05"], 4) for k, v in r.items()})
    res["eval"] = ev
    json.dump(res, open(out_json, "w"), indent=1, default=float)
    log("done")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 10)
