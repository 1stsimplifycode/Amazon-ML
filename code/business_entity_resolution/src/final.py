"""Final cascade: fit on train, predict test, write matching_results.tsv.

usage:
  python final.py fit     <work_dir> <dataset_dir> [stage2_frac=1.0] [n_keep=10]
  python final.py predict <work_dir> <out_dir> [threshold=0.7] [n_keep=10]

Mirrors train_cv.py exactly (same features, same pruning, same models):
  stage 1 : XGBoost on fold 0 of train, cheap features on every blocked pair
  prune   : top n_keep per S1 by p1 and p1 >= 0.001
  context : competition features over the FULL pruned set of the split (label-free)
  stage 2 : XGBoost on folds 1-4 (optionally a stage2_frac hash subsample of S1)
  decide  : keep pairs with p >= threshold
Models go to <work_dir>/model_s{1,2}.json with the stage-2 column order.
"""
import gc
import json
import os
import sys
import time
import zlib

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb

import decide
from blocking import fold_of
from io_utils import load_gt_pairs
from pipeline import context_features, group_stats, index_pairs, load_frames, name_addr_df, stage2_features
from train_cv import BIG, s1_feats, xgb_fit, xgb_pred

T0 = time.time()
P1_MIN = 0.001


def log(*a):
    rss = psutil.Process().memory_info().rss / 1e9
    print(f"[{time.time()-T0:6.0f}s {rss:4.1f}GB]", *a, flush=True)


def load_cand(work, split, s1, pool, use=None):
    tbl = pq.read_table(f"{work}/cand_{split}.parquet")
    qi, pi = index_pairs(tbl, s1, pool)
    assert (qi >= 0).all() and (pi >= 0).all(), "candidate ID missing from frames"
    cand = tbl.drop(["s1", "m"]).to_pandas()
    del tbl; gc.collect()
    pa.default_memory_pool().release_unused()
    cand["qi"] = qi.astype(np.int32); cand["pi"] = pi.astype(np.int32)
    for k, v in group_stats(cand.qi.to_numpy(), cand.pi.to_numpy(), cand.score.to_numpy()).items():
        cand[k] = v
    return cand


def score_s1(m1, cand, s1, pool, chunk=4_000_000):
    """Stage-1 probabilities for every row of cand, features built chunk by chunk."""
    p1 = np.empty(len(cand), np.float32)
    for s in range(0, len(cand), chunk):
        ii = np.arange(s, min(s + chunk, len(cand)))
        p1[ii] = xgb_pred(m1, s1_feats(cand, ii, s1, pool))
    return p1


def prune_and_context(cand, n_keep):
    cand["r1"] = cand.groupby("qi", sort=False).p1.rank(ascending=False, method="first").astype(np.float32)
    wk = cand[((cand.r1 <= n_keep) & (cand.p1 >= P1_MIN)).to_numpy()].reset_index(drop=True)
    C1 = context_features(wk.rename(columns={"qi": "s1", "pi": "m"}), wk.p1.to_numpy(), "c1_")
    return wk, C1


def stage2_X(wk, C1, s1, pool):
    df_tok = name_addr_df(s1, pool)
    F1 = s1_feats(wk, np.arange(len(wk)), s1, pool)
    F2 = stage2_features(wk, s1, pool, wk.qi.to_numpy(), wk.pi.to_numpy(), df_tok, df_tok, len(s1) + len(pool))
    X = pd.concat([F1, C1.reset_index(drop=True), F2], axis=1)
    X["p1"] = wk.p1.to_numpy(np.float32)
    return X


def fit(work, data, frac=1.0, n_keep=10):
    s1, pool = load_frames(work, "train")
    fold_s1 = fold_of(s1.entity_id)
    _, pairs = load_gt_pairs(data)
    tq, tp = index_pairs(pairs, s1, pool)
    del pairs; gc.collect()
    cand = load_cand(work, "train", s1, pool)
    tcode = np.sort(tq.astype(np.int64) * BIG + tp)
    code = cand.qi.to_numpy(np.int64) * BIG + cand.pi.to_numpy(np.int64)
    cand["y"] = tcode[np.minimum(np.searchsorted(tcode, code), len(tcode) - 1)] == code
    cand["fold"] = fold_s1[cand.qi.to_numpy()]
    del code, tcode; gc.collect()
    log("train cand", len(cand), "positives", int(cand.y.sum()))
    f0 = np.flatnonzero(cand.fold.to_numpy() == 0)
    m1 = xgb_fit(s1_feats(cand, f0, s1, pool), cand.y.to_numpy()[f0], rounds=200, depth=7)
    m1.save_model(f"{work}/model_s1.json")
    log("stage1 fitted")
    cand["p1"] = score_s1(m1, cand, s1, pool)
    wk, C1 = prune_and_context(cand, n_keep)
    del cand; gc.collect()
    sel = (wk.fold >= 1).to_numpy()
    if frac < 1.0:
        m = 10**6
        h = np.fromiter((zlib.crc32(s.encode()) // 5 % m for s in s1.entity_id.tolist()), np.int64, len(s1))
        sel &= h[wk.qi.to_numpy()] < frac * m
    wk, C1 = wk[sel].reset_index(drop=True), C1[sel].reset_index(drop=True)
    log("stage2 rows", len(wk), "positives", int(wk.y.sum()))
    X = stage2_X(wk, C1, s1, pool)
    log("stage2 X", X.shape)
    m2 = xgb_fit(X, wk.y.to_numpy(), rounds=500, depth=8, seed=1)
    m2.save_model(f"{work}/model_s2.json")
    json.dump(list(X.columns), open(f"{work}/model_s2_columns.json", "w"))
    log("stage2 fitted")


def predict(work, out_dir, thr=0.7, n_keep=10):
    from write_outputs import write_matches
    m1, m2 = xgb.Booster(), xgb.Booster()
    m1.load_model(f"{work}/model_s1.json"); m2.load_model(f"{work}/model_s2.json")
    cols = json.load(open(f"{work}/model_s2_columns.json"))
    s1, pool = load_frames(work, "test")
    cand = load_cand(work, "test", s1, pool)
    log("test cand", len(cand), "S1", len(s1), "pool", len(pool))
    cand["p1"] = score_s1(m1, cand, s1, pool)
    wk, C1 = prune_and_context(cand, n_keep)
    del cand; gc.collect()
    log("pruned", len(wk))
    X = stage2_X(wk, C1, s1, pool)[cols]
    wk["p"] = xgb_pred(m2, X)
    del X; gc.collect()
    wk[["qi", "pi", "p1", "p"]].to_parquet(f"{work}/test_scores.parquet", index=False)
    e = decide.by_threshold(wk.rename(columns={"qi": "s1", "pi": "m"})[["s1", "m", "p"]], thr)
    s1_ids, pool_ids = s1.entity_id.to_numpy(object), pool.entity_id.to_numpy(object)
    matches = {}
    for q, p in zip(s1_ids[e.s1.to_numpy()].tolist(), pool_ids[e.m.to_numpy()].tolist()):
        matches.setdefault(q, set()).add(p)
    stats = write_matches(out_dir, s1_ids.tolist(), matches, set(pool_ids.tolist()))
    log("written", stats)


if __name__ == "__main__":
    mode, work = sys.argv[1], sys.argv[2]
    if mode == "fit":
        fit(work, sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 1.0,
            int(sys.argv[5]) if len(sys.argv) > 5 else 10)
    else:
        predict(work, sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 0.7,
                int(sys.argv[5]) if len(sys.argv) > 5 else 10)
