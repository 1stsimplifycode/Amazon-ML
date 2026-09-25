"""Blocking recall of a candidate file against train ground truth (integer-encoded).

usage: python blocking_recall.py <work_dir> <data_dir> <cand.parquet>
Reports recall of forward candidates at several K, of forward + reverse, and the
number of true pairs found only by the reverse pass.
"""
import sys

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

import io_utils


def enc(ids) -> np.ndarray:
    """'S<k>-<n>' -> k * 10**9 + n (int64); asserts n < 10**9 so the code is lossless."""
    a = pc.cast(ids, "string") if not isinstance(ids, np.ndarray) else ids
    src = pc.cast(pc.utf8_slice_codeunits(a, 1, 2), "int64").to_numpy()
    num = pc.cast(pc.utf8_slice_codeunits(a, 3), "int64").to_numpy()
    assert num.max() < 10**9
    return src * 10**9 + num


def pair_code(s1, m) -> np.ndarray:
    # pool codes are < 4 * 10**9, S1 numbers < 10**9: product stays below int64 max
    return s1.astype(np.int64) * 4 * 10**9 + m


if __name__ == "__main__":
    work, data, cand_path = sys.argv[1:4]
    import pyarrow as pa
    _, pairs = io_utils.load_gt_pairs(data)
    c = pq.read_table(cand_path, columns=["s1", "m", "rank", "rev"])
    s1_in = np.unique(enc(c.column("s1").combine_chunks()) % 10**9)
    gs = enc(pa.array(pairs.s1.to_numpy(object))) % 10**9
    gm = enc(pa.array(pairs.m.to_numpy(object)))
    sel = np.isin(gs, s1_in)
    g = np.unique(pair_code(gs[sel], gm[sel]))
    cc = pair_code(enc(c.column("s1").combine_chunks()) % 10**9, enc(c.column("m").combine_chunks()))
    rank, rev = c.column("rank").to_numpy(), c.column("rev").to_numpy()
    hit = np.isin(cc, g)
    print(f"S1 {len(s1_in)}  true pairs {len(g)}  candidates {len(cc)} ({len(cc)/len(s1_in):.1f}/S1)")
    for K in (5, 10, 20, 30):
        print(f"fwd @K={K}: {hit[(rev == 0) & (rank < K)].sum() / len(g):.4f}")
    print(f"fwd+rev: {hit.sum() / len(g):.4f}  rev-only true pairs: {int(hit[rev == 1].sum())}"
          f" of {int((rev == 1).sum())} rev-only candidates")
