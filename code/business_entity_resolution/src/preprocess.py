"""Stage 1: normalise every record once and cache compact parquet.

usage: python preprocess.py <dataset_dir> <work_dir> <split: train|test> [sources, e.g. 23]
Output: <work_dir>/<split>_s{1,2,3}.parquet with raw + normalised columns.
"""
import os
import sys
import time
from multiprocessing import Pool

import pandas as pd

from io_utils import load_source
from normalize import addr_forms, name_forms

N_KEYS = ["n_tok", "n_digit", "n_core", "n_legal", "n_url", "n_indic"]
A_KEYS = ["a_tok", "a_nums", "a_postal", "a_landmark", "a_blank", "a_indic"]


def _work(args):
    names, addrs = args
    rows = []
    for n, a in zip(names, addrs):
        nf, af = name_forms(n), addr_forms(a)
        rows.append([nf[k] for k in N_KEYS] + [af[k] for k in A_KEYS])
    return rows


def normalise_frame(df: pd.DataFrame, procs: int = 8, chunk: int = 20000) -> pd.DataFrame:
    names, addrs = df.business_name.tolist(), df.business_address.tolist()
    jobs = [(names[i:i + chunk], addrs[i:i + chunk]) for i in range(0, len(names), chunk)]
    with Pool(procs) as p:
        parts = p.map(_work, jobs, chunksize=1)
    out = pd.DataFrame([r for part in parts for r in part], columns=N_KEYS + A_KEYS)
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].astype("string[pyarrow]")
    out.insert(0, "entity_id", df.entity_id.values)
    out.insert(1, "country", df.country.values)
    out["name_raw"] = df.business_name.values
    out["addr_raw"] = df.business_address.values
    return out


if __name__ == "__main__":
    data, work, split = sys.argv[1:4]
    os.makedirs(work, exist_ok=True)
    for k in [int(c) for c in (sys.argv[4] if len(sys.argv) > 4 else "123")]:
        t = time.time()
        df = load_source(data, split, k)
        out = normalise_frame(df)
        out.to_parquet(os.path.join(work, f"{split}_s{k}.parquet"), compression="zstd", index=False)
        print(split, k, len(out), f"{time.time()-t:.0f}s", flush=True)
