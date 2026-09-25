"""Strict TSV loading shared by every stage.

Every read uses sep='\\t', QUOTE_NONE, str dtype and keep_default_na=False, so a
business literally named "NA" or an empty address is never turned into NaN, and IDs
are never cast to int (they are variable-length).
"""
from __future__ import annotations

import csv
import os

import pandas as pd

SRC_COLS = ["entity_id", "business_name", "business_address", "country"]
GT_COLS = ["source1_entity_id", "matched_entity_ids"]


def read_tsv(path: str, expected_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype="string[pyarrow]", keep_default_na=False,
                     na_filter=False, quoting=csv.QUOTE_NONE, encoding="utf-8")
    if list(df.columns) != expected_cols:
        raise ValueError(f"{path}: columns {list(df.columns)} != {expected_cols} (bad separator?)")
    return df


def load_source(data_dir: str, split: str, k: int) -> pd.DataFrame:
    df = read_tsv(os.path.join(data_dir, split, f"{split}_source{k}.tsv"), SRC_COLS)
    bad = ~df.entity_id.str.fullmatch(rf"S{k}-\d+")
    if bad.any():
        raise ValueError(f"{split} source{k}: {int(bad.sum())} malformed IDs, e.g. {df.entity_id[bad].head(3).tolist()}")
    if df.entity_id.duplicated().any():
        raise ValueError(f"{split} source{k}: duplicate entity_id")
    return df


def load_gt_pairs(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (gt_rows, pairs). pairs has one row per (s1, matched id)."""
    gt = read_tsv(os.path.join(data_dir, "train", "train_ground_truth.tsv"), GT_COLS)
    lst = gt.matched_entity_ids.astype(str).str.split(",")
    pairs = pd.DataFrame({"s1": gt.source1_entity_id.repeat(lst.str.len()).values,
                          "m": [x for xs in lst for x in xs]})
    pairs = pairs[pairs.m != ""].reset_index(drop=True).astype("string[pyarrow]")
    return gt, pairs
