"""Exact reproduction of the challenge metric: entity-level macro F0.5.

Rules (problem statement p6):
  * F0.5 = 1.25*P*R / (0.25*P + R), computed per Source-1 entity, then averaged.
  * Singleton (empty truth): 1.0 if prediction empty, else 0.0.
  * Truth non-empty, prediction empty: 0.0 (recall 0; statement silent, see A7).
"""
from __future__ import annotations

import csv
from typing import Dict, Iterable, Mapping, Set

BETA2 = 0.25  # beta = 0.5


def read_id_lists(path: str, id_col: str, list_col: str) -> Dict[str, Set[str]]:
    """Read a one-row-per-S1 TSV of comma-separated ID lists into {s1: set(ids)}.

    Uses the csv module with QUOTE_NONE so that nothing is silently unquoted.
    Raises on duplicate S1 rows or duplicate IDs within a list.
    """
    out: Dict[str, Set[str]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        if reader.fieldnames != [id_col, list_col]:
            raise ValueError(f"{path}: header {reader.fieldnames} != {[id_col, list_col]}")
        for row in reader:
            s1 = row[id_col].strip()
            raw = (row[list_col] or "").strip()
            ids = [x.strip() for x in raw.split(",") if x.strip()] if raw else []
            if s1 in out:
                raise ValueError(f"{path}: duplicate row for {s1}")
            if len(ids) != len(set(ids)):
                raise ValueError(f"{path}: duplicate ids in list for {s1}")
            out[s1] = set(ids)
    return out


def entity_f05(pred: Set[str], truth: Set[str]) -> float:
    """F0.5 for one Source-1 entity, with the official singleton convention."""
    if not truth:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p = tp / len(pred)
    r = tp / len(truth)
    return (1 + BETA2) * p * r / (BETA2 * p + r)


def macro_f05(pred: Mapping[str, Set[str]], truth: Mapping[str, Set[str]],
              s1_ids: Iterable[str] | None = None) -> float:
    """Macro F0.5 over s1_ids (default: all truth keys). Missing preds = empty."""
    ids = list(s1_ids) if s1_ids is not None else list(truth)
    if not ids:
        raise ValueError("empty evaluation set")
    return sum(entity_f05(set(pred.get(s, ())), set(truth.get(s, ()))) for s in ids) / len(ids)


def report(pred: Mapping[str, Set[str]], truth: Mapping[str, Set[str]],
           s1_ids: Iterable[str] | None = None) -> Dict[str, float]:
    """Full diagnostic breakdown: macro F0.5, singleton accuracy, non-singleton F0.5,
    micro pair precision/recall, and share of false merges on singletons."""
    ids = list(s1_ids) if s1_ids is not None else list(truth)
    sing = [s for s in ids if not truth.get(s)]
    non = [s for s in ids if truth.get(s)]
    tp = fp = fn = 0
    for s in ids:
        p, t = set(pred.get(s, ())), set(truth.get(s, ()))
        tp += len(p & t); fp += len(p - t); fn += len(t - p)
    return {
        "n_entities": len(ids),
        "macro_f05": macro_f05(pred, truth, ids),
        "singleton_rate": len(sing) / len(ids),
        "singleton_acc": (sum(not pred.get(s) for s in sing) / len(sing)) if sing else float("nan"),
        "nonsingleton_f05": macro_f05(pred, truth, non) if non else float("nan"),
        "nonsingleton_empty_pred_rate": (sum(not pred.get(s) for s in non) / len(non)) if non else float("nan"),
        "pair_precision": tp / (tp + fp) if tp + fp else float("nan"),
        "pair_recall": tp / (tp + fn) if tp + fn else float("nan"),
    }


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="Score matching_results.tsv against a GT TSV.")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--truth", required=True)
    a = ap.parse_args()
    t = read_id_lists(a.truth, "source1_entity_id", "matched_entity_ids")
    p = read_id_lists(a.pred, "source1_entity_id", "matched_entity_ids")
    extra = set(p) - set(t)
    if extra:
        raise SystemExit(f"prediction has {len(extra)} S1 ids not in truth")
    print(json.dumps(report(p, t), indent=2))
