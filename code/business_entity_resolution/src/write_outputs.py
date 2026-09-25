"""Write matching_results.tsv / candidate_pairs.tsv and hard-check them.

Checks (raise on failure, never auto-fix silently):
  * exact headers, TAB separated, UTF-8, '\\n' line endings
  * every test S1 exactly once, nothing else
  * no duplicate IDs inside a list, only S2-/S3- IDs that exist in the test pool
  * matches subset of candidates
"""
from __future__ import annotations

import os

MATCH_HDR = "source1_entity_id\tmatched_entity_ids"
CAND_HDR = "source1_entity_id\tcandidate_entity_ids"


def _write(path: str, header: str, s1_order: list, sets: dict) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(header + "\n")
        for s in s1_order:
            ids = sorted(sets.get(s, ()))
            f.write(s + "\t" + ",".join(ids) + "\n")


def write_and_check(out_dir: str, s1_order: list, matches: dict, cands: dict, valid_pool: set) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    if len(set(s1_order)) != len(s1_order):
        raise ValueError("duplicate S1 in required order")
    req = set(s1_order)
    for name, d in (("matches", matches), ("candidates", cands)):
        extra = set(d) - req
        if extra:
            raise ValueError(f"{name}: {len(extra)} S1 not in test, e.g. {list(extra)[:3]}")
        for s, ids in d.items():
            bad = [x for x in ids if not (x.startswith("S2-") or x.startswith("S3-")) or x not in valid_pool]
            if bad:
                raise ValueError(f"{name}: invalid ids for {s}: {bad[:3]}")
    outside = [s for s, ids in matches.items() if set(ids) - set(cands.get(s, ()))]
    if outside:
        raise ValueError(f"{len(outside)} S1 have matches outside candidates, e.g. {outside[:3]}")
    mp, cp = os.path.join(out_dir, "matching_results.tsv"), os.path.join(out_dir, "candidate_pairs.tsv")
    _write(mp, MATCH_HDR, s1_order, matches)
    _write(cp, CAND_HDR, s1_order, cands)
    # re-read and verify byte-level structure
    stats = {}
    for path, hdr in ((mp, MATCH_HDR), (cp, CAND_HDR)):
        with open(path, encoding="utf-8", newline="") as f:
            lines = f.read().split("\n")
        assert lines[-1] == "", "file must end with newline"
        lines = lines[:-1]
        assert lines[0] == hdr, f"bad header {lines[0]!r}"
        assert all("\r" not in l for l in lines), "CR found"
        rows = [l.split("\t") for l in lines[1:]]
        assert all(len(r) == 2 for r in rows), "row without exactly one TAB"
        ids = [r[0] for r in rows]
        assert ids == s1_order, "S1 order/coverage mismatch"
        for r in rows:
            lst = r[1].split(",") if r[1] else []
            assert len(lst) == len(set(lst)), f"dup id in {r[0]}"
            assert all(x for x in lst), f"empty token in {r[0]}"
        stats[os.path.basename(path)] = dict(rows=len(rows), empty=sum(1 for r in rows if not r[1]),
                                             ids=sum(len(r[1].split(",")) for r in rows if r[1]))
    return stats


def write_matches(out_dir: str, s1_order: list, matches: dict, valid_pool: set) -> dict:
    """Write matching_results.tsv alone (leaderboard file) with the same hard checks."""
    os.makedirs(out_dir, exist_ok=True)
    if len(set(s1_order)) != len(s1_order):
        raise ValueError("duplicate S1 in required order")
    extra = set(matches) - set(s1_order)
    if extra:
        raise ValueError(f"matches: {len(extra)} S1 not in test, e.g. {list(extra)[:3]}")
    for s, ids in matches.items():
        bad = [x for x in ids if not (x.startswith("S2-") or x.startswith("S3-")) or x not in valid_pool]
        if bad:
            raise ValueError(f"matches: invalid ids for {s}: {bad[:3]}")
    mp = os.path.join(out_dir, "matching_results.tsv")
    _write(mp, MATCH_HDR, s1_order, matches)
    with open(mp, encoding="utf-8", newline="") as f:
        lines = f.read().split("\n")
    assert lines[-1] == "" and lines[0] == MATCH_HDR and all("\r" not in l for l in lines)
    rows = [l.split("\t") for l in lines[1:-1]]
    assert all(len(r) == 2 for r in rows) and [r[0] for r in rows] == s1_order
    return dict(rows=len(rows), empty=sum(1 for r in rows if not r[1]),
                ids=sum(len(r[1].split(",")) for r in rows if r[1]))
