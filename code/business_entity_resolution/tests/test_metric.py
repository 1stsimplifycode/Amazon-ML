"""Unit tests pinning metric.py to the official definition (problem statement p6)."""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from metric import entity_f05, macro_f05  # noqa: E402


def test_official_worked_example():
    # pred {47,193,812}, truth {47,812} -> 0.714
    f = entity_f05({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"})
    assert math.isclose(f, 1.25 * (2 / 3) / (0.25 * (2 / 3) + 1.0))
    assert round(f, 3) == 0.714


def test_singleton_rules():
    assert entity_f05(set(), set()) == 1.0
    assert entity_f05({"S2-1"}, set()) == 0.0


def test_missed_nonsingleton_is_zero():
    assert entity_f05(set(), {"S2-1"}) == 0.0


def test_disjoint_is_zero():
    assert entity_f05({"S2-2"}, {"S2-1"}) == 0.0


def test_precision_weighted_over_recall():
    # half recall at full precision beats full recall at half precision
    assert entity_f05({"a"}, {"a", "b"}) > entity_f05({"a", "b"}, {"a"})


def test_macro_counts_missing_pred_as_empty():
    truth = {"S1-1": set(), "S1-2": {"S2-1"}}
    assert macro_f05({}, truth) == 0.5


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("all metric tests passed")
