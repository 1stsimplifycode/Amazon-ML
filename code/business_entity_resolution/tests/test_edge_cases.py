"""Hostile edge-case suite (normalisation, features, decision, output format).

Run: python -m pytest tests/test_edge_cases.py -q   (from code/business_entity_resolution)
Model-behaviour cases (does the trained scorer rank X above Y) live in
tests/test_model_edges.py and need a trained model file.
"""
import os
import sys
import tempfile

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from decide import exclusive, expected_f, by_threshold  # noqa: E402
from features import pair_features  # noqa: E402
from normalize import addr_forms, name_forms, romanize_indic  # noqa: E402
from write_outputs import write_and_check  # noqa: E402


def rec(name, addr, country="US"):
    n, a = name_forms(name), addr_forms(addr)
    return {**n, **a, "name_raw": name, "addr_raw": addr, "country": country}


def feats(r1, r2):
    a, b = pd.DataFrame([r1]), pd.DataFrame([r2])
    return pair_features(a, b, {}, {}, 1000).iloc[0]


# ---- 1-3 exact / conflicting combos -------------------------------------------------
def test_01_exact_name_exact_address():
    f = feats(rec("Ace Cargo Inc", "30261 Meadowbrook Drive, Willoughby Hills, OH"),
              rec("Ace Cargo Inc", "30261 Meadowbrook Drive, Willoughby Hills, OH"))
    assert f.n_raw_eq == 1 and f.a_eq == 1 and f.num_first_eq == 1


def test_02_exact_name_different_address():
    f = feats(rec("Ace Cargo Inc", "30261 Meadowbrook Drive, Willoughby Hills, OH"),
              rec("Ace Cargo Inc", "118 Elm St, Morganton, NC"))
    assert f.n_core_eq == 1 and f.num_conflict == 1 and f.x_name_eq_num_conflict == 1


def test_03_different_name_exact_address():
    f = feats(rec("Ace Cargo Inc", "30261 Meadowbrook Drive, Willoughby Hills, OH"),
              rec("Zephyr Dental", "30261 Meadowbrook Drive, Willoughby Hills, OH"))
    assert f.a_eq == 1 and f.n_jacc == 0 and f.x_addr_high_name_low == 1


# ---- 4-11 name noise --------------------------------------------------------------
@pytest.mark.parametrize("a,b", [
    ("Shiv Infra Private Limited", "Shiv Infra Pvt. Ltd."),            # 4 legal suffix
    ("Stephenie's Auto Sales L.L.C.", "Stephenies Auto Sales LLC"),    # 5 punctuation/apostrophe
    ("Newton Equity Partners, Inc", "Newton Equity Partners Inc."),    # 5 punctuation
    ("Remy Villines LLC", "Villines Remy LLC"),                         # 9 token reorder (sorted)
    ("Caldent Fócus Partners", "Caldent Focus Partners"),              # 11 accent
    ("Établissements Sakado", "ETABLISSEMENTS SAKADO"),                # 44 French accent/case
])
def test_04_11_name_equivalences(a, b):
    na, nb = name_forms(a), name_forms(b)
    assert na["n_sorted"] == nb["n_sorted"], (na, nb)


def test_06_ampersand_vs_and():
    assert name_forms("Pyle & Arp")["n_tok"] == name_forms("Pyle and Arp")["n_tok"]


def test_07_abbreviation_legal_canonical():
    assert name_forms("Acme Corporation")["n_legal"] == name_forms("ACME CORP")["n_legal"] == "corp"


def test_08_typo_keeps_high_similarity():
    f = feats(rec("Supreme Healthcare", "A/104 New Ajit Society, Thane"), rec("Supreme Helathcare", "A/104 New Ajit Society, Thane"))
    assert f.n_ratio > 85 and f.n_jw > 0.9


def test_10_transliteration_devanagari():
    assert romanize_indic("लिमिटेड") == "limited"
    assert name_forms("ग्रीन कंसल्टेंसी")["n_indic"] is True


def test_33_numeric_name_digits_preserved():
    n = name_forms("24x7 Express 360 Services")
    assert "360" in n["n_digit"] and "360" in n["n_tok"]


def test_leet_digits_in_alpha_tokens():
    n = name_forms("Private Exp0 Technology Limited")
    assert "expo" in n["n_tok"] and "exp0" in n["n_digit"]


def test_url_name():
    n = name_forms("arihantmanagement.com")
    assert n["n_url"] and n["n_nospace"] == "arihantmanagement"


# ---- 12-20 address ------------------------------------------------------------------
def test_12_13_missing_and_reordered_address():
    f = feats(rec("X", "1528 84th Place, Chicago, IL"), rec("X", "IL, CHICAGO, 1528 84TH PL"))
    assert f.a_tset == 100 and f.num_jacc == 1


def test_14_landmark_only():
    a = addr_forms("Near SBI ATM, Opp Bus Stand, Rohtak, Haryana")
    assert "sbi" in a["a_landmark"].split() and a["a_nums"] == ""


def test_15_16_postal_missing_and_wrong():
    f = feats(rec("X", "12 MG Road, Pune 411001"), rec("X", "12 MG Road, Pune"))
    assert f.postal_one_missing == 1 and f.postal_conflict == 0
    f = feats(rec("X", "12 MG Road, Pune 411001"), rec("X", "12 MG Road, Pune 560001"))
    assert f.postal_conflict == 1 and f.postal_eq == 0


def test_17_18_building_number_match_and_conflict():
    f = feats(rec("X", "Flat 501, 18 Srinidhi Nest"), rec("X", "18 Srinidhi Nest, Flat 501"))
    assert f.num_jacc == 1
    f = feats(rec("X", "Flat 501, 18 Srinidhi Nest"), rec("X", "Flat 502, 19 Srinidhi Nest"))
    assert f.num_conflict == 1


def test_19_20_leading_zero_unit_numbers_equal():
    f = feats(rec("X", "A/104 New Ajit Society"), rec("X", "A/0104 NEW AJIT SOCIETY"))
    assert f.num_any_shared == 1 and f.num_conflict == 0


def test_null_token_removed():
    assert "null" not in addr_forms("5 Mary Ella Court, null, Smyrna, DE")["a_tok"].split()


# ---- 29-37 degenerate inputs ------------------------------------------------------------
@pytest.mark.parametrize("name,addr", [("", ""), ("A", ""), ("", "X"), ("NA", "null"), ("   ", "  "),
                                       ("<< -- >>", "##"), ("Ｓｍａｒｔ　Ｆｏｏｄｓ", "１２ ＭＧ Ｒｏａｄ"),
                                       ("O'Brien-Smith  &  Co", "12-B,  Main   St")])
def test_29_37_degenerate_no_crash(name, addr):
    f = feats(rec(name, addr), rec(name, addr))
    assert not f.isna().any()


def test_34_unicode_fullwidth_folds():
    assert name_forms("Ｓｍａｒｔ Ｆｏｏｄｓ")["n_tok"] == "smart foods"


def test_36_hyphenation():
    assert name_forms("Surgical-Center LLC")["n_core"] == name_forms("Surgical Center LLC")["n_core"]


def test_37_multiple_spaces():
    assert name_forms("Sree  Law")["n_tok"] == "sree law"


def test_literal_na_name_is_a_string():
    assert name_forms("NA")["n_tok"] == "na"


# ---- 25-28, 38-40 decision layer ---------------------------------------------------------
def D(rows):
    return pd.DataFrame(rows, columns=["s1", "m", "p"])


def test_25_27_multi_match_kept_by_expected_f():
    sel = expected_f(D([("a", "S2-1", .95), ("a", "S2-2", .9), ("a", "S3-1", .9), ("a", "S3-2", .02)]))
    assert set(sel.m) == {"S2-1", "S2-2", "S3-1"}


def test_28_singleton_rejected():
    assert expected_f(D([("a", "S2-1", .2), ("a", "S2-2", .1)])).empty


def test_38_duplicate_candidate_rows_do_not_duplicate_output():
    sel = by_threshold(D([("a", "S2-1", .9), ("a", "S2-1", .9)]), .5).drop_duplicates(["s1", "m"])
    assert len(sel) == 1


def test_exclusivity_one_pool_record_one_s1():
    sel = exclusive(D([("a", "S2-1", .9), ("b", "S2-1", .6), ("b", "S2-2", .7)]))
    assert set(map(tuple, sel[["s1", "m"]].values)) == {("a", "S2-1"), ("b", "S2-2")}


# ---- output format -----------------------------------------------------------------------------
def test_writer_rejects_match_outside_candidates():
    with tempfile.TemporaryDirectory() as d, pytest.raises(ValueError):
        write_and_check(d, ["S1-1"], {"S1-1": {"S2-9"}}, {"S1-1": {"S2-1"}}, {"S2-1", "S2-9"})


def test_writer_rejects_unknown_id():
    with tempfile.TemporaryDirectory() as d, pytest.raises(ValueError):
        write_and_check(d, ["S1-1"], {"S1-1": {"S2-404"}}, {"S1-1": {"S2-404"}}, {"S2-1"})


def test_45_writer_empty_row_for_no_candidate():
    with tempfile.TemporaryDirectory() as d:
        st = write_and_check(d, ["S1-1", "S1-2"], {"S1-1": {"S2-1"}}, {"S1-1": {"S2-1"}}, {"S2-1"})
        assert st["matching_results.tsv"]["empty"] == 1
        txt = open(os.path.join(d, "matching_results.tsv"), encoding="utf-8").read()
        assert txt == "source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1\nS1-2\t\n"
