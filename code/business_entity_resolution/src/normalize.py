"""Multi-representation normalisation for names and addresses.

Design rules:
  * Raw text is never overwritten; every representation is a new column.
  * Open set: nothing depends on the country label. Abbreviation tables hold
    US, India and French forms side by side and apply to every row.
  * Indic scripts are romanised in-house (no GPL unidecode). All nine major Indic
    blocks share the ISCII layout, so one offset table covers Devanagari, Bengali,
    Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada and Malayalam. A learned
    token dictionary (translit_dict.py) can override the rule output.
  * Numbers are kept. Digits embedded inside alphabetic name tokens (Exp0, Pri0rity)
    are treated as leetspeak in the *name letters* form only; the digit-preserving
    form keeps them.
"""
from __future__ import annotations

import re
import unicodedata

# ----------------------------------------------------------------------------
# Indic romanisation (offset table relative to each 128-codepoint block)
# ----------------------------------------------------------------------------
_V = {0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri", 0x0C: "li",
      0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au",
      0x60: "ri", 0x61: "li"}
_C = {0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch", 0x1B: "chh", 0x1C: "j",
      0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n", 0x24: "t",
      0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n", 0x2A: "p", 0x2B: "f", 0x2C: "b",
      0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "zh",
      0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
      0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "d", 0x5D: "dh", 0x5E: "f", 0x5F: "y"}
_M = {0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x44: "ri", 0x45: "e",
      0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au", 0x57: "au",
      0x62: "li", 0x63: "li"}
_VIRAMA, _ANUSVARA = 0x4D, (0x01, 0x02)
_INDIC_LO, _INDIC_HI = 0x0900, 0x0D7F


def _is_indic(ch: str) -> bool:
    return _INDIC_LO <= ord(ch) <= _INDIC_HI


def romanize_indic(text: str) -> str:
    """Rule-based romanisation with inherent-a and word-final schwa deletion."""
    if not any(_is_indic(c) for c in text):
        return text
    out, pending = [], False  # pending: last emitted consonant still owns an inherent 'a'
    def flush():
        nonlocal pending
        if pending:
            out.append("a")
        pending = False
    for ch in text:
        cp = ord(ch)
        if not (_INDIC_LO <= cp <= _INDIC_HI):
            if pending:  # word-final schwa deletion
                pending = False
            out.append(ch)
            continue
        o = cp & 0x7F
        if o in _C:
            flush(); out.append(_C[o]); pending = True
        elif o in _M:
            pending = False; out.append(_M[o])
        elif o == _VIRAMA:
            pending = False
        elif o in _V:
            flush(); out.append(_V[o])
        elif o in _ANUSVARA:
            flush(); out.append("n")
        elif o == 0x03:
            flush(); out.append("h")
        elif 0x66 <= o <= 0x6F:
            flush(); out.append(str(o - 0x66))
        elif o in (0x64, 0x65):
            pending = False; out.append(" ")
        # nukta, avagraha, other marks: drop
    return "".join(out)


def has_indic(text: str) -> bool:
    return any(_is_indic(c) for c in text)


# ----------------------------------------------------------------------------
# Latin folding
# ----------------------------------------------------------------------------
_SPECIAL = str.maketrans({"ß": "ss", "æ": "ae", "Æ": "ae", "œ": "oe", "Œ": "oe", "ø": "o", "Ø": "o",
                          "ł": "l", "Ł": "l", "đ": "d", "Đ": "d", "ð": "d", "þ": "th", "ı": "i",
                          "’": "'", "‘": "'", "`": "'", "´": "'", "“": '"', "”": '"', "–": "-", "—": "-",
                          "°": " ", "º": " ", "ª": " ", "№": " no "})


def fold(text: str) -> str:
    """NFKC -> Indic romanisation -> strip accents -> lowercase. Keeps punctuation."""
    t = unicodedata.normalize("NFKC", text)
    t = romanize_indic(t) if has_indic(t) else t
    t = t.translate(_SPECIAL)
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    return t.lower()


# ----------------------------------------------------------------------------
# Names
# ----------------------------------------------------------------------------
LEGAL = {
    "pvt": "pvt", "private": "pvt", "pvt.": "pvt", "prvt": "pvt",
    "ltd": "ltd", "limited": "ltd", "ltd.": "ltd",
    "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co", "llc": "llc", "llp": "llp", "plc": "plc", "lp": "lp",
    "pllc": "pllc", "pc": "pc", "ltda": "ltda", "gmbh": "gmbh", "opc": "opc",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sa": "sa", "sci": "sci", "snc": "snc",
    "sca": "sca", "scop": "scop", "selarl": "selarl", "cie": "cie", "ets": "ets",
    "etablissements": "ets", "etablissement": "ets", "public": "public",
}
# dotted forms collapse before tokenising: l.l.c. -> llc, s.a.r.l. -> sarl, p.c. -> pc
_DOTTED = re.compile(r"\b((?:[a-z]\.){2,}[a-z]?\.?)")
_URL = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9\-]+)\.(?:com|net|org|in|co\.in|co|biz|info|fr|us|io)\b")
_LEET = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"}
_TOKEN = re.compile(r"[a-z0-9]+")


def _dedot(t: str) -> str:
    return _DOTTED.sub(lambda m: m.group(1).replace(".", ""), t)


def name_forms(raw: str) -> dict:
    f = fold(raw)
    punct_kept = " ".join(f.split())
    t = _dedot(f)
    m = _URL.match(t.strip())
    is_url = bool(m)
    if m:
        t = m.group(1) + t[m.end():]
    t = t.replace("&", " and ").replace("+", " and ")
    t = re.sub(r"\bm/s\b", " ", t)  # Indian "Messrs" prefix
    t = t.replace("'", "")          # apostrophes join: stephenie's -> stephenies
    digit_tokens = _TOKEN.findall(t)
    # letters-only form: leetspeak inside alphabetic tokens, pure numbers kept
    letters = []
    for tok in digit_tokens:
        if tok.isdigit() or not any(c.isalpha() for c in tok):
            letters.append(tok)
        elif any(c.isdigit() for c in tok) and tok[0].isalpha():
            letters.append("".join(_LEET.get(c, c) for c in tok))
        else:
            letters.append(tok)
    legal = [LEGAL[x] for x in letters if x in LEGAL]
    core = [x for x in letters if x not in LEGAL and x != "and" and x != "the"]
    return {
        "n_fold": punct_kept,
        "n_tok": " ".join(letters),
        "n_digit": " ".join(digit_tokens),
        "n_core": " ".join(core),
        "n_sorted": " ".join(sorted(core)),
        "n_legal": " ".join(sorted(set(legal))),
        "n_nospace": "".join(core),
        "n_url": is_url,
        "n_indic": has_indic(raw),
    }


# ----------------------------------------------------------------------------
# Addresses
# ----------------------------------------------------------------------------
ADDR_ABBR = {
    # street types (US / IN / FR share the table; open set)
    "street": "st", "str": "st", "road": "rd", "avenue": "ave", "av": "ave", "avn": "ave", "drive": "dr",
    "boulevard": "blvd", "bd": "blvd", "boul": "blvd", "lane": "ln", "court": "ct", "place": "pl",
    "circle": "cir", "terrace": "ter", "trail": "trl", "parkway": "pkwy", "highway": "hwy", "hiway": "hwy",
    "square": "sq", "suite": "ste", "apartment": "apt", "apartments": "apts", "building": "bldg",
    "floor": "fl", "flr": "fl", "number": "no", "nos": "no", "nagar": "ngr", "sector": "sec",
    "north": "n", "south": "s", "east": "e", "west": "w", "mount": "mt", "fort": "ft",
    "rue": "rue", "chemin": "ch", "chem": "ch", "impasse": "imp", "allee": "all", "route": "rte",
    "quai": "qu", "faubourg": "fbg", "residence": "res", "saint": "st", "sainte": "ste",
    "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th",
}
LANDMARK = {"near", "nr", "opp", "opposite", "behind", "beside", "next", "adjacent", "facing", "besides", "infront"}
NULLISH = {"null", "none", "nan", "na", "n/a"}
_NUM = re.compile(r"\d+")
_POSTAL = re.compile(r"(?<![\d])(\d{6}|\d{5}(?:-\d{4})?|\d{3}\s\d{3})(?![\d])")


def addr_forms(raw: str) -> dict:
    f = fold(raw)
    t = _dedot(f).replace("&", " and ")
    toks = [x for x in _TOKEN.findall(t) if x not in NULLISH]
    norm = [ADDR_ABBR.get(x, x) for x in toks]
    nums = _NUM.findall(t)
    postal = [p.replace(" ", "") for p in _POSTAL.findall(t)]
    # first number-bearing token = building/house/shop number candidate
    num_toks = [x for x in norm if any(c.isdigit() for c in x)]
    lm = [i for i, x in enumerate(toks) if x in LANDMARK]
    landmark = [toks[i + 1] for i in lm if i + 1 < len(toks)]
    alpha = [x for x in norm if not any(c.isdigit() for c in x)]
    return {
        "a_fold": " ".join(f.split()),
        "a_tok": " ".join(norm),
        "a_alpha": " ".join(alpha),
        "a_nums": " ".join(nums),
        "a_numtok": " ".join(num_toks),
        "a_first_num": nums[0] if nums else "",
        "a_postal": " ".join(postal),
        "a_landmark": " ".join(landmark),
        "a_blank": not raw.strip(),
        "a_indic": has_indic(raw),
    }
