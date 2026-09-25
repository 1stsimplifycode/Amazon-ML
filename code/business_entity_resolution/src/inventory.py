"""00 dataset inventory: one file per invocation, prints a JSON blob of stats.

Reads with sep='\t', QUOTE_NONE, dtype=str, keep_default_na=False, so blanks and a
literal "NA" survive. Also re-reads the header without sep to prove the one-column
failure mode, and counts raw lines to catch embedded newlines / dropped rows.
"""
import csv, json, re, sys
import pandas as pd

path = sys.argv[1]
out = {"path": path}
with open(path, encoding="utf-8") as f:
    header = f.readline().rstrip("\n")
    nlines = 1 + sum(1 for _ in f)
out["raw_lines_incl_header"] = nlines
out["header_tab_count"] = header.count("\t")
out["header_if_parsed_as_csv"] = len(next(csv.reader([header])))
df = pd.read_csv(path, sep="\t", dtype="string[pyarrow]", keep_default_na=False, na_filter=False,
                 quoting=csv.QUOTE_NONE, encoding="utf-8")
out["rows"] = len(df); out["cols"] = df.shape[1]; out["columns"] = list(df.columns)
out["rows_vs_lines_diff"] = nlines - 1 - len(df)
out["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}
out["null"] = {c: int(df[c].isna().sum()) for c in df}
out["blank"] = {c: int((df[c].str.strip() == "").sum()) for c in df}
out["lead_trail_ws"] = {c: int((df[c] != df[c].str.strip()).sum()) for c in df}
out["literal_nullish"] = {c: int(df[c].str.strip().str.lower().isin(["nan", "null", "none", "na", "n/a", "-"]).sum()) for c in df}
out["dup_rows_full"] = int(df.duplicated().sum())
idc = df.columns[0]
out["dup_first_col"] = int(df[idc].duplicated().sum())
if idc == "entity_id":
    out["dup_rows_excl_id"] = int(df.iloc[:, 1:].duplicated().sum())
    pref = df[idc].str.slice(0, 3).value_counts().to_dict()
    out["id_prefixes"] = {str(k): int(v) for k, v in pref.items()}
    ok = df[idc].str.fullmatch(r"S[123]-\d+")
    out["malformed_ids"] = int((~ok).sum())
    out["malformed_examples"] = df.loc[~ok, idc].head(10).tolist()
    num = df[idc].str.slice(3)
    out["id_num_len"] = {str(k): int(v) for k, v in num.str.len().value_counts().sort_index().to_dict().items()}
    out["id_leading_zero"] = int(num.str.match(r"0\d").sum())
    out["country"] = {str(k): int(v) for k, v in df["country"].value_counts().to_dict().items()}
    out["name_len_q"] = df.business_name.str.len().quantile([0, .5, .99, 1]).tolist()
    out["addr_len_q"] = df.business_address.str.len().quantile([0, .5, .99, 1]).tolist()
    out["name_one_char"] = int((df.business_name.str.strip().str.len() == 1).sum())
    out["addr_one_char"] = int((df.business_address.str.strip().str.len() == 1).sum())
    out["both_blank"] = int(((df.business_name.str.strip() == "") & (df.business_address.str.strip() == "")).sum())
    out["contains_comma_in_id"] = int(df[idc].str.contains(",").sum())
    samp = df.sample(min(6, len(df)), random_state=0)
    out["sample"] = samp.values.tolist()
    per_c = {}
    for c, g in df.groupby("country"):
        per_c[str(c)] = {"n": len(g), "blank_name": int((g.business_name.str.strip() == "").sum()),
                         "blank_addr": int((g.business_address.str.strip() == "").sum()),
                         "sample": g.sample(min(3, len(g)), random_state=1).values.tolist()}
    out["per_country"] = per_c
else:
    lst = df.iloc[:, 1]
    n = lst.str.split(",").map(lambda x: 0 if x == [""] else len(x))
    out["list_len_dist"] = {str(k): int(v) for k, v in n.value_counts().sort_index().to_dict().items()}
    ok = df[idc].str.fullmatch(r"S1-\d+")
    out["malformed_ids"] = int((~ok).sum())
    allids = [x for s in lst for x in s.split(",") if x]
    bad = [x for x in allids if not re.fullmatch(r"S[23]-\d+", x)]
    out["malformed_list_ids"] = len(bad); out["malformed_list_examples"] = bad[:10]
    out["list_ids_total"] = len(allids); out["list_ids_unique"] = len(set(allids))
    out["list_ids_with_space"] = sum(1 for x in allids if x != x.strip())
    out["intra_list_dupes"] = int(sum(1 for s in lst if s and len(s.split(",")) != len(set(s.split(",")))))
    out["sample"] = df.sample(6, random_state=0).values.tolist()
print(json.dumps(out, ensure_ascii=False, default=str))
