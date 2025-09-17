# make_reversed_and_keepers.py
# Streamlit app to produce:
#  - reversed_filters.csv/txt  (for filters that should be reversed)
#  - keepers_formatted.csv/txt (for filters that should NOT be reversed)
#
# Inputs:
#  - A results file from the Filter Runner (either "reversal_candidates.csv/txt"
#    OR a full "filter_results.csv/txt" summary), and
#  - The original filter batch file (txt/csv) so we can pull the canonical expression.
#
# Behavior:
#  - If you upload nothing, it tries to read repo files:
#      "reversal_candidates.csv|txt"  OR  "filter_results.csv|txt"
#      plus "test 4pwrballfilters.txt" (or "test_4pwrballfilters.txt" fallback)
#
# Notes:
#  - Reversal option: "Flip comparators" (<= ↔ >, < ↔ >=, == ↔ !=, >= ↔ <, > ↔ <=) [default]
#                     OR "Wrap with not(...)" if you prefer logical negation.
#  - Variant guard: include optional "variant_name == '<variant>'" gate in each expression.

from __future__ import annotations

import io
import re
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd
import streamlit as st

# ---------------------------
# General helpers
# ---------------------------

def read_any_table(file_or_str) -> pd.DataFrame:
    """
    Robust reader for CSV or tab-ish text. Accepts a file-like (uploader) or string buffer.
    Tries CSV first; falls back to TSV; then pandas' python engine sniff.
    """
    if hasattr(file_or_str, "read"):
        raw = file_or_str.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="ignore")
        buf = io.StringIO(raw)
    else:
        # assume it's a ready-made text buffer
        buf = file_or_str

    buf.seek(0)
    try:
        return pd.read_csv(buf)
    except Exception:
        buf.seek(0)
        try:
            return pd.read_csv(buf, sep="\t")
        except Exception:
            buf.seek(0)
            return pd.read_csv(buf, engine="python")


# Token normalization used in your main app
LEGACY_MAP = [
    (r"\bcombo_structure\b", "winner_structure"),
    (r"\bcombo_sum\b", "winner"),
    (r"\bcombo_total\b", "winner"),
    (r"\bcombo\b", "winner"),
    (r"\bseed_sum\b", "seed"),
    (r"\bseed_total\b", "seed"),
    (r"\bones_total\b", "winner"),
    (r"\btens_total\b", "winner"),
    (r"\bfull_combo\b", "winner"),
    (r"“|”|‘|’", "\""),
    (r"≤", "<="),
    (r"≥", ">="),
    (r"≠", "!="),
    (r"–", "-"),
]

BOOLEAN_LITERALS = {"true", "false", "1", "0", "yes", "no"}

def is_boolean_literal(s: str) -> bool:
    return s.strip().lower() in BOOLEAN_LITERALS

def normalize_expression(expr: str) -> str:
    if not isinstance(expr, str):
        return ""
    e = expr.strip()
    if not e:
        return ""
    e = e.strip("\"'")
    low = e.lower()
    if "see prior" in low or "see conversation" in low:
        return ""
    for pat, repl in LEGACY_MAP:
        e = re.sub(pat, repl, e, flags=re.IGNORECASE)
    e = re.sub(r"\bAND\b", "and", e)
    e = re.sub(r"\bOR\b", "or", e)
    if is_boolean_literal(e):
        return ""
    return e

def layman_explanation(expr: str) -> str:
    if not expr:
        return "Unparseable / constant"
    repl = {
        "seed": "seed value",
        "winner": "winner value",
        "==": "equals",
        "<=": "is ≤",
        ">=": "is ≥",
        "<": "is <",
        ">": "is >",
        " and ": " AND ",
        " or ": " OR ",
    }
    text = expr
    for k, v in repl.items():
        text = text.replace(k, v)
    return f"Eliminate if {text}"

# ---------------------------
# Reversal logic
# ---------------------------

def reverse_by_comparator_flip(expr: str) -> str:
    """
    Flip relational operators to get the logical complement per atomic comparison:
      <= -> >
      <  -> >=
      >= -> <
      >  -> <=
      == -> !=
      != -> ==
    We do a two-phase token-protect pass to avoid accidental double-replacements.
    """
    if not expr:
        return ""

    e = expr

    # protect multi-char first
    protect = {
        "<=": "§LE§",
        ">=": "§GE§",
        "==": "§EQ§",
        "!=": "§NE§",
    }
    for k, v in protect.items():
        e = e.replace(k, v)

    # single-char
    e = e.replace("<", "§LT§").replace(">", "§GT§")

    # flip
    flip = {
        "§LE§": ">",
        "§GE§": "<",
        "§LT§": ">=",
        "§GT§": "<=",
        "§EQ§": "!=",
        "§NE§": "==",
    }
    for k, v in flip.items():
        e = e.replace(k, v)

    return e

def reverse_by_not_wrapper(expr: str) -> str:
    """
    Logical negation fallback. If already not(...), strip it; otherwise wrap with not(...).
    """
    e = expr.strip()
    if e.startswith("not(") and e.endswith(")"):
        # unwrap
        return e[4:-1].strip()
    return f"not({e})"

def make_variant_guard(variant: str) -> str:
    # Always single quotes for safety; variant names are pos1..pos5, possum1..possum5, ones, tens, full
    return f"(variant_name == '{variant}')"

def build_output_row(fid: str,
                     variant: str,
                     expr: str,
                     eliminated: Optional[int] = None,
                     total: Optional[int] = None,
                     extra_cols: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    stat = ""
    if eliminated is not None and total is not None and total > 0:
        stat = f"{eliminated}/{total}"
    row = {
        "filter_id": fid,
        "variant": variant,
        "expression": expr,
        "layman_explanation": layman_explanation(expr),
        "stat": stat
    }
    if extra_cols:
        row.update(extra_cols)
    return row

# ---------------------------
# Filter sources & joins
# ---------------------------

def load_filters_any(text: str) -> pd.DataFrame:
    """
    Parses a batch filter file (CSV/TXT).
    Returns DataFrame with at least ['filter_id','expression'].
    Accepts:
      - header file with 'expression'/'expr' column
      - id,expression per line (no header)
    """
    lines = text.splitlines()
    if not lines:
        return pd.DataFrame(columns=["filter_id", "expression"])

    hdr = lines[0].lower()
    header_like = ("," in lines[0]) and (
        ("expression" in hdr) or ("applicable" in hdr) or ("expr" in hdr)
    )

    if header_like:
        df = pd.read_csv(io.StringIO(text))
        # find expression column
        expr_col = None
        for c in df.columns:
            if c.lower() in ("expression", "expr"):
                expr_col = c
                break
        if expr_col is None:
            # look for applicable_if with non-boolean content
            for c in df.columns:
                if "applicable" in c.lower():
                    vals = set(str(x).strip().lower() for x in df[c].dropna().unique().tolist())
                    if not vals or vals.issubset(BOOLEAN_LITERALS):
                        continue
                    expr_col = c
                    break
        if expr_col is None:
            return pd.DataFrame(columns=["filter_id", "expression"])

        # id column
        id_col = None
        for cand in ("id", "filter_id", "name"):
            if cand in df.columns:
                id_col = cand
                break
        if id_col is None:
            id_col = df.columns[0]

        out_rows = []
        for _, row in df.iterrows():
            fid = str(row[id_col])
            raw = row[expr_col]
            expr = normalize_expression("" if pd.isna(raw) else str(raw))
            if expr:
                out_rows.append({"filter_id": fid, "expression": expr})
        return pd.DataFrame(out_rows)

    # simple id,expression (no header)
    out_rows = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",", 1)]
        if len(parts) == 2:
            fid, expr = parts[0], normalize_expression(parts[1])
            if expr:
                out_rows.append({"filter_id": fid, "expression": expr})
    return pd.DataFrame(out_rows)

# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="Reversed + Keepers CSV Builder", layout="wide")
st.title("🧰 Reversed + Keepers CSV Builder")

with st.sidebar:
    st.header("Inputs")
    st.caption("Upload **one** results file (either `reversal_candidates.csv/txt` OR full `filter_results.csv/txt`).")
    up_results = st.file_uploader("Results file", type=["csv", "txt"])

    st.caption("Upload the ORIGINAL filter batch file so we can pull canonical expressions.")
    up_filters = st.file_uploader("Original filters (Batch CSV/TXT or id,expression)", type=["csv", "txt"])

    st.divider()
    st.header("Options")
    include_variant_guard = st.checkbox("Include variant guard in expression", value=True,
                                        help="Adds (variant_name == '<variant>') AND ... to each expression.")
    use_comparator_flip = st.checkbox("Reverse by comparator flip (preferred)", value=True,
                                      help="If OFF, we will use not(...) wrapping instead.")

    st.caption("Keeper threshold (max elimination rate). Used only when a full summary is supplied.")
    keeper_max = st.number_input("Keeper max elimination rate", min_value=0.0, max_value=1.0, value=0.25, step=0.01)

    st.caption("Reversal threshold (min elimination rate). Used only when a full summary is supplied.")
    reverse_min = st.number_input("Reverse min elimination rate", min_value=0.0, max_value=1.0, value=0.75, step=0.01)

# Utility: repo fallback reader
def load_results_df() -> pd.DataFrame:
    if up_results is not None:
        return read_any_table(up_results)

    # Fallback to repo files
    for cand in ["reversal_candidates.csv", "reversal_candidates.txt", "filter_results.csv", "filter_results.txt"]:
        try:
            with open(cand, encoding="utf-8") as f:
                return read_any_table(io.StringIO(f.read()))
        except FileNotFoundError:
            continue
    st.error("Please upload a results file or add reversal_candidates.csv/txt or filter_results.csv/txt to repo.")
    st.stop()

def load_original_filters_df() -> pd.DataFrame:
    if up_filters is not None:
        raw = up_filters.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="ignore")
        return load_filters_any(raw)

    # fallback names
    for cand in ["test 4pwrballfilters.txt", "test_4pwrballfilters.txt", "filters.csv", "filters.txt"]:
        try:
            with open(cand, encoding="utf-8") as f:
                return load_filters_any(f.read())
        except FileNotFoundError:
            continue
    st.error("Please upload your original filter batch file (e.g., 'test 4pwrballfilters.txt').")
    st.stop()

go = st.button("Build CSVs")

if go:
    try:
        results_df = load_results_df()
        base_df = load_original_filters_df()

        # Minimal column normalization
        cols = {c.lower(): c for c in results_df.columns}
        # Try to detect which results type we got
        has_stat = ("stat" in cols)
        has_elim = ("eliminated" in cols) and ("total" in cols)
        has_expr = ("expression" in cols)
        has_variant = ("variant" in cols)
        has_filter_id = ("filter_id" in cols) or ("id" in cols) or ("name" in cols)

        if not has_variant:
            st.error("Results file must contain a 'variant' column.")
            st.stop()
        if not has_filter_id:
            st.error("Results file must contain a 'filter_id' (or 'id'/'name') column.")
            st.stop()

        # Canonical id col in results
        id_col = "filter_id" if "filter_id" in cols else ("id" if "id" in cols else "name")
        id_col_actual = cols[id_col]
        variant_col_actual = cols["variant"]

        # Ensure we have expressions in results; otherwise join from base
        if has_expr:
            expr_col_actual = cols["expression"]
            work_df = results_df[[id_col_actual, variant_col_actual, expr_col_actual]].copy()
            work_df.rename(columns={id_col_actual: "filter_id", variant_col_actual: "variant",
                                    expr_col_actual: "expression"}, inplace=True)
            work_df["expression"] = work_df["expression"].astype(str).map(normalize_expression)
        else:
            # join with base (id -> expression)
            base = base_df[["filter_id", "expression"]].copy()
            work_df = results_df[[id_col_actual, variant_col_actual]].copy()
            work_df.rename(columns={id_col_actual: "filter_id", variant_col_actual: "variant"}, inplace=True)
            work_df = work_df.merge(base, on="filter_id", how="left")
            work_df["expression"] = work_df["expression"].astype(str).map(normalize_expression)

        # Add eliminated/total if present
        if has_elim:
            work_df["eliminated"] = results_df[cols["eliminated"]]
            work_df["total"] = results_df[cols["total"]]
        else:
            # Try to parse from stat if available
            if has_stat:
                stcol = cols["stat"]
                tmp = results_df[stcol].astype(str).str.extract(r"(\d+)\s*/\s*(\d+)")
                work_df["eliminated"] = pd.to_numeric(tmp[0], errors="coerce")
                work_df["total"] = pd.to_numeric(tmp[1], errors="coerce")
            else:
                work_df["eliminated"] = pd.NA
                work_df["total"] = pd.NA

        # Normalize expression (again) to be safe
        work_df["expression"] = work_df["expression"].astype(str).map(normalize_expression)

        # If this file is already the "reversal_candidates" export, treat every row as a reversal.
        likely_reversal_input = ("threshold" in cols) and ("layman_explanation" in cols) and (
            not results_df.empty and results_df.shape[1] <= 10
        )

        # Partition into keepers vs reversals
        if likely_reversal_input:
            reversals = work_df.copy()
            keepers = pd.DataFrame(columns=work_df.columns)
        else:
            # Compute elimination rate from eliminated/total, if available
            if work_df["total"].notna().any():
                erate = pd.to_numeric(work_df["eliminated"], errors="coerce") / pd.to_numeric(work_df["total"], errors="coerce")
            else:
                erate = pd.Series([pd.NA]*len(work_df))

            # Reverse if rate >= reverse_min; keepers if rate <= keeper_max; drop flagged/NaNs
            reversals = work_df[erate >= reverse_min].copy()
            keepers = work_df[erate <= keeper_max].copy()

        # Build reversed rows
        reversed_rows = []
        for _, r in reversals.iterrows():
            fid = str(r["filter_id"])
            var = str(r["variant"])
            expr = str(r["expression"])

            if not expr:
                # If blank, skip
                continue

            # Reversal transform
            try:
                if use_comparator_flip:
                    rev_core = reverse_by_comparator_flip(expr)
                else:
                    rev_core = reverse_by_not_wrapper(expr)
            except Exception:
                # fallback if something odd happens
                rev_core = f"not({expr})"

            final_expr = rev_core
            if include_variant_guard:
                final_expr = f"{make_variant_guard(var)} and ({rev_core})"

            new_id = f"REV_{fid}_{var}"

            reversed_rows.append(build_output_row(
                new_id, var, final_expr,
                eliminated=int(r["eliminated"]) if pd.notna(r["eliminated"]) else None,
                total=int(r["total"]) if pd.notna(r["total"]) else None
            ))

        reversed_df = pd.DataFrame(reversed_rows)

        # Build keeper rows (original expressions, canonicalized)
        keeper_rows = []
        for _, r in keepers.iterrows():
            fid = str(r["filter_id"])
            var = str(r["variant"])
            expr = str(r["expression"])
            if not expr:
                continue

            final_expr = expr
            if include_variant_guard:
                final_expr = f"{make_variant_guard(var)} and ({expr})"

            keeper_rows.append(build_output_row(
                fid, var, final_expr,
                eliminated=int(r["eliminated"]) if pd.notna(r["eliminated"]) else None,
                total=int(r["total"]) if pd.notna(r["total"]) else None
            ))

        keepers_df = pd.DataFrame(keeper_rows)

        # ---------------------------
        # UI outputs + downloads
        # ---------------------------
        st.subheader("Reversed Filters")
        st.dataframe(reversed_df, use_container_width=True, height=280)
        if not reversed_df.empty:
            csv_b = reversed_df.to_csv(index=False).encode("utf-8")
            txt_b = reversed_df.to_csv(index=False, sep="\t").encode("utf-8")
            c1, c2 = st.columns(2)
            with c1:
                st.download_button("⬇️ Download reversed_filters.csv", csv_b,
                                   "reversed_filters.csv", "text/csv")
            with c2:
                st.download_button("⬇️ Download reversed_filters.txt", txt_b,
                                   "reversed_filters.txt", "text/plain")
        else:
            st.info("No reversed filters.")

        st.subheader("Keepers (No Reversal Needed)")
        st.dataframe(keepers_df, use_container_width=True, height=280)
        if not keepers_df.empty:
            csv_b = keepers_df.to_csv(index=False).encode("utf-8")
            txt_b = keepers_df.to_csv(index=False, sep="\t").encode("utf-8")
            c1, c2 = st.columns(2)
            with c1:
                st.download_button("⬇️ Download keepers_formatted.csv", csv_b,
                                   "keepers_formatted.csv", "text/csv")
            with c2:
                st.download_button("⬇️ Download keepers_formatted.txt", txt_b,
                                   "keepers_formatted.txt", "text/plain")
        else:
            st.info("No keepers found under current settings.")

        st.success(f"Done. Reversed: {len(reversed_df):,} | Keepers: {len(keepers_df):,}")

    except Exception as e:
        st.exception(e)
        st.stop()
