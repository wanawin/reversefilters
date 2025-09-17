# make_reversed_and_keepers.py
# One app to:
#  1) Build reversed filters (when you ask for reversal),
#  2) Build formatted keepers (no reversal), and
#  3) Let you download multiple files without auto-reset.

from __future__ import annotations
import io, re
from typing import Dict, Any, Optional
import pandas as pd
import streamlit as st

# ---------------------------
# Normalization & layman text
# ---------------------------
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
    (r"≤", "<="), (r"≥", ">="), (r"≠", "!="), (r"–", "-"),
]
BOOLEAN_LITERALS = {"true","false","1","0","yes","no"}

def is_boolean_literal(s: str) -> bool:
    return s.strip().lower() in BOOLEAN_LITERALS

def normalize_expression(expr: str) -> str:
    if not isinstance(expr, str): return ""
    e = expr.strip()
    if not e: return ""
    e = e.strip("\"'")
    low = e.lower()
    if "see prior" in low or "see conversation" in low: return ""
    for pat,repl in LEGACY_MAP:
        e = re.sub(pat, repl, e, flags=re.IGNORECASE)
    e = re.sub(r"\bAND\b","and",e); e = re.sub(r"\bOR\b","or",e)
    if is_boolean_literal(e): return ""
    return e

def layman_explanation(expr: str) -> str:
    if not expr: return "Eliminate if <missing expression>"
    r = {
        "seed":"seed value","winner":"winner value","==":"equals","<=":"is ≤",
        ">=":"is ≥","<":"is <",">":"is >"," and ":" AND "," or ":" OR ",
    }
    t = expr
    for k,v in r.items(): t = t.replace(k,v)
    return f"Eliminate if {t}"

# ---------------------------
# IO helpers
# ---------------------------
def read_any_table(file_or_str) -> pd.DataFrame:
    if hasattr(file_or_str, "read"):
        raw = file_or_str.read()
        if isinstance(raw, bytes): raw = raw.decode("utf-8", errors="ignore")
        buf = io.StringIO(raw)
    else:
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

def load_filters_any(text: str) -> pd.DataFrame:
    lines = text.splitlines()
    if not lines: return pd.DataFrame(columns=["filter_id","expression"])
    hdr = lines[0].lower()
    header_like = ("," in lines[0]) and (("expression" in hdr) or ("applicable" in hdr) or ("expr" in hdr))
    if header_like:
        df = pd.read_csv(io.StringIO(text))
        expr_col = None
        for c in df.columns:
            if c.lower() in ("expression","expr"): expr_col = c; break
        if expr_col is None:
            for c in df.columns:
                if "applicable" in c.lower():
                    vals = set(str(x).strip().lower() for x in df[c].dropna().unique().tolist())
                    if not vals or vals.issubset(BOOLEAN_LITERALS): continue
                    expr_col = c; break
        if expr_col is None: return pd.DataFrame(columns=["filter_id","expression"])
        id_col = None
        for cand in ("id","filter_id","name"):
            if cand in df.columns: id_col = cand; break
        if id_col is None: id_col = df.columns[0]
        out=[]
        for _,row in df.iterrows():
            fid = str(row[id_col])
            raw = row[expr_col]
            expr = normalize_expression("" if pd.isna(raw) else str(raw))
            if expr: out.append({"filter_id":fid,"expression":expr})
        return pd.DataFrame(out)
    # id,expression per line
    out=[]
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"): continue
        parts = [p.strip() for p in line.split(",",1)]
        if len(parts)==2:
            fid, expr = parts[0], normalize_expression(parts[1])
            if expr: out.append({"filter_id":fid,"expression":expr})
    return pd.DataFrame(out)

# ---------------------------
# Reversal transforms
# ---------------------------
def reverse_by_comparator_flip(expr: str) -> str:
    if not expr: return ""
    e = expr
    # protect multi-char ops first
    repl = {"<=":"§LE§", ">=":"§GE§", "==":"§EQ§", "!=":"§NE§"}
    for k,v in repl.items(): e = e.replace(k,v)
    # single-char
    e = e.replace("<","§LT§").replace(">","§GT§")
    # flip
    flip = {"§LE§":">","§LT§":">=","§GE§":"<","§GT§":"<=","§EQ§":"!=","§NE§":"=="}
    for k,v in flip.items(): e = e.replace(k,v)
    return e

def reverse_by_not_wrapper(expr: str) -> str:
    e = expr.strip()
    if e.startswith("not(") and e.endswith(")"): return e[4:-1].strip()
    return f"not({e})"

def make_variant_guard(variant: str, on: bool) -> str:
    return f"(variant_name == '{variant}') and " if on and variant else ""

# ---------------------------
# Builder
# ---------------------------
def build_output_row(fid: str, var: str, expr: str,
                     eliminated: Optional[int]=None, total: Optional[int]=None,
                     extras: Optional[Dict[str,Any]]=None) -> Dict[str,Any]:
    stat = ""
    if eliminated is not None and total is not None and total>0:
        stat = f"{eliminated}/{total}"
    row = {
        "filter_id": fid, "variant": var or "",
        "expression": expr,
        "layman_explanation": layman_explanation(expr),
        "stat": stat
    }
    if extras: row.update(extras)
    return row

# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="Reversed + Keepers CSV Builder", layout="wide")
st.title("🧰 Reversed + Keepers CSV Builder")

with st.sidebar:
    mode = st.radio("Mode", ["Format only (no reversals)", "Reverse only", "Auto by thresholds"], index=0,
                    help="Format-only never reverses anything. Reverse-only reverses everything. Auto uses thresholds.")
    include_variant_guard = st.checkbox("Include variant guard", value=True)
    use_comparator_flip   = st.checkbox("Reverse by comparator flip (preferred)", value=True,
                                        help="If OFF, use not(...) wrapping.")
    keeper_max = st.slider("Keeper max elimination rate (Auto mode)", 0.0, 1.0, 0.25, 0.01)
    reverse_min = st.slider("Reverse min elimination rate (Auto mode)", 0.0, 1.0, 0.75, 0.01)

    st.divider()
    st.caption("Upload ONE results file and the ORIGINAL filter batch file.")
    up_results = st.file_uploader("Results (reversal_candidates.csv / filter_results.csv / txt)", type=["csv","txt"])
    up_filters = st.file_uploader("Original filters (Batch CSV/TXT or id,expression)", type=["csv","txt"])
    st.caption("Tip: if you use ‘Format only’, you can give a curated results file (just id/variant/expression).")

# persist results so downloads don't reset the app
if "outputs" not in st.session_state: st.session_state.outputs = None

def repo_fallback_results() -> pd.DataFrame:
    for cand in ["reversal_candidates.csv","reversal_candidates.txt","filter_results.csv","filter_results.txt"]:
        try:
            with open(cand, encoding="utf-8") as f: return read_any_table(io.StringIO(f.read()))
        except FileNotFoundError:
            continue
    return pd.DataFrame()

def repo_fallback_filters() -> pd.DataFrame:
    for cand in ["test 4pwrballfilters.txt","test_4pwrballfilters.txt","filters.csv","filters.txt"]:
        try:
            with open(cand, encoding="utf-8") as f: return load_filters_any(f.read())
        except FileNotFoundError:
            continue
    return pd.DataFrame()

def capture_bytes(up):
    if up is None: return None
    b = up.getvalue() if hasattr(up,"getvalue") else up.read()
    return b if isinstance(b, bytes) else str(b).encode("utf-8")

go = st.button("Build CSVs")
reset = st.button("Reset")

if reset:
    st.session_state.outputs = None
    st.experimental_rerun()

if go:
    try:
        # Snapshotted inputs (so downloads won't clear them)
        res_bytes = capture_bytes(up_results)
        fil_bytes = capture_bytes(up_filters)

        # Load results DF
        if res_bytes:
            results_df = read_any_table(io.StringIO(res_bytes.decode("utf-8","ignore")))
        else:
            results_df = repo_fallback_results()
        if results_df.empty:
            st.error("Please upload a results file or place one in repo.")
            st.stop()

        # Load original filters (for canonical expressions if needed)
        if fil_bytes:
            base_df = load_filters_any(fil_bytes.decode("utf-8","ignore"))
        else:
            base_df = repo_fallback_filters()
        if base_df.empty and mode != "Format only (no reversals)":
            # For reversal/auto on summary-only files we can proceed without base,
            # but it's better to have it if results lack expression.
            pass

        # Column normalization
        cols = {c.lower(): c for c in results_df.columns}
        has_variant   = ("variant" in cols)
        has_filter_id = ("filter_id" in cols) or ("id" in cols) or ("name" in cols)
        has_expr      = ("expression" in cols)
        has_elim      = ("eliminated" in cols) and ("total" in cols)
        has_stat      = ("stat" in cols)

        if not has_filter_id:
            st.error("Results file must have a filter_id (or id/name) column."); st.stop()
        if not has_variant:
            st.warning("No 'variant' column found. Variant guard will be skipped for all rows.")

        id_col = "filter_id" if "filter_id" in cols else ("id" if "id" in cols else "name")
        id_col_actual = cols[id_col]
        var_col_actual = cols["variant"] if has_variant else None

        # Build working DF with id/variant/expression (+ elim/total if present)
        if has_expr:
            expr_col_actual = cols["expression"]
            work = results_df[[id_col_actual] + ([var_col_actual] if var_col_actual else []) + [expr_col_actual]].copy()
            rename_map = {id_col_actual:"filter_id", expr_col_actual:"expression"}
            if var_col_actual: rename_map[var_col_actual] = "variant"
            work.rename(columns=rename_map, inplace=True)
            work["expression"] = work["expression"].astype(str).map(normalize_expression)
        else:
            # Join with base filters to retrieve expression
            base = base_df[["filter_id","expression"]] if "expression" in base_df.columns else base_df
            base = base.copy()
            base["expression"] = base["expression"].astype(str).map(normalize_expression)
            work = results_df[[id_col_actual] + ([var_col_actual] if var_col_actual else [])].copy()
            work.rename(columns={id_col_actual:"filter_id", **({var_col_actual:"variant"} if var_col_actual else {})}, inplace=True)
            work = work.merge(base, on="filter_id", how="left")
            work["expression"] = work["expression"].astype(str).map(normalize_expression)

        # add elim/total if present or derivable
        if has_elim:
            work["eliminated"] = pd.to_numeric(results_df[cols["eliminated"]], errors="coerce")
            work["total"]      = pd.to_numeric(results_df[cols["total"]], errors="coerce")
        elif has_stat:
            tmp = results_df[cols["stat"]].astype(str).str.extract(r"(\d+)\s*/\s*(\d+)")
            work["eliminated"] = pd.to_numeric(tmp[0], errors="coerce")
            work["total"]      = pd.to_numeric(tmp[1], errors="coerce")
        else:
            work["eliminated"] = pd.NA
            work["total"]      = pd.PNA

        # Mode behavior
        reversed_rows, formatted_rows = [], []

        # Helper: add guarded expression
        def guarded(var, expr):
            guard = make_variant_guard(var, include_variant_guard)
            return f"{guard}({expr})" if guard else f"({expr})"

        if mode == "Format only (no reversals)":
            # Do NOT reverse anything—just normalize & format
            for _, r in work.iterrows():
                fid = str(r["filter_id"])
                var = (str(r["variant"]) if "variant" in r and pd.notna(r["variant"]) else "")
                expr = str(r["expression"])
                if not expr: continue
                final_expr = guarded(var, expr)
                formatted_rows.append(build_output_row(fid, var, final_expr,
                                                       int(r["eliminated"]) if pd.notna(r["eliminated"]) else None,
                                                       int(r["total"]) if pd.notna(r["total"]) else None))
            reversed_df = pd.DataFrame(columns=["filter_id","variant","expression","layman_explanation","stat"])
            formatted_df = pd.DataFrame(formatted_rows)

        elif mode == "Reverse only":
            for _, r in work.iterrows():
                fid = str(r["filter_id"])
                var = (str(r["variant"]) if "variant" in r and pd.notna(r["variant"]) else "")
                expr = str(r["expression"])
                if not expr: continue
                rev_core = reverse_by_comparator_flip(expr) if use_comparator_flip else reverse_by_not_wrapper(expr)
                final_expr = guarded(var, rev_core)
                new_id = f"REV_{fid}_{var}" if var else f"REV_{fid}"
                reversed_rows.append(build_output_row(new_id, var, final_expr,
                                                      int(r["eliminated"]) if pd.notna(r["eliminated"]) else None,
                                                      int(r["total"]) if pd.notna(r["total"]) else None))
            reversed_df = pd.DataFrame(reversed_rows)
            formatted_df = pd.DataFrame(columns=["filter_id","variant","expression","layman_explanation","stat"])

        else:  # Auto by thresholds
            er = (pd.to_numeric(work["eliminated"], errors="coerce") /
                  pd.to_numeric(work["total"], errors="coerce"))
            for idx, r in work.iterrows():
                fid = str(r["filter_id"])
                var = (str(r["variant"]) if "variant" in r and pd.notna(r["variant"]) else "")
                expr = str(r["expression"])
                if not expr: continue
                e_rate = float(er.iloc[idx]) if pd.notna(er.iloc[idx]) else None
                if (e_rate is not None) and (e_rate >= reverse_min):
                    rev_core = reverse_by_comparator_flip(expr) if use_comparator_flip else reverse_by_not_wrapper(expr)
                    final_expr = guarded(var, rev_core)
                    new_id = f"REV_{fid}_{var}" if var else f"REV_{fid}"
                    reversed_rows.append(build_output_row(new_id, var, final_expr,
                                                          int(r["eliminated"]) if pd.notna(r["eliminated"]) else None,
                                                          int(r["total"]) if pd.notna(r["total"]) else None))
                elif (e_rate is not None) and (e_rate <= keeper_max):
                    final_expr = guarded(var, expr)
                    formatted_rows.append(build_output_row(fid, var, final_expr,
                                                           int(r["eliminated"]) if pd.notna(r["eliminated"]) else None,
                                                           int(r["total"]) if pd.notna(r["total"]) else None))
                else:
                    # in-between: ignore (neither reversed nor keeper)
                    pass
            reversed_df  = pd.DataFrame(reversed_rows)
            formatted_df = pd.DataFrame(formatted_rows)

        # Persist outputs so downloads don't reset anything
        st.session_state.outputs = {
            "reversed": reversed_df, "formatted": formatted_df
        }
        st.success(f"Built CSVs — Reversed: {len(reversed_df):,} | Formatted (keepers): {len(formatted_df):,}")

    except Exception as e:
        st.exception(e)
        st.stop()

# ---------------------------
# Display + Downloads (persisted)
# ---------------------------
outs = st.session_state.outputs
if outs is not None:
    rev_df = outs["reversed"]; fmt_df = outs["formatted"]

    st.subheader("Reversed Filters")
    st.dataframe(rev_df, use_container_width=True, height=280)
    if not rev_df.empty:
        c1,c2 = st.columns(2)
        with c1:
            st.download_button("⬇️ Download reversed_filters.csv",
                               rev_df.to_csv(index=False).encode("utf-8"),
                               "reversed_filters.csv","text/csv")
        with c2:
            st.download_button("⬇️ Download reversed_filters.txt",
                               rev_df.to_csv(index=False, sep="\t").encode("utf-8"),
                               "reversed_filters.txt","text/plain")

    st.subheader("Keepers (Formatted, no reversal)")
    st.dataframe(fmt_df, use_container_width=True, height=280)
    if not fmt_df.empty:
        c1,c2 = st.columns(2)
        with c1:
            st.download_button("⬇️ Download keepers_formatted.csv",
                               fmt_df.to_csv(index=False).encode("utf-8"),
                               "keepers_formatted.csv","text/csv")
        with c2:
            st.download_button("⬇️ Download keepers_formatted.txt",
                               fmt_df.to_csv(index=False, sep="\t").encode("utf-8"),
                               "keepers_formatted.txt","text/plain")
