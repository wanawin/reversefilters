# make_reversed_op_filters.py
# Build "operator-reversed" filters for your working app.

import re
import pandas as pd

# --------- Config ---------
IN_FILE  = "reversal_candidates.csv"   # table you downloaded from the app
OUT_FILE = "reversed_op_filters.csv"   # file to upload back into the app
SCOPE_BY_VARIANT = True                # prepend (variant_name == "possum1") AND ...
# --------------------------

# Convert "Eliminate if ..." layman text into the pythonic condition your app understands.
def layman_to_expr(s: str) -> str:
    if not isinstance(s, str) or not s.strip():
        return ""
    s = re.sub(r'^\s*Eliminate\s+if\s+', '', s, flags=re.IGNORECASE).strip()

    # normalize tokens used by your app
    repl = {
        "winner value": "winner",
        "seed value":   "seed",
        "combo_structure": "winner_structure",  # legacy alias
        " is ≤ ": " <= ", " is ≥ ": " >= ", " is < ": " < ", " is > ": " > ",
        "≤": "<=", "≥": ">=",
        " OR ": " or ", " AND ": " and ",
        "abs(winner value - seed value)": "abs(winner - seed)",
        "seed digits": "seed_digits",
        "seed value_digits": "seed_digits",
        "combo digits": "combo_digits",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Flip each comparison operator WITHOUT wrapping in not(...).
# We do placeholder passes so <= isn't partly replaced by <, etc.
def flip_comparators(expr: str) -> str:
    if not expr:
        return expr
    t = expr

    # Placeholders first (order matters)
    t = t.replace(">=", "§GE§")
    t = t.replace("<=", "§LE§")
    t = t.replace("==", "§EQ§")
    t = t.replace("!=", "§NE§")
    # Singles
    # Use regex to avoid touching arrows or shifts (not in your rules, but be safe).
    t = re.sub(r"(?<![<>=!])<(?![<>=])", "§LT§", t)
    t = re.sub(r"(?<![<>=!])>(?![<>=])", "§GT§", t)

    # Inverse mapping
    mapping = {
        "§LE§": ">",   # <= becomes >
        "§LT§": ">=",  # <  becomes >=
        "§GE§": "<",   # >= becomes <
        "§GT§": "<=",  # >  becomes <=
        "§EQ§": "!=",  # == becomes !=
        "§NE§": "==",  # != becomes ==
    }
    for ph, op in mapping.items():
        t = t.replace(ph, op)
    return t

def pick_source_expression(row: pd.Series) -> str:
    # Prefer a real 'expression' column if your CSV has it; otherwise fall back to layman.
    if "expression" in row and isinstance(row["expression"], str) and row["expression"].strip():
        return row["expression"].strip()
    if "layman_explanation" in row:
        return layman_to_expr(str(row["layman_explanation"]))
    # Some exports call it 'layman' or similar
    for alt in ("layman", "layman_text"):
        if alt in row and isinstance(row[alt], str):
            return layman_to_expr(row[alt])
    return ""

def build_row(row: pd.Series) -> dict | None:
    fid     = str(row.get("filter_id", "")).strip() or str(row.get("id", "")).strip()
    variant = str(row.get("variant", "")).strip() or "full"

    src_expr = pick_source_expression(row)
    if not src_expr:
        return None

    flipped = flip_comparators(src_expr)

    # Scope to variant if desired, so each reversed rule is tested “variant to itself”
    if SCOPE_BY_VARIANT and variant:
        final_expr = f'(variant_name == "{variant}") and ({flipped})'
    else:
        final_expr = f"({flipped})"

    out_id = f"REVOP_{fid}_{variant}"
    return {
        "id": out_id,
        "expression": final_expr,
        # keep extra columns for traceability (your app only needs id+expression)
        "source_filter_id": fid,
        "variant": variant,
        "source_expression": src_expr,
        "flipped_expression": flipped,
    }

def main():
    df = pd.read_csv(IN_FILE)
    out = []
    for _, r in df.iterrows():
        built = build_row(r)
        if built:
            out.append(built)

    out_df = pd.DataFrame(out, columns=[
        "id","expression","source_filter_id","variant","source_expression","flipped_expression"
    ])
    out_df.to_csv(OUT_FILE, index=False)
    print(f"Done. Wrote {len(out_df):,} operator-reversed filters to {OUT_FILE}")

if __name__ == "__main__":
    main()
