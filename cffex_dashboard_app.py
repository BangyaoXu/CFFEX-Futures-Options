# -*- coding: utf-8 -*-
"""
CFFEX Futures & Options Volatility Dashboard (Streamlit)
============================================================================

This version removes Streamlit uploaders and instead reads *.html/*.htm/*.mhtml
directly from the SAME FOLDER as this script (or any folder you choose).

Included fixes:
1) ✅ Robust table extraction: prefer pandas.read_html (colspan/rowspan-safe), fallback to BeautifulSoup.
2) ✅ Options multi-expiry within ONE table:
   - Detect marker rows like: 看涨  HO2602  看跌
   - Apply current expiry to numeric rows until next marker row.
3) ✅ Skew RR25/BF25 robust (Fix #2): nearest-delta within tolerance (no strict interpolation).

Install:
  pip install streamlit pandas numpy scipy plotly beautifulsoup4 lxml

Run:
  streamlit run cffex_app.py
Place your CFFEX html/mhtml files in the same folder as cffex_app.py
(or point the app to another folder in the sidebar).
"""

from __future__ import annotations

import re
import math
import datetime as dt
from pathlib import Path
from email import message_from_binary_file
from typing import List, Optional, Dict, Tuple
from io import StringIO

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from scipy.optimize import brentq
from bs4 import BeautifulSoup


# =========================
# USER MAPPING (ENFORCED)
# =========================
FUTURES_NAME_MAP = {
    "IF": "沪深300股指期货",
    "IM": "中证1000股指期货",
    "IH": "上证50股指期货",
}
OPTIONS_NAME_MAP = {
    "IO": "沪深300股指期权",
    "MO": "中证1000股指期权",
    "HO": "上证50股指期权",
}
FUT_PREFIXES = list(FUTURES_NAME_MAP.keys())
OPT_PREFIXES = list(OPTIONS_NAME_MAP.keys())


# =========================
# Utilities
# =========================
def _num(x) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip().replace(",", "")
    if s in ("", "-", "—", "–", "NaN", "nan", "None"):
        return None
    s = re.sub(r"[^0-9\.\-\+]", "", s)
    try:
        return float(s)
    except Exception:
        return None


def third_friday(year: int, month: int) -> dt.date:
    first = dt.date(year, month, 1)
    off = (4 - first.weekday()) % 7  # Friday=4
    return first + dt.timedelta(days=off + 14)


def infer_expiry_from_yymm(yymm: str) -> Optional[dt.date]:
    try:
        yy, mm = int(yymm[:2]), int(yymm[2:])
        return third_friday(2000 + yy, mm)
    except Exception:
        return None


def infer_expiry(sym: str) -> Optional[dt.date]:
    s = sym.upper().strip()
    m = re.search(r"(IF|IH|IM)(\d{4})", s)
    if m:
        return infer_expiry_from_yymm(m.group(2))
    m2 = re.search(r"(IO|HO|MO)(\d{4})", s)
    if m2:
        return infer_expiry_from_yymm(m2.group(2))
    return None


def yearfrac(today: dt.date, expiry: dt.date) -> float:
    return max((expiry - today).days, 0) / 365.0


def _find_col(cols: List[str], key: str) -> Optional[str]:
    return next((c for c in cols if key in c), None)


def pick_adjacent_or_nearest_latest(cols: List[str], strike_idx: int) -> Tuple[Optional[str], Optional[str]]:
    """
    CFFEX paired option table:
      CallPx = column immediately LEFT of 行权价 (prefer 最新价)
      PutPx  = column immediately RIGHT of 行权价 (prefer 最新价)
    Fallback to nearest 最新价 on each side if adjacency is not 最新价.
    """
    call_px_col = None
    put_px_col = None

    if strike_idx - 1 >= 0 and "最新价" in cols[strike_idx - 1]:
        call_px_col = cols[strike_idx - 1]
    if strike_idx + 1 < len(cols) and "最新价" in cols[strike_idx + 1]:
        put_px_col = cols[strike_idx + 1]

    if call_px_col is None:
        for j in range(strike_idx - 1, -1, -1):
            if "最新价" in cols[j]:
                call_px_col = cols[j]
                break

    if put_px_col is None:
        for j in range(strike_idx + 1, len(cols)):
            if "最新价" in cols[j]:
                put_px_col = cols[j]
                break

    return call_px_col, put_px_col


# =========================
# Header scoring (fallback path)
# =========================
def _score_header_row(row: List[str], want: str) -> int:
    r = [str(x).strip() for x in row]
    s = 0
    if want == "options":
        s += 50 if any("行权价" in x for x in r) else 0
        s += 30 if sum(("最新价" in x) for x in r) >= 2 else 0
        s += 10 if any("看涨" in x for x in r) else 0
        s += 10 if any("看跌" in x for x in r) else 0
        s += 5 if any("成交量" in x for x in r) else 0
        s += 5 if any("持仓量" in x for x in r) else 0
        s += 2 if any("涨跌" in x for x in r) else 0
    else:
        s += 40 if any("合约名称" in x for x in r) else 0
        s += 20 if any("最新价" in x for x in r) else 0
        s += 10 if any("收盘" in x for x in r) else 0
        s += 10 if any("结算" in x for x in r) else 0
        s += 5 if any("成交量" in x for x in r) else 0
        s += 5 if any("持仓量" in x for x in r) else 0
    return s


def _make_unique_cols(cols: List[str]) -> List[str]:
    out = []
    for i, c in enumerate(cols):
        name = str(c).strip()
        out.append(f"{name}__{i}" if name else f"__blank__{i}")
    return out


# =========================
# MHTML/HTML table extraction
# =========================
def extract_tables_from_mhtml_or_html(path: Path) -> List[pd.DataFrame]:
    html_parts: List[str] = []
    suffix = path.suffix.lower()

    if suffix == ".mhtml":
        with open(path, "rb") as f:
            msg = message_from_binary_file(f)
        for part in msg.walk():
            if part.get_content_type() == "text/html":
                payload = part.get_payload(decode=True)
                if payload:
                    html_parts.append(payload.decode(errors="ignore"))
    else:
        html_parts.append(path.read_text(errors="ignore"))

    tables_all: List[pd.DataFrame] = []

    for html in html_parts:
        # 1) Prefer pandas.read_html (handles colspan/rowspan)
        try:
            dfs = pd.read_html(StringIO(html))
            for df0 in dfs:
                if df0 is None or df0.empty:
                    continue
                df0 = df0.copy()

                df0.columns = _make_unique_cols([str(c) for c in df0.columns])

                # Keep rows with numbers OR key tokens (so we keep marker rows 看涨 HO2602 看跌)
                def _row_keep(sr: pd.Series) -> bool:
                    vals = ["" if pd.isna(x) else str(x).strip() for x in sr.values]
                    has_num = any(_num(x) is not None for x in vals)
                    has_key = any(any(k in v for k in ["行权价", "最新价", "合约名称", "看涨", "看跌"]) for v in vals)
                    return has_num or has_key

                df0 = df0[df0.apply(_row_keep, axis=1)].reset_index(drop=True)
                if len(df0) >= 3:
                    tables_all.append(df0)

            if tables_all:
                continue
        except Exception:
            pass

        # 2) Fallback: BeautifulSoup + header-row detect
        soup = BeautifulSoup(html, "lxml")
        for table in soup.find_all("table"):
            rows = []
            for tr in table.find_all("tr"):
                cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                if cells:
                    rows.append(cells)
            if len(rows) <= 2:
                continue

            maxc = max(len(r) for r in rows)
            padded = [r + [""] * (maxc - len(r)) for r in rows]

            N = min(4, len(padded))
            scored = []
            for i in range(N):
                scored.append((max(_score_header_row(padded[i], "options"),
                                   _score_header_row(padded[i], "futures")), i))
            scored.sort(reverse=True)
            header_idx = scored[0][1]

            header = padded[header_idx]
            data = padded[header_idx + 1 :]

            cleaned = []
            for r in data:
                has_num = any(_num(x) is not None for x in r)
                keep_marker = (not has_num) and (any("看涨" in str(x) for x in r) or any("看跌" in str(x) for x in r))
                if has_num or keep_marker:
                    cleaned.append(r)

            if len(cleaned) < 2:
                continue

            df = pd.DataFrame(cleaned, columns=_make_unique_cols(header))
            tables_all.append(df)

    return tables_all


def pick_best_table(tables: List[pd.DataFrame], pattern: str, want: str) -> Optional[pd.DataFrame]:
    best, best_score = None, -1
    pat = re.compile(pattern, re.I)

    for df in tables:
        if df is None or df.empty:
            continue

        flat = df.astype(str).stack()
        cols = list(df.columns)

        score = 0
        if flat.str.contains(pat).any():
            score += 10

        if want == "futures":
            if any("合约名称" in c for c in cols):
                score += 6
            if any(k in " ".join(cols) for k in ["最新价", "收盘", "结算"]):
                score += 4
        else:
            if any("行权价" in c for c in cols):
                score += 14
            if sum(("最新价" in c) for c in cols) >= 2:
                score += 12
            if any("看涨" in c for c in cols) or flat.str.contains("看涨").any():
                score += 2
            if any("看跌" in c for c in cols) or flat.str.contains("看跌").any():
                score += 2

        if df.shape[0] < 3:
            score -= 5

        if score > best_score:
            best, best_score = df, score

    return best


# =========================
# Futures parsing
# =========================
FUT_COLS = ["product", "product_name", "symbol", "expiry", "last", "volume", "oi"]

def parse_futures(df: Optional[pd.DataFrame], prefix: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=FUT_COLS)

    cols = list(df.columns)
    sym_col = _find_col(cols, "合约名称") or cols[0]

    last_col = (
        _find_col(cols, "最新价")
        or _find_col(cols, "收盘价")
        or _find_col(cols, "结算价")
        or _find_col(cols, "收盘")
        or _find_col(cols, "结算")
        or (cols[1] if len(cols) > 1 else cols[0])
    )

    vol_col = _find_col(cols, "成交量")
    oi_col = _find_col(cols, "持仓量")

    out = []
    for _, r in df.iterrows():
        sym = str(r.get(sym_col, "")).strip().upper()
        if not re.match(rf"^{prefix}\d{{4}}$", sym):
            continue
        expiry = infer_expiry(sym)
        last = _num(r.get(last_col))
        if expiry is None or last is None:
            continue
        out.append({
            "product": prefix,
            "product_name": FUTURES_NAME_MAP[prefix],
            "symbol": sym,
            "expiry": expiry,
            "last": float(last),
            "volume": _num(r.get(vol_col)) if vol_col else None,
            "oi": _num(r.get(oi_col)) if oi_col else None,
        })

    if not out:
        return pd.DataFrame(columns=FUT_COLS)

    return pd.DataFrame(out).sort_values("expiry").reset_index(drop=True)


# =========================
# Options parsing (MULTI-EXPIRY WITHIN ONE TABLE)
# =========================
OPT_COLS = ["product", "product_name", "expiry", "cp", "K", "price", "volume", "oi", "symbol"]

def _extract_yymm_marker_from_row(values: List[str], prefix: str) -> Optional[str]:
    joined = " ".join(values).upper()
    if ("看涨" in joined) and ("看跌" in joined):
        m = re.search(rf"{prefix}(\d{{4}})", joined, re.I)
        if m:
            return m.group(1)
    return None


def parse_options_paired_cn_mode(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=OPT_COLS)

    cols = list(df.columns)
    strike_col = _find_col(cols, "行权价")
    if strike_col is None:
        return pd.DataFrame(columns=OPT_COLS)

    strike_idx = cols.index(strike_col)
    call_price_col, put_price_col = pick_adjacent_or_nearest_latest(cols, strike_idx)
    if call_price_col is None or put_price_col is None:
        return pd.DataFrame(columns=OPT_COLS)

    # Optional: volumes / OI
    call_side_cols = cols[:strike_idx]
    put_side_cols = cols[strike_idx + 1 :]

    call_vol_col = next((c for c in call_side_cols[::-1] if "成交量" in c), None)
    call_oi_col  = next((c for c in call_side_cols[::-1] if "持仓量" in c), None)
    put_vol_col  = next((c for c in put_side_cols if "成交量" in c), None)
    put_oi_col   = next((c for c in put_side_cols if "持仓量" in c), None)

    out = []
    current_yymm: Optional[str] = None
    current_expiry: Optional[dt.date] = None

    for _, row in df.iterrows():
        vals = ["" if pd.isna(x) else str(x).strip() for x in row.values]

        # marker row (expiry selector)
        yymm = _extract_yymm_marker_from_row(vals, prefix)
        if yymm:
            current_yymm = yymm
            current_expiry = infer_expiry(prefix + yymm)
            continue

        # data rows require current marker
        if current_yymm is None or current_expiry is None:
            continue

        K = _num(row.get(strike_col))
        if K is None:
            continue

        call_px = _num(row.get(call_price_col))
        put_px = _num(row.get(put_price_col))

        # poison guard (old bug symptom)
        if call_px is not None and abs(call_px - K) < 1e-12:
            call_px = None
        if put_px is not None and abs(put_px - K) < 1e-12:
            put_px = None

        if call_px is not None and call_px > 0:
            out.append({
                "product": prefix,
                "product_name": OPTIONS_NAME_MAP[prefix],
                "expiry": current_expiry,
                "cp": "C",
                "K": float(K),
                "price": float(call_px),
                "volume": _num(row.get(call_vol_col)) if call_vol_col else None,
                "oi": _num(row.get(call_oi_col)) if call_oi_col else None,
                "symbol": f"{prefix}{current_yymm}-C-{int(K)}",
            })

        if put_px is not None and put_px > 0:
            out.append({
                "product": prefix,
                "product_name": OPTIONS_NAME_MAP[prefix],
                "expiry": current_expiry,
                "cp": "P",
                "K": float(K),
                "price": float(put_px),
                "volume": _num(row.get(put_vol_col)) if put_vol_col else None,
                "oi": _num(row.get(put_oi_col)) if put_oi_col else None,
                "symbol": f"{prefix}{current_yymm}-P-{int(K)}",
            })

    if not out:
        return pd.DataFrame(columns=OPT_COLS)

    return pd.DataFrame(out).sort_values(["expiry", "K", "cp"]).reset_index(drop=True)


def parse_options(df: Optional[pd.DataFrame], prefix: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=OPT_COLS)
    return parse_options_paired_cn_mode(df, prefix)


# =========================
# Option math
# =========================
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price_forward(F: float, K: float, T: float, r: float, vol: float, is_call: bool) -> float:
    if T <= 0 or vol <= 0 or F <= 0 or K <= 0:
        return 0.0
    df = math.exp(-r * T)
    sig = vol * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * vol * vol * T) / sig
    d2 = d1 - sig
    if is_call:
        return df * (F * norm_cdf(d1) - K * norm_cdf(d2))
    return df * (K * norm_cdf(-d2) - F * norm_cdf(-d1))


def implied_forward_from_parity(C: float, P: float, K: float, T: float, r: float) -> float:
    return K + (C - P) * math.exp(r * T)


def implied_vol(price: float, F: float, K: float, T: float, r: float, is_call: bool) -> Optional[float]:
    if price is None or price <= 0 or T <= 0 or F <= 0 or K <= 0:
        return None

    df = math.exp(-r * T)
    intrinsic = df * max((F - K) if is_call else (K - F), 0.0)
    if price < intrinsic - 1e-8:
        return None

    def f(v):
        return bs_price_forward(F, K, T, r, v, is_call) - price

    try:
        return float(brentq(f, 1e-4, 3.0, maxiter=200))
    except Exception:
        return None


def delta_forward(F: float, K: float, T: float, vol: float, is_call: bool) -> Optional[float]:
    if T <= 0 or vol is None or vol <= 0 or F <= 0 or K <= 0:
        return None
    sig = vol * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * vol * vol * T) / sig
    if is_call:
        return norm_cdf(d1)
    return norm_cdf(d1) - 1.0


# =========================
# ✅ Fix 2: Robust RR/BF via nearest delta (no strict interpolation)
# =========================
def rr_bf_25d(ivdf: pd.DataFrame, tol: float = 0.08) -> Optional[Dict[str, float]]:
    """
    Robust RR/BF:
    - pick nearest delta point to +0.25 on calls and -0.25 on puts
    - require within tol (default 0.08)
    - ATM proxy = nearest call delta to 0.50 (within tol)
    """
    d = ivdf.dropna(subset=["iv", "F", "T"]).copy()
    if d.empty:
        return None

    d["is_call"] = d["cp"].eq("C")
    d["delta"] = d.apply(lambda r: delta_forward(r["F"], r["K"], r["T"], r["iv"], bool(r["is_call"])), axis=1)
    d = d.dropna(subset=["delta"])
    if d.empty:
        return None

    calls = d[d["cp"] == "C"].copy()
    puts  = d[d["cp"] == "P"].copy()
    if calls.empty or puts.empty:
        return None

    def nearest(subdf: pd.DataFrame, target: float) -> Optional[float]:
        subdf = subdf.copy()
        subdf["dist"] = (subdf["delta"] - target).abs()
        best = subdf.sort_values("dist").iloc[0]
        if float(best["dist"]) > tol:
            return None
        return float(best["iv"])

    iv_c25 = nearest(calls, 0.25)
    iv_p25 = nearest(puts, -0.25)
    iv_atm = nearest(calls, 0.50)

    if iv_c25 is None or iv_p25 is None or iv_atm is None:
        return None

    rr25 = iv_c25 - iv_p25
    bf25 = 0.5 * (iv_c25 + iv_p25) - iv_atm
    return {"IV_25C": iv_c25, "IV_25P": iv_p25, "IV_ATM": iv_atm, "RR25": rr25, "BF25": bf25}


# =========================
# Robust forward from parity
# =========================
def compute_robust_forward_from_parity(s: pd.DataFrame, T: float, r: float) -> Tuple[Optional[float], pd.DataFrame, str]:
    calls = s[s["cp"] == "C"][["K", "price"]].rename(columns={"price": "C"})
    puts  = s[s["cp"] == "P"][["K", "price"]].rename(columns={"price": "P"})
    par = pd.merge(calls, puts, on="K", how="inner").dropna()
    if par.empty:
        return None, par, "No C/P pairs"

    par["F_impl"] = par.apply(lambda row: implied_forward_from_parity(row["C"], row["P"], row["K"], T, r), axis=1)
    par = par.dropna(subset=["F_impl"]).sort_values("K")
    if par.empty:
        return None, par, "No valid F_impl"

    par = par[par["F_impl"] > 0].copy()
    if par.empty:
        return None, par, "All F_impl <= 0 (bad prices)"

    med = float(par["F_impl"].median())
    par = par[(par["F_impl"] > 0.5 * med) & (par["F_impl"] < 1.5 * med)].copy()
    if par.empty:
        return None, par, "All filtered as outliers (bad prices)"

    lo = int(len(par) * 0.3)
    hi = int(len(par) * 0.7)
    core = par.iloc[lo:hi] if hi > lo else par
    F = float(core["F_impl"].median())

    note = ""
    if len(core) < 5:
        note = "Too few parity strikes after filtering."
    return F, par, note


# =========================
# Local-folder file discovery
# =========================
def list_local_files(folder: Path) -> List[Path]:
    exts = (".html", ".htm", ".mhtml")
    files: List[Path] = []
    for ext in exts:
        files.extend(folder.glob(f"*{ext}"))
        files.extend(folder.glob(f"*{ext.upper()}"))
    return sorted(set(files))


def file_text_contains_any(path: Path, needles: List[str], max_bytes: int = 300_000) -> bool:
    """Lightweight sniff to classify files without fully parsing (fast)."""
    try:
        b = path.read_bytes()[:max_bytes]
        s = b.decode(errors="ignore").upper()
        return any(n.upper() in s for n in needles)
    except Exception:
        return False


def split_fut_opt(files: List[Path]) -> Tuple[List[Path], List[Path]]:
    fut_files, opt_files = [], []
    for p in files:
        # classify by content sniffing
        if file_text_contains_any(p, [r"合约名称", "IF", "IH", "IM"]) and file_text_contains_any(p, ["IF", "IH", "IM"]):
            fut_files.append(p)
        if file_text_contains_any(p, ["行权价", "看涨", "看跌", "IO", "HO", "MO"]):
            opt_files.append(p)
    return fut_files, opt_files


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="CFFEX Vol Dashboard", layout="wide")
st.title("CFFEX Futures & Options — Volatility Trading Dashboard")

SCRIPT_DIR = Path(__file__).resolve().parent

with st.sidebar:
    st.header("Inputs")
    today = st.date_input("Valuation date", dt.date.today())
    r = st.number_input("Risk-free rate r (cont.)", value=0.02, step=0.005, format="%.4f")

    folder_str = st.text_input("Data folder (default: script folder)", value=str(SCRIPT_DIR))
    data_dir = Path(folder_str).expanduser().resolve()

    show_debug = st.checkbox("Show debug", value=False)
    tol = st.slider("RR/BF nearest-delta tolerance", min_value=0.02, max_value=0.20, value=0.08, step=0.01)

    if st.button("Reload files"):
        st.rerun()

if not data_dir.exists() or not data_dir.is_dir():
    st.error(f"Folder not found: {data_dir}")
    st.stop()

all_files = list_local_files(data_dir)
if not all_files:
    st.warning(f"No .html/.htm/.mhtml files found in: {data_dir}")
    st.stop()

fut_paths, opt_paths = split_fut_opt(all_files)

st.caption(f"Reading files from: {data_dir}")
with st.expander("Detected files"):
    st.write("All files:", [p.name for p in all_files])
    st.write("Futures candidates:", [p.name for p in fut_paths])
    st.write("Options candidates:", [p.name for p in opt_paths])

if not fut_paths or not opt_paths:
    st.error("Need BOTH futures and options files in the folder (or your sniff rules didn't match).")
    st.stop()

# ---- Futures
futures_all: List[pd.DataFrame] = []
fut_debug = []

for p in fut_paths:
    tabs = extract_tables_from_mhtml_or_html(p)

    for prefix in FUT_PREFIXES:
        best = pick_best_table(tabs, rf"{prefix}\d{{4}}", want="futures")
        parsed = parse_futures(best, prefix)

        if show_debug:
            fut_debug.append((p.name, prefix, None if best is None else list(best.columns),
                              None if best is None else best.head(8)))

        if not parsed.empty:
            futures_all.append(parsed)

if not futures_all:
    st.error("No futures contracts parsed.")
    if show_debug:
        st.subheader("Debug: Futures picked tables")
        for fname, prefix, cols_preview, head in fut_debug:
            st.markdown(f"**{fname} | {prefix}**")
            st.write("Columns:", cols_preview)
            if head is not None:
                st.dataframe(head, use_container_width=True)
    st.stop()

futures_df = pd.concat(futures_all, ignore_index=True).drop_duplicates()

# ---- Options
options_all: List[pd.DataFrame] = []
opt_debug: List[Dict[str, object]] = []

for p in opt_paths:
    tabs = extract_tables_from_mhtml_or_html(p)

    for prefix in OPT_PREFIXES:
        best = pick_best_table(tabs, rf"{prefix}\d{{4}}", want="options")
        parsed = parse_options(best, prefix)

        if show_debug:
            diag: Dict[str, object] = {"file": p.name, "prefix": prefix, "picked": best is not None}
            if best is not None and not best.empty:
                cols = list(best.columns)
                strike_col = _find_col(cols, "行权价")
                if strike_col is not None:
                    k_idx = cols.index(strike_col)
                    call_px_col, put_px_col = pick_adjacent_or_nearest_latest(cols, k_idx)
                else:
                    k_idx, call_px_col, put_px_col = None, None, None
                diag.update({
                    "strike_col": strike_col,
                    "strike_idx": k_idx,
                    "call_px_col": call_px_col,
                    "put_px_col": put_px_col,
                    "cols": cols,
                })
                diag["head20"] = best.head(20)
            opt_debug.append(diag)

        if not parsed.empty:
            options_all.append(parsed)

if not options_all:
    st.error("No options contracts parsed.")
    if show_debug:
        st.subheader("Debug: Options picked tables")
        for d in opt_debug:
            st.markdown(f"**{d.get('file')} | {d.get('prefix')}**")
            st.write({k: d.get(k) for k in ["strike_col", "strike_idx", "call_px_col", "put_px_col"]})
            if d.get("head20") is not None:
                st.dataframe(d["head20"], use_container_width=True)
    st.stop()

options_df = pd.concat(options_all, ignore_index=True).drop_duplicates()

# ---- Sanity: remove price==K poison
options_df = options_df.dropna(subset=["price"]).copy()
options_df = options_df[~(np.abs(options_df["price"] - options_df["K"]) < 1e-12)].copy()

if options_df.empty:
    st.error("All options filtered out by sanity checks (price==K). Parsing still wrong.")
    st.stop()


# =========================
# 1) Futures curve
# =========================
st.subheader("1) Futures term structure")
for pfx in sorted(futures_df["product"].unique()):
    cname = FUTURES_NAME_MAP.get(pfx, pfx)
    sub = futures_df[futures_df["product"] == pfx].sort_values("expiry").copy()
    st.markdown(f"### {cname} ({pfx})")
    st.plotly_chart(
        px.line(sub, x="expiry", y="last", markers=True, title=f"{cname} — Futures curve"),
        use_container_width=True
    )
    st.dataframe(sub, use_container_width=True)


# =========================
# 2) Options analytics
# =========================
st.subheader("2) Options analytics (Smile / Forward / Surface / Skew / Carry & Roll)")

for pfx in sorted(options_df["product"].unique()):
    cname = OPTIONS_NAME_MAP.get(pfx, pfx)
    st.markdown(f"## {cname} ({pfx})")

    prod = options_df[options_df["product"] == pfx].copy()
    expiries = sorted(prod["expiry"].unique())

    fwd_rows, atm_rows, skew_rows, surf_rows = [], [], [], []

    for expiry in expiries:
        s = prod[prod["expiry"] == expiry].copy()
        T = yearfrac(today, expiry)
        if T <= 0:
            continue

        F, par_dbg, note = compute_robust_forward_from_parity(s[["cp", "K", "price"]].copy(), T, r)
        if F is None:
            if show_debug:
                st.warning(f"{cname} {expiry}: forward not computed ({note})")
            continue

        fwd_rows.append({"expiry": expiry, "T": T, "F": F})
        if note:
            st.warning(f"{cname} {expiry}: {note}")

        s["T"] = T
        s["F"] = F
        s["is_call"] = s["cp"].eq("C")
        s["iv"] = s.apply(lambda row: implied_vol(row["price"], F, row["K"], T, r, bool(row["is_call"])), axis=1)
        ivdf = s.dropna(subset=["iv"]).copy()

        if not ivdf.empty:
            st.plotly_chart(
                px.line(ivdf, x="K", y="iv", color="cp", markers=True,
                        title=f"{cname} — IV Smile | Expiry {expiry} | F≈{F:.2f}"),
                use_container_width=True
            )

            atm_iv = ivdf[(ivdf["K"] >= 0.98 * F) & (ivdf["K"] <= 1.02 * F)]["iv"].median()
            if pd.notna(atm_iv):
                atm_rows.append({"expiry": expiry, "T": T, "ATM_IV": float(atm_iv)})

            rr = rr_bf_25d(ivdf[["cp", "K", "iv", "F", "T"]], tol=float(tol))
            if rr:
                skew_rows.append({"expiry": expiry, "T": T, "RR25": rr["RR25"], "BF25": rr["BF25"], "IV_ATM_proxy": rr["IV_ATM"]})

            ivdf["m"] = ivdf["K"] / F
            ivdf = ivdf[(ivdf["m"] >= 0.8) & (ivdf["m"] <= 1.2)]
            for _, rr0 in ivdf.iterrows():
                surf_rows.append({"expiry": expiry, "T": T, "m": rr0["m"], "iv": rr0["iv"]})

    fwd_df = pd.DataFrame(fwd_rows, columns=["expiry", "T", "F"])
    if not fwd_df.empty:
        fwd_df = fwd_df.sort_values("expiry")
        st.markdown(f"### {cname} — Implied forward curve")
        st.plotly_chart(px.line(fwd_df, x="expiry", y="F", markers=True, title="Implied Forward vs Expiry"), use_container_width=True)
        st.dataframe(fwd_df, use_container_width=True)

        st.markdown(f"### {cname} — Forward carry & roll (adjacent expiries)")
        fwd_df2 = fwd_df.sort_values("T").reset_index(drop=True)
        rows = []
        for i in range(len(fwd_df2) - 1):
            e1, e2 = fwd_df2.loc[i, "expiry"], fwd_df2.loc[i + 1, "expiry"]
            T1, T2 = float(fwd_df2.loc[i, "T"]), float(fwd_df2.loc[i + 1, "T"])
            F1, F2 = float(fwd_df2.loc[i, "F"]), float(fwd_df2.loc[i + 1, "F"])
            dT = max(T2 - T1, 1e-9)
            roll_pct = (F2 / F1 - 1.0) * 100.0
            carry_ann = ((F2 / F1) ** (1.0 / dT) - 1.0) * 100.0
            rows.append({"roll_from": str(e1), "roll_to": str(e2), "F_from": F1, "F_to": F2, "ΔT_years": dT,
                         "roll_%": roll_pct, "carry_annualized_%": carry_ann})
        carry_df = pd.DataFrame(rows)
        if not carry_df.empty:
            st.dataframe(carry_df, use_container_width=True)
            st.plotly_chart(px.bar(carry_df, x="roll_to", y="carry_annualized_%", title="Carry (annualized) by next expiry"),
                            use_container_width=True)
    else:
        st.info("Forward curve unavailable (missing valid C/P parity pairs).")

    atm_df = pd.DataFrame(atm_rows, columns=["expiry", "T", "ATM_IV"])
    if not atm_df.empty:
        atm_df = atm_df.sort_values("expiry")
        st.markdown(f"### {cname} — ATM IV term structure")
        st.plotly_chart(px.line(atm_df, x="expiry", y="ATM_IV", markers=True, title="ATM IV vs Expiry"), use_container_width=True)
        st.dataframe(atm_df, use_container_width=True)
    else:
        st.info("ATM IV term structure unavailable.")

    skew_df = pd.DataFrame(skew_rows, columns=["expiry", "T", "RR25", "BF25", "IV_ATM_proxy"])
    if not skew_df.empty:
        skew_df = skew_df.sort_values("expiry")
        st.markdown(f"### {cname} — Skew term structure (RR25 / BF25)")
        st.plotly_chart(px.line(skew_df, x="expiry", y="RR25", markers=True, title="RR25 vs Expiry"), use_container_width=True)
        st.plotly_chart(px.line(skew_df, x="expiry", y="BF25", markers=True, title="BF25 vs Expiry"), use_container_width=True)
        st.dataframe(skew_df, use_container_width=True)
    else:
        st.info("RR25/BF25 unavailable (insufficient strikes/IV near ±25d).")

    surf_df = pd.DataFrame(surf_rows, columns=["expiry", "T", "m", "iv"])
    if not surf_df.empty:
        st.markdown(f"### {cname} — IV Surface (Expiry × Moneyness K/F)")
        bins = np.linspace(0.8, 1.2, 17)
        surf_df["m_bucket"] = pd.cut(surf_df["m"], bins=bins, include_lowest=True)
        piv = surf_df.groupby(["expiry", "m_bucket"])["iv"].median().reset_index()
        piv["m_mid"] = piv["m_bucket"].apply(lambda x: float((x.left + x.right) / 2.0))
        heat = piv.pivot(index="expiry", columns="m_mid", values="iv").sort_index()

        st.plotly_chart(px.imshow(heat, aspect="auto", title="IV Surface (median IV by moneyness bucket)",
                                  labels={"x": "Moneyness (K/F)", "y": "Expiry", "color": "IV"}),
                        use_container_width=True)
    else:
        st.info("IV surface unavailable (not enough solved IV).")


if show_debug:
    st.subheader("Debug")
    st.write("Options parsed preview (first 200 rows)")
    st.dataframe(options_df.head(200), use_container_width=True)
    st.write("Unique expiries by product:")
    st.write(options_df.groupby("product")["expiry"].nunique())
    st.write(options_df.groupby("product")["expiry"].unique())
