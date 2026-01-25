# -*- coding: utf-8 -*-
"""
CFFEX Futures & Options Dashboard (Streamlit)
============================================================================

Install:
  pip install streamlit pandas numpy scipy plotly beautifulsoup4 lxml requests

Run:
  streamlit run cffex_app.py

Place your CFFEX html/mhtml files in the same folder as cffex_app.py
(or point the app to another folder in the sidebar).

Notes:
- Adds ETF spot fetching (Sina quote) + uses spot+futures term structure for an "excess carry" style tilt.
- Keeps RR25/BF25/ATM/Surface logic the same as your current version.
- Signal panel is collected at the end of the GUI with Chinese names + ETF ticker/name.
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
import requests


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

# Options product -> matching futures product
OPT_TO_FUT = {"IO": "IF", "HO": "IH", "MO": "IM"}

# Optional English display names
OPTIONS_NAME_EN = {
    "IO": "CSI 300 Index Options",
    "MO": "CSI 1000 Index Options",
    "HO": "SSE 50 Index Options",
}
FUTURES_NAME_EN = {
    "IF": "CSI 300 Index Futures",
    "IM": "CSI 1000 Index Futures",
    "IH": "SSE 50 Index Futures",
}

# Corresponding ETF spot instruments (edit to your tradable choices)
ETF_MAP = {
    "IO": {"name_cn": "沪深300 ETF", "ticker": "510300.SH"},
    "MO": {"name_cn": "中证1000 ETF", "ticker": "159845.SZ"},
    "HO": {"name_cn": "上证50 ETF", "ticker": "510050.SH"},
}


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
                scored.append(
                    (max(_score_header_row(padded[i], "options"), _score_header_row(padded[i], "futures")), i)
                )
            scored.sort(reverse=True)
            header_idx = scored[0][1]

            header = padded[header_idx]
            data = padded[header_idx + 1 :]

            cleaned = []
            for r in data:
                has_num = any(_num(x) is not None for x in r)
                keep_marker = (not has_num) and (
                    any("看涨" in str(x) for x in r) or any("看跌" in str(x) for x in r)
                )
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
        out.append(
            {
                "product": prefix,
                "product_name": FUTURES_NAME_MAP[prefix],
                "symbol": sym,
                "expiry": expiry,
                "last": float(last),
                "volume": _num(r.get(vol_col)) if vol_col else None,
                "oi": _num(r.get(oi_col)) if oi_col else None,
            }
        )

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
    call_oi_col = next((c for c in call_side_cols[::-1] if "持仓量" in c), None)
    put_vol_col = next((c for c in put_side_cols if "成交量" in c), None)
    put_oi_col = next((c for c in put_side_cols if "持仓量" in c), None)

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
            out.append(
                {
                    "product": prefix,
                    "product_name": OPTIONS_NAME_MAP[prefix],
                    "expiry": current_expiry,
                    "cp": "C",
                    "K": float(K),
                    "price": float(call_px),
                    "volume": _num(row.get(call_vol_col)) if call_vol_col else None,
                    "oi": _num(row.get(call_oi_col)) if call_oi_col else None,
                    "symbol": f"{prefix}{current_yymm}-C-{int(K)}",
                }
            )

        if put_px is not None and put_px > 0:
            out.append(
                {
                    "product": prefix,
                    "product_name": OPTIONS_NAME_MAP[prefix],
                    "expiry": current_expiry,
                    "cp": "P",
                    "K": float(K),
                    "price": float(put_px),
                    "volume": _num(row.get(put_vol_col)) if put_vol_col else None,
                    "oi": _num(row.get(put_oi_col)) if put_oi_col else None,
                    "symbol": f"{prefix}{current_yymm}-P-{int(K)}",
                }
            )

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


def rr_bf_25d(ivdf: pd.DataFrame, tol: float = 0.08) -> Optional[Dict[str, float]]:
    """
    Robust RR/BF:
    - pick nearest delta point to +0.25 on calls and -0.25 on puts
    - require within tol
    - ATM proxy = nearest call delta to 0.50
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
    puts = d[d["cp"] == "P"].copy()
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


def compute_robust_forward_from_parity(s: pd.DataFrame, T: float, r: float) -> Tuple[Optional[float], pd.DataFrame, str]:
    calls = s[s["cp"] == "C"][["K", "price"]].rename(columns={"price": "C"})
    puts = s[s["cp"] == "P"][["K", "price"]].rename(columns={"price": "P"})
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
    try:
        b = path.read_bytes()[:max_bytes]
        s = b.decode(errors="ignore").upper()
        return any(n.upper() in s for n in needles)
    except Exception:
        return False


def split_fut_opt(files: List[Path]) -> Tuple[List[Path], List[Path]]:
    fut_files, opt_files = [], []
    for p in files:
        if file_text_contains_any(p, [r"合约名称", "IF", "IH", "IM"]) and file_text_contains_any(p, ["IF", "IH", "IM"]):
            fut_files.append(p)
        if file_text_contains_any(p, ["行权价", "看涨", "看跌", "IO", "HO", "MO"]):
            opt_files.append(p)
    return fut_files, opt_files


# =========================
# ETF Spot Fetch (Sina quote)
# =========================
def _ticker_to_sina_code(ticker: str) -> Optional[str]:
    """
    Input: '510300.SH' or '159845.SZ'
    Output: 'sh510300' or 'sz159845'
    """
    if not ticker:
        return None
    t = ticker.strip().upper()
    m = re.match(r"^(\d{6})\.(SH|SZ)$", t)
    if not m:
        return None
    code, ex = m.group(1), m.group(2)
    return ("sh" if ex == "SH" else "sz") + code


@st.cache_data(ttl=60, show_spinner=False)
def fetch_spot_sina(ticker: str) -> Dict[str, object]:
    """
    Uses: http://hq.sinajs.cn/list=sh510300
    Response is GBK text: var hq_str_sh510300="...,...,最新价,...,日期,时间";
    We parse:
      name = fields[0]
      last = fields[3]
      date = fields[30], time=fields[31] (common layout)
    If parsing fails, returns ok=False.
    """
    code = _ticker_to_sina_code(ticker)
    if code is None:
        return {"ok": False, "err": "Bad ticker format (expect 510300.SH / 159845.SZ)", "ticker": ticker}

    url = f"http://hq.sinajs.cn/list={code}"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://finance.sina.com.cn",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=5)
        resp.encoding = "gbk"
        txt = resp.text.strip()
        # var hq_str_sh510300="...";
        m = re.search(r'="(.*)";', txt)
        if not m:
            return {"ok": False, "err": "No payload", "ticker": ticker, "raw": txt[:120]}
        payload = m.group(1)
        if payload.strip() == "":
            return {"ok": False, "err": "Empty payload", "ticker": ticker}

        fields = payload.split(",")
        if len(fields) < 5:
            return {"ok": False, "err": f"Too few fields ({len(fields)})", "ticker": ticker}

        name = fields[0].strip()
        last = _num(fields[3])
        # Some variants have date/time near the end; try common spots
        d_str = fields[30].strip() if len(fields) > 30 else ""
        t_str = fields[31].strip() if len(fields) > 31 else ""
        ts = (d_str + " " + t_str).strip() if d_str and t_str else ""

        if last is None:
            return {"ok": False, "err": "Last price parse failed", "ticker": ticker, "name": name}

        return {"ok": True, "ticker": ticker, "name": name, "last": float(last), "ts": ts, "source": "Sina"}
    except Exception as e:
        return {"ok": False, "err": str(e), "ticker": ticker}


# =========================
# ETF Spot Signal Panel (summary at end)
# =========================
def _as_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if np.isnan(v):
            return None
        return v
    except Exception:
        return None


def _pick_front_expiry(df: pd.DataFrame, col: str = "expiry") -> Optional[dt.date]:
    if df is None or df.empty:
        return None
    xs = sorted(df[col].dropna().unique())
    return xs[0] if xs else None


def _sigmoid_to_0_100(x: float) -> int:
    y = 1.0 / (1.0 + math.exp(-x))
    return int(round(100.0 * y))


def _surface_horizon_from_points(surf_points: Optional[pd.DataFrame]) -> str:
    if surf_points is None or surf_points.empty:
        return "3–10 trading days"

    hi = surf_points.dropna(subset=["iv", "expiry"]).sort_values("iv", ascending=False).head(30)
    if hi.empty:
        return "3–10 trading days"

    e_mode = hi["expiry"].mode()
    e_star = e_mode.iloc[0] if len(e_mode) else None
    if e_star is None:
        return "3–10 trading days"

    expiries_sorted = sorted(surf_points["expiry"].dropna().unique())
    if not expiries_sorted:
        return "3–10 trading days"

    front = expiries_sorted[0]
    if e_star == front:
        return "1–5 trading days (event / front-expiry dominated)"

    if len(expiries_sorted) >= 2 and e_star == expiries_sorted[1]:
        return "3–10 trading days"

    return "2–6 weeks (mid/long-expiry dominated)"


def _implied_rate_between(f1: float, t1: float, f2: float, t2: float) -> Optional[float]:
    if f1 <= 0 or f2 <= 0:
        return None
    dt_ = t2 - t1
    if dt_ <= 1e-9:
        return None
    return math.log(f2 / f1) / dt_


def etf_spot_signal_panel(
    *,
    futures_sub: Optional[pd.DataFrame],
    spot_px: Optional[float],
    spot_ts: str,
    atm_term: Optional[pd.DataFrame],
    skew_term: Optional[pd.DataFrame],
    surf_points: Optional[pd.DataFrame],
    r: float,
    q: float = 0.0,  # dividend yield assumption (you asked: assume zero dividend -> default 0)
) -> Dict[str, object]:
    """
    Informational ETF spot tilt from derivatives-implied pricing.

    Key change vs your previous "contango bearish/backwardation bullish":
    - For equity index (q≈0), *upward* futures term structure can be normal due to financing (r).
    - We therefore look at "excess carry" vs (r-q) rather than slope sign alone.
    """
    score = 0.0
    drivers: List[str] = []
    metrics: Dict[str, Optional[float]] = {}

    # ---- Spot
    metrics["spot_px"] = spot_px
    if spot_px is not None:
        drivers.append(f"ETF现货价：{spot_px:.4f}" + (f"（{spot_ts}）" if spot_ts else ""))
    else:
        drivers.append("ETF现货价不可得（抓取失败/未提供）。")

    # ---- Futures curve: implied rate vs r-q
    r_impl_curve = None
    if futures_sub is not None and not futures_sub.empty and futures_sub.shape[0] >= 2:
        s = futures_sub.sort_values("expiry").copy()
        # Convert expiries to year fractions relative to "today" later (we don't have today here),
        # so we store in metrics as "curve_slope_raw" using first/last levels as a proxy,
        # and we compute an implied rate using sequential expiries in the main loop (below).
        # Here we compute r_impl between front and far using actual expiries passed via futures_sub already sorted later.
        # We'll compute T using calendar days from "today" outside and pass in as columns if desired.
        # For robustness, approximate T from days-to-expiry if provided; else skip.
        pass

    # We compute implied rate using futures_sub if it has a "T" column; otherwise compute using expiry with today later.
    # To keep your app minimal, we compute it here using expiry vs valuation date indirectly by st.session_state.
    # We'll rely on a global "today" captured via closure in Streamlit. (set below)
    today0 = st.session_state.get("__VAL_DATE__", None)
    if today0 is not None and futures_sub is not None and not futures_sub.empty and futures_sub.shape[0] >= 2:
        s = futures_sub.sort_values("expiry").copy()
        s["T"] = s["expiry"].apply(lambda e: yearfrac(today0, e))
        s = s[s["T"] > 0].copy()
        if s.shape[0] >= 2:
            f1 = _as_float(s["last"].iloc[0])
            fN = _as_float(s["last"].iloc[-1])
            t1 = _as_float(s["T"].iloc[0])
            tN = _as_float(s["T"].iloc[-1])
            if f1 is not None and fN is not None and t1 is not None and tN is not None:
                r_impl_curve = _implied_rate_between(f1, t1, fN, tN)

    metrics["r_impl_curve"] = r_impl_curve

    # Excess carry vs (r-q)
    carry_excess = None if r_impl_curve is None else (r_impl_curve - (r - q))
    metrics["carry_excess"] = carry_excess

    # Interpretation: only act if materially different from financing
    ex_th = 0.02  # 2%/yr threshold
    if r_impl_curve is None:
        drivers.append("期货曲线隐含利率不可得（合约不足/到期信息不足）。")
    else:
        drivers.append(f"期货曲线隐含年化carry≈{r_impl_curve*100:.2f}%；基准(r-q)≈{(r-q)*100:.2f}%")
        if carry_excess is not None:
            if carry_excess > ex_th:
                score += 0.6
                drivers.append(f"Carry偏高（相对基准 +{carry_excess*100:.2f}%/yr）：可能反映风险偏好/多头需求 ↑")
            elif carry_excess < -ex_th:
                score -= 0.8
                drivers.append(f"Carry偏低（相对基准 {carry_excess*100:.2f}%/yr）：可能反映贴水/对冲压力 ↑")
            else:
                drivers.append(f"Carry接近基准（偏差 {carry_excess*100:.2f}%/yr）：对方向信息弱")

    # ---- Options-implied forward vs spot (front expiry) (informational basis)
    fwd_basis = None
    front_F = None
    front_T = None
    if surf_points is not None and not surf_points.empty:
        # surf_points created from ivdf rows; does not include F directly. So we can't use it.
        # Instead derive from atm_term/skew_term? Not available. We'll compute basis outside and pass via metrics in caller.
        pass

    # We'll accept optional injection via session_state:
    front_F = st.session_state.get("__FRONT_OPT_F__", None)
    front_T = st.session_state.get("__FRONT_OPT_T__", None)

    if spot_px is not None and front_F is not None and front_T is not None and front_T > 0:
        theo_F = float(spot_px) * math.exp((r - q) * float(front_T))
        fwd_basis = float(front_F) / theo_F - 1.0
        metrics["front_opt_F"] = float(front_F)
        metrics["front_opt_T"] = float(front_T)
        metrics["fwd_basis"] = float(fwd_basis)

        b_th = 0.003  # 0.3%
        drivers.append(f"近月期权隐含F≈{front_F:.2f}；以现货推导理论F≈{theo_F:.2f}（q={q:.2%}）")
        if fwd_basis > b_th:
            score += 0.35
            drivers.append(f"隐含F偏高（+{fwd_basis*100:.2f}%）：偏risk-on/多头需求线索")
        elif fwd_basis < -b_th:
            score -= 0.35
            drivers.append(f"隐含F偏低（{fwd_basis*100:.2f}%）：偏risk-off/对冲线索")
        else:
            drivers.append(f"隐含F与理论F接近（{fwd_basis*100:.2f}%）：方向信息弱")
    else:
        drivers.append("期权隐含F vs 现货基差不可得（缺少现货或近月F/T）。")

    # ---- Front-expiry RR25/BF25 (unchanged)
    rr25 = bf25 = None
    if skew_term is not None and not skew_term.empty:
        front_e = _pick_front_expiry(skew_term, "expiry")
        if front_e is not None:
            row = skew_term[skew_term["expiry"] == front_e].iloc[0]
            rr25 = _as_float(row.get("RR25"))
            bf25 = _as_float(row.get("BF25"))
    metrics["RR25"] = rr25
    metrics["BF25"] = bf25

    rr_th = 0.005  # 0.5 vol point if IV is decimal
    bf_th = 0.005

    if rr25 is not None:
        if rr25 <= -rr_th:
            score -= 1.8
            drivers.append(f"RR25为负（put更贵/偏保护）：RR25≈{rr25*100:.2f} vol pts → 风险偏谨慎")
        elif rr25 >= rr_th:
            score += 1.8
            drivers.append(f"RR25为正（call更贵/偏上行）：RR25≈{rr25*100:.2f} vol pts → 风险偏积极")
        else:
            drivers.append(f"RR25接近平：RR25≈{rr25*100:.2f} vol pts")
    else:
        drivers.append("RR25不可得（±25Δ附近缺少可用IV）。")

    if bf25 is not None:
        if bf25 >= bf_th:
            score -= 1.0
            drivers.append(f"BF25偏高（两翼更贵/尾部风险定价高）：BF25≈{bf25*100:.2f} vol pts")
        elif bf25 <= -bf_th:
            score += 0.7
            drivers.append(f"BF25偏低（两翼更便宜/偏carry）：BF25≈{bf25*100:.2f} vol pts")
        else:
            drivers.append(f"BF25中性：BF25≈{bf25*100:.2f} vol pts")
    else:
        drivers.append("BF25不可得。")

    # ---- Front-expiry ATM IV (unchanged)
    atm_iv = None
    if atm_term is not None and not atm_term.empty:
        front_e = _pick_front_expiry(atm_term, "expiry")
        if front_e is not None:
            row = atm_term[atm_term["expiry"] == front_e].iloc[0]
            atm_iv = _as_float(row.get("ATM_IV"))
    metrics["ATM_IV"] = atm_iv

    if atm_iv is not None:
        if atm_iv >= 0.25:
            score -= 0.8
            drivers.append(f"近月ATM IV偏高：~{atm_iv*100:.2f}%（波动预期高/偏防守）")
        elif atm_iv <= 0.15:
            score += 0.5
            drivers.append(f"近月ATM IV偏低：~{atm_iv*100:.2f}%（稳定/偏风险）")
        else:
            drivers.append(f"近月ATM IV中等：~{atm_iv*100:.2f}%")
    else:
        drivers.append("ATM IV不可得。")

    # ---- Horizon via IV surface concentration (unchanged)
    horizon = _surface_horizon_from_points(surf_points)

    # ---- Bias mapping
    if score >= 1.5:
        bias = "做多 ETF 现货 (LONG / risk-on)"
    elif score <= -1.5:
        bias = "做空/降低仓位 (SHORT/UNDERWEIGHT / risk-off)"
    else:
        bias = "观望/中性 (NEUTRAL)"

    confidence = _sigmoid_to_0_100(1.2 * score)

    return {
        "bias": bias,
        "horizon": horizon,
        "confidence": confidence,
        "score": score,
        "drivers": drivers,
        "metrics": metrics,
    }


def render_etf_spot_panel_row(
    *,
    opt_pfx: str,
    opt_cn: str,
    opt_en: str,
    fut_pfx: Optional[str],
    fut_cn: str,
    fut_en: str,
    etf_name_cn: str,
    etf_ticker: str,
    spot_name: str,
    sig: Dict[str, object],
) -> None:
    st.markdown(
        f"""
        <div class="etf-panel">
          <h3>📌 {etf_name_cn} ({etf_ticker})</h3>
          <div class="subtitle">
            对应期权: <b>{opt_cn}</b> ({opt_pfx}) <span class="smallcap">/ {opt_en}</span>
            ｜ 对应期货: {fut_cn} ({fut_pfx or "N/A"}) <span class="smallcap">/ {fut_en if fut_pfx else ""}</span>
            {'｜ 现货标的: ' + spot_name if spot_name else ''}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([2.4, 2.0, 1.3])
    with c1:
        st.markdown('<div class="etf-panel kpi-label">方向 / Bias</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="etf-panel kpi">{sig["bias"]}</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="etf-panel kpi-label">持有周期 / Horizon</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="etf-panel kpi">{sig["horizon"]}</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="etf-panel kpi-label">置信度 / Confidence</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="etf-panel kpi">{sig["confidence"]}/100</div>', unsafe_allow_html=True)

    with st.expander("驱动因素 & 指标 (Drivers & metrics)", expanded=False):
        for d in sig.get("drivers", []):
            st.markdown(f'<div class="etf-panel driver">- {d}</div>', unsafe_allow_html=True)

        m = sig.get("metrics", {})
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown('<div class="etf-panel metricbox"><b>Key metrics:</b></div>', unsafe_allow_html=True)
        st.write(
            {
                "spot_px": None if m.get("spot_px") is None else round(float(m["spot_px"]), 4),
                "r_impl_curve_%": None if m.get("r_impl_curve") is None else round(100 * float(m["r_impl_curve"]), 3),
                "carry_excess_%": None if m.get("carry_excess") is None else round(100 * float(m["carry_excess"]), 3),
                "front_opt_F": None if m.get("front_opt_F") is None else round(float(m["front_opt_F"]), 3),
                "fwd_basis_%": None if m.get("fwd_basis") is None else round(100 * float(m["fwd_basis"]), 3),
                "RR25_vol_pts": None if m.get("RR25") is None else round(100 * float(m["RR25"]), 3),
                "BF25_vol_pts": None if m.get("BF25") is None else round(100 * float(m["BF25"]), 3),
                "front_ATM_IV_%": None if m.get("ATM_IV") is None else round(100 * float(m["ATM_IV"]), 2),
                "raw_score": round(float(sig.get("score", 0.0)), 3),
            }
        )

        st.markdown(
            '<div class="etf-panel smallcap">注：基于衍生品隐含信息的提示性信号（教育/研究用途），不构成投资建议。</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="CFFEX Dashboard", layout="wide")
st.title("CFFEX Futures & Options Dashboard")

# --- Global CSS for ETF signal panel fonts ---
st.markdown(
    """
    <style>
      .etf-panel h3 { margin: 0.25rem 0 0.5rem 0; font-size: 1.05rem; }
      .etf-panel .subtitle { color: #666; font-size: 0.85rem; margin-bottom: 0.5rem; }
      .etf-panel .kpi { font-size: 0.92rem; font-weight: 650; }
      .etf-panel .kpi-label { font-size: 0.78rem; color: #666; }
      .etf-panel .driver { font-size: 0.86rem; line-height: 1.25rem; }
      .etf-panel .metricbox { font-size: 0.82rem; }
      .etf-panel .smallcap { font-size: 0.78rem; color: #777; }
    </style>
    """,
    unsafe_allow_html=True,
)

SCRIPT_DIR = Path(__file__).resolve().parent

with st.sidebar:
    st.header("Inputs")
    today = st.date_input("Valuation date", dt.date.today())
    r = st.number_input("Risk-free rate r (cont.)", value=0.02, step=0.005, format="%.4f")
    q = st.number_input("Dividend yield q (assumed)", value=0.0000, step=0.0025, format="%.4f")  # default 0

    folder_str = st.text_input("Data folder (default: script folder)", value=str(SCRIPT_DIR))
    data_dir = Path(folder_str).expanduser().resolve()

    fetch_spot = st.checkbox("Fetch ETF spot price (Sina)", value=True)
    show_debug = st.checkbox("Show debug", value=False)
    tol = st.slider("RR/BF nearest-delta tolerance", min_value=0.02, max_value=0.20, value=0.08, step=0.01)

    if st.button("Reload files"):
        st.rerun()

# store valuation date so signal function can compute futures T
st.session_state["__VAL_DATE__"] = today

if not data_dir.exists() or not data_dir.is_dir():
    st.error(f"Folder not found: {data_dir}")
    st.stop()

all_files = list_local_files(data_dir)
if not all_files:
    st.warning(f"No .html/.htm/.mhtml files found in: {data_dir}")
    st.stop()

fut_paths, opt_paths = split_fut_opt(all_files)

st.caption(f"Reading files from: {data_dir}")
if show_debug:
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
            fut_debug.append(
                (p.name, prefix, None if best is None else list(best.columns), None if best is None else best.head(8))
            )

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
                diag.update(
                    {"strike_col": strike_col, "strike_idx": k_idx, "call_px_col": call_px_col, "put_px_col": put_px_col, "cols": cols}
                )
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
    st.plotly_chart(px.line(sub, x="expiry", y="last", markers=True, title=f"{cname} — Futures curve"), use_container_width=True)
    st.dataframe(sub, use_container_width=True)


# =========================
# 2) Options analytics
# =========================
st.subheader("2) Options analytics (Smile / Forward / Surface / Skew / Carry & Roll)")

signals_summary: List[Dict[str, object]] = []

for pfx in sorted(options_df["product"].unique()):
    cname = OPTIONS_NAME_MAP.get(pfx, pfx)
    st.markdown(f"## {cname} ({pfx})")

    prod = options_df[options_df["product"] == pfx].copy()
    expiries = sorted(prod["expiry"].unique())

    fwd_rows, atm_rows, skew_rows, surf_rows = [], [], [], []
    front_opt_F: Optional[float] = None
    front_opt_T: Optional[float] = None

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

        # capture front-expiry forward for the ETF panel (to compute F-vs-spot basis)
        if front_opt_F is None:
            front_opt_F = float(F)
            front_opt_T = float(T)

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
                px.line(ivdf, x="K", y="iv", color="cp", markers=True, title=f"{cname} — IV Smile | Expiry {expiry} | F≈{F:.2f}"),
                use_container_width=True,
            )

            atm_iv = ivdf[(ivdf["K"] >= 0.98 * F) & (ivdf["K"] <= 1.02 * F)]["iv"].median()
            if pd.notna(atm_iv):
                atm_rows.append({"expiry": expiry, "T": T, "ATM_IV": float(atm_iv)})

            rr = rr_bf_25d(ivdf[["cp", "K", "iv", "F", "T"]], tol=float(tol))
            if rr:
                skew_rows.append({"expiry": expiry, "T": T, "RR25": rr["RR25"], "BF25": rr["BF25"], "IV_ATM_proxy": rr["IV_ATM"]})

            ivdf["m"] = ivdf["K"] / F
            ivdf2 = ivdf[(ivdf["m"] >= 0.8) & (ivdf["m"] <= 1.2)].copy()
            for _, rr0 in ivdf2.iterrows():
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
            rows.append(
                {"roll_from": str(e1), "roll_to": str(e2), "F_from": F1, "F_to": F2, "ΔT_years": dT, "roll_%": roll_pct, "carry_annualized_%": carry_ann}
            )
        carry_df = pd.DataFrame(rows)
        if not carry_df.empty:
            st.dataframe(carry_df, use_container_width=True)
            st.plotly_chart(px.bar(carry_df, x="roll_to", y="carry_annualized_%", title="Carry (annualized) by next expiry"), use_container_width=True)
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

        st.plotly_chart(
            px.imshow(heat, aspect="auto", title="IV Surface (median IV by moneyness bucket)", labels={"x": "Moneyness (K/F)", "y": "Expiry", "color": "IV"}),
            use_container_width=True,
        )
    else:
        st.info("IV surface unavailable (not enough solved IV).")

    # ---- ETF spot fetch
    etf_info = ETF_MAP.get(pfx, {"name_cn": f"{cname} 对应ETF", "ticker": "TBD"})
    etf_ticker = etf_info.get("ticker", "TBD")
    spot_px = None
    spot_name = ""
    spot_ts = ""
    spot_note = ""

    if fetch_spot and etf_ticker != "TBD":
        spot = fetch_spot_sina(etf_ticker)
        if spot.get("ok"):
            spot_px = float(spot["last"])
            spot_name = str(spot.get("name", ""))
            spot_ts = str(spot.get("ts", ""))
        else:
            spot_note = f"Spot fetch failed: {spot.get('err')}"

    if spot_note:
        st.info(f"{etf_info.get('name_cn','ETF')} ({etf_ticker}) {spot_note}")

    # ---- Collect signal for summary at end
    fut_pfx = OPT_TO_FUT.get(pfx)
    fut_sub = futures_df[futures_df["product"] == fut_pfx].sort_values("expiry").copy() if fut_pfx else None

    # Put front option F/T into session_state so etf_spot_signal_panel can use it (keeps signature stable)
    st.session_state["__FRONT_OPT_F__"] = front_opt_F
    st.session_state["__FRONT_OPT_T__"] = front_opt_T

    sig = etf_spot_signal_panel(
        futures_sub=fut_sub,
        spot_px=spot_px,
        spot_ts=spot_ts,
        atm_term=atm_df,
        skew_term=skew_df,
        surf_points=surf_df,
        r=float(r),
        q=float(q),
    )

    signals_summary.append(
        {
            "opt_pfx": pfx,
            "opt_cn": OPTIONS_NAME_MAP.get(pfx, pfx),
            "opt_en": OPTIONS_NAME_EN.get(pfx, pfx),
            "fut_pfx": fut_pfx,
            "fut_cn": FUTURES_NAME_MAP.get(fut_pfx, fut_pfx or ""),
            "fut_en": FUTURES_NAME_EN.get(fut_pfx, fut_pfx or ""),
            "etf_name_cn": etf_info.get("name_cn", "ETF"),
            "etf_ticker": etf_ticker,
            "spot_name": spot_name,
            "sig": sig,
        }
    )


# =========================
# 3) ETF Spot Signal Panel Summary (END)
# =========================
st.subheader("3) ETF Spot Signal Panel")

if not signals_summary:
    st.info("No ETF spot signals available (missing curves / IV / skew inputs).")
else:
    signals_summary = sorted(signals_summary, key=lambda x: x["sig"].get("confidence", 0), reverse=True)

    for item in signals_summary:
        render_etf_spot_panel_row(
            opt_pfx=item["opt_pfx"],
            opt_cn=item["opt_cn"],
            opt_en=item["opt_en"],
            fut_pfx=item["fut_pfx"],
            fut_cn=item["fut_cn"],
            fut_en=item["fut_en"],
            etf_name_cn=item["etf_name_cn"],
            etf_ticker=item["etf_ticker"],
            spot_name=item.get("spot_name", ""),
            sig=item["sig"],
        )


if show_debug:
    st.subheader("Debug")
    st.write("Options parsed preview (first 200 rows)")
    st.dataframe(options_df.head(200), use_container_width=True)
    st.write("Unique expiries by product:")
    st.write(options_df.groupby("product")["expiry"].nunique())
    st.write(options_df.groupby("product")["expiry"].unique())
