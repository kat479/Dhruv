"""
shared/data.py — Shared data loading, scoring, and helpers
Used by all three pages to avoid duplication.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
import time
from io import StringIO
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
CACHE_FILE      = "nifty500_cache.csv"
SYMBOLS_FILE    = "nifty500_yahoo_symbols.csv"
CACHE_TTL_HOURS = 24
BATCH_SIZE      = 50
RISK_FREE_RATE  = 0.068   # Indian 10Y G-Sec

NSE_URL     = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.nseindia.com/",
}

NUMERIC_COLS = [
    "current_price", "market_cap", "52w_high", "52w_low",
    "pct_from_high", "pct_from_low", "pe", "fwd_pe", "peg",
    "pb", "ps", "rev_growth", "earn_growth", "roe", "roa",
    "debt_equity", "free_cashflow", "inst_holding", "insider_hold",
    "div_yield", "score",
]


# ─────────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────────

def multibagger_score(row) -> int:
    score = 0
    rg = row.get("rev_growth")
    if rg is not None and pd.notna(rg):
        score += 20 if rg > 0.25 else (12 if rg > 0.15 else 0)

    eg = row.get("earn_growth")
    if eg is not None and pd.notna(eg):
        score += 20 if eg > 0.25 else (10 if eg > 0.15 else 0)

    roe = row.get("roe")
    if roe is not None and pd.notna(roe):
        score += 15 if roe > 0.20 else (8 if roe > 0.15 else 0)

    de = row.get("debt_equity")
    if de is not None and pd.notna(de):
        score += 15 if de < 0.3 else (8 if de < 50 else 0)

    peg = row.get("peg")
    if peg is not None and pd.notna(peg) and peg > 0:
        score += 15 if peg < 1.0 else (8 if peg < 1.5 else 0)

    ins = row.get("insider_hold")
    if ins is not None and pd.notna(ins):
        score += 15 if ins > 0.50 else (8 if ins > 0.35 else 0)

    return min(score, 100)


def multibagger_score_detail(row) -> tuple:
    """Returns (score, list of (icon, text) checks)."""
    score, checks = 0, []

    rg = row.get("rev_growth")
    if rg is not None and pd.notna(rg):
        if rg > 0.25:   score += 20; checks.append(("✅", f"Revenue growth: {rg*100:.1f}% (>25%)"))
        elif rg > 0.15: score += 12; checks.append(("🟡", f"Revenue growth: {rg*100:.1f}% (>15%)"))
        else:                         checks.append(("❌", f"Revenue growth: {rg*100:.1f}% (<15%)"))
    else: checks.append(("⚪", "Revenue growth: N/A"))

    eg = row.get("earn_growth")
    if eg is not None and pd.notna(eg):
        if eg > 0.25:   score += 20; checks.append(("✅", f"Earnings growth: {eg*100:.1f}% (>25%)"))
        elif eg > 0.15: score += 10; checks.append(("🟡", f"Earnings growth: {eg*100:.1f}% (>15%)"))
        else:                         checks.append(("❌", f"Earnings growth: {eg*100:.1f}% (<15%)"))
    else: checks.append(("⚪", "Earnings growth: N/A"))

    roe = row.get("roe")
    if roe is not None and pd.notna(roe):
        if roe > 0.20:   score += 15; checks.append(("✅", f"ROE: {roe*100:.1f}% (>20%)"))
        elif roe > 0.15: score += 8;  checks.append(("🟡", f"ROE: {roe*100:.1f}% (>15%)"))
        else:                          checks.append(("❌", f"ROE: {roe*100:.1f}% (<15%)"))
    else: checks.append(("⚪", "ROE: N/A"))

    de = row.get("debt_equity")
    if de is not None and pd.notna(de):
        if de < 0.3:  score += 15; checks.append(("✅", f"D/E: {de/100:.2f} (very low)"))
        elif de < 50: score += 8;  checks.append(("🟡", f"D/E: {de/100:.2f} (moderate)"))
        else:                       checks.append(("❌", f"D/E: {de/100:.2f} (high)"))
    else: checks.append(("⚪", "D/E: N/A"))

    peg = row.get("peg")
    if peg is not None and pd.notna(peg) and peg > 0:
        if peg < 1.0:   score += 15; checks.append(("✅", f"PEG: {peg:.2f} (<1 — undervalued)"))
        elif peg < 1.5: score += 8;  checks.append(("🟡", f"PEG: {peg:.2f} (<1.5 — fair value)"))
        else:                         checks.append(("❌", f"PEG: {peg:.2f} (>1.5 — expensive)"))
    else: checks.append(("⚪", "PEG: N/A"))

    ins = row.get("insider_hold")
    if ins is not None and pd.notna(ins):
        if ins > 0.50:   score += 15; checks.append(("✅", f"Insider/Promoter: {ins*100:.1f}% (>50%)"))
        elif ins > 0.35: score += 8;  checks.append(("🟡", f"Insider/Promoter: {ins*100:.1f}% (>35%)"))
        else:                          checks.append(("❌", f"Insider/Promoter: {ins*100:.1f}% (<35%)"))
    else: checks.append(("⚪", "Insider holding: N/A"))

    return min(score, 100), checks


def kpi_components(row) -> dict:
    rg  = row.get("rev_growth");   eg  = row.get("earn_growth")
    roe = row.get("roe");          de  = row.get("debt_equity")
    peg = row.get("peg");          ins = row.get("insider_hold")
    return {
        "Rev Growth": 20 if (rg and pd.notna(rg) and rg > 0.25) else (12 if (rg and pd.notna(rg) and rg > 0.15) else 0),
        "EPS Growth": 20 if (eg and pd.notna(eg) and eg > 0.25) else (10 if (eg and pd.notna(eg) and eg > 0.15) else 0),
        "ROE":        15 if (roe and pd.notna(roe) and roe > 0.20) else (8 if (roe and pd.notna(roe) and roe > 0.15) else 0),
        "Low Debt":   15 if (de and pd.notna(de) and de < 0.3) else (8 if (de and pd.notna(de) and de < 50) else 0),
        "Valuation":  15 if (peg and pd.notna(peg) and 0 < peg < 1.0) else (8 if (peg and pd.notna(peg) and peg < 1.5) else 0),
        "Promoter":   15 if (ins and pd.notna(ins) and ins > 0.50) else (8 if (ins and pd.notna(ins) and ins > 0.35) else 0),
    }


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def fmt_num(val, prefix="", suffix="", decimals=2):
    if val is None or (isinstance(val, float) and val != val): return "N/A"
    try:
        val = float(val)
        if abs(val) >= 1e9:  return f"{prefix}{val/1e9:.{decimals}f}B{suffix}"
        if abs(val) >= 1e7:  return f"{prefix}{val/1e7:.{decimals}f}Cr{suffix}"
        if abs(val) >= 1e6:  return f"{prefix}{val/1e6:.{decimals}f}M{suffix}"
        return f"{prefix}{val:.{decimals}f}{suffix}"
    except Exception: return "N/A"


def score_color(score: int) -> str:
    if score >= 60: return "#4ade80"
    if score >= 40: return "#fb923c"
    return "#f87171"


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ─────────────────────────────────────────────
# CACHE MANAGEMENT
# ─────────────────────────────────────────────

def cache_is_fresh() -> bool:
    if not os.path.exists(CACHE_FILE): return False
    mtime = datetime.fromtimestamp(os.path.getmtime(CACHE_FILE))
    return datetime.now() - mtime < timedelta(hours=CACHE_TTL_HOURS)


def load_cache() -> pd.DataFrame:
    if not os.path.exists(CACHE_FILE): return pd.DataFrame()
    df = pd.read_csv(CACHE_FILE)
    return coerce_numeric(df)


def save_cache(df: pd.DataFrame):
    df.to_csv(CACHE_FILE, index=False)


# ─────────────────────────────────────────────
# PEG / ROE CALCULATORS
# ─────────────────────────────────────────────

def _calc_peg(info: dict):
    peg_yf = info.get("pegRatio")
    if peg_yf is not None and 0 < peg_yf < 50: return peg_yf
    pe = info.get("trailingPE")
    if pe is None or pe <= 0: return None
    growth = info.get("earningsGrowth") or info.get("revenueGrowth")
    if growth is None or growth <= 0: return None
    peg = pe / (growth * 100)
    return round(peg, 2) if 0 < peg < 50 else None


def _calc_roe(info: dict, ticker_obj=None):
    roe_yf = info.get("returnOnEquity")
    if roe_yf is not None and -2 < roe_yf < 10: return roe_yf
    try:
        if ticker_obj is not None:
            financials = ticker_obj.financials
            balance    = ticker_obj.balance_sheet
            if financials is not None and not financials.empty:
                net_income = None
                for key in ["Net Income", "Net Income Common Stockholders"]:
                    if key in financials.index:
                        net_income = float(financials.loc[key].iloc[0]); break
                equity = None
                for key in ["Stockholders Equity", "Common Stock Equity", "Total Stockholders Equity"]:
                    if balance is not None and not balance.empty and key in balance.index:
                        equity = float(balance.loc[key].iloc[0]); break
                if net_income and equity and equity > 0:
                    roe = net_income / equity
                    if -2 < roe < 10: return round(roe, 4)
    except Exception: pass
    try:
        eps = info.get("trailingEps"); bvps = info.get("bookValue")
        if eps and bvps and bvps > 0:
            roe = eps / bvps
            if -2 < roe < 10: return round(roe, 4)
    except Exception: pass
    return None


# ─────────────────────────────────────────────
# SYMBOL LIST
# ─────────────────────────────────────────────

@st.cache_data(ttl=86400)
def load_nifty500_symbols() -> pd.DataFrame:
    if os.path.exists(SYMBOLS_FILE):
        df = pd.read_csv(SYMBOLS_FILE)
        if {"Company Name", "Yahoo Symbol"}.issubset(df.columns): return df
    try:
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=NSE_HEADERS, timeout=10)
        time.sleep(0.5)
        r = session.get(NSE_URL, headers=NSE_HEADERS, timeout=20)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        df.columns = df.columns.str.strip()
        sym_col  = next((c for c in df.columns if "symbol"  in c.lower()), None)
        name_col = next((c for c in df.columns if "company" in c.lower() or "name" in c.lower()), None)
        ind_col  = next((c for c in df.columns if "industry" in c.lower()), None)
        isin_col = next((c for c in df.columns if "isin"    in c.lower()), None)
        records  = []
        for _, row in df.iterrows():
            sym = str(row[sym_col]).strip().upper()
            records.append({
                "Company Name": row[name_col].strip() if name_col else sym,
                "NSE Symbol":   sym,
                "Yahoo Symbol": f"{sym}.NS",
                "Industry":     row[ind_col].strip()  if ind_col  else "",
                "ISIN":         row[isin_col].strip() if isin_col else "",
            })
        result = pd.DataFrame(records)
        result.to_csv(SYMBOLS_FILE, index=False)
        return result
    except Exception as e:
        st.error(f"❌ Could not download Nifty 500 list: {e}")
        st.stop()


# ─────────────────────────────────────────────
# BATCH FETCH
# ─────────────────────────────────────────────

def fetch_batch_fundamentals(tickers: list) -> dict:
    results = {}
    ticker_str = " ".join(tickers)
    try:
        hist_all = yf.download(ticker_str, period="1y", group_by="ticker",
                               auto_adjust=True, progress=False, threads=True)
    except Exception:
        hist_all = pd.DataFrame()
    try:
        tickers_obj = yf.Tickers(ticker_str)
    except Exception:
        tickers_obj = None

    for ticker in tickers:
        try:
            info = {}
            t    = None
            if tickers_obj:
                t    = tickers_obj.tickers.get(ticker)
                info = t.info or {} if t else {}
            hist = pd.DataFrame()
            if not hist_all.empty:
                if len(tickers) == 1:
                    hist = hist_all
                elif ticker in hist_all.columns.get_level_values(0):
                    hist = hist_all[ticker].dropna(how="all")
            high_52w = hist["High"].max() if not hist.empty and "High" in hist.columns else None
            low_52w  = hist["Low"].min()  if not hist.empty and "Low"  in hist.columns else None
            current  = info.get("currentPrice") or info.get("regularMarketPrice")
            if current is None and not hist.empty and "Close" in hist.columns:
                closes = hist["Close"].dropna()
                current = float(closes.iloc[-1]) if len(closes) else None
            pct_from_high = ((current - high_52w) / high_52w * 100) if (current and high_52w) else None
            pct_from_low  = ((current - low_52w)  / low_52w  * 100) if (current and low_52w)  else None
            results[ticker] = {
                "ticker": ticker, "name": info.get("longName", ticker),
                "sector": info.get("sector", ""), "industry": info.get("industry", ""),
                "current_price": current, "market_cap": info.get("marketCap"),
                "52w_high": high_52w, "52w_low": low_52w,
                "pct_from_high": pct_from_high, "pct_from_low": pct_from_low,
                "pe": info.get("trailingPE"), "fwd_pe": info.get("forwardPE"),
                "peg": _calc_peg(info), "pb": info.get("priceToBook"),
                "ps": info.get("priceToSalesTrailing12Months"),
                "rev_growth": info.get("revenueGrowth"), "earn_growth": info.get("earningsGrowth"),
                "roe": _calc_roe(info, t), "roa": info.get("returnOnAssets"),
                "debt_equity": info.get("debtToEquity"), "free_cashflow": info.get("freeCashflow"),
                "inst_holding": info.get("heldPercentInstitutions"),
                "insider_hold": info.get("heldPercentInsiders"),
                "div_yield": info.get("dividendYield"), "error": None,
            }
        except Exception as e:
            results[ticker] = {"ticker": ticker, "error": str(e)}
    return results


def fetch_all_nifty500(symbols_df: pd.DataFrame) -> pd.DataFrame:
    tickers = symbols_df["Yahoo Symbol"].tolist()
    batches = [tickers[i:i+BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    st.info(f"📡 Fetching **{len(tickers)} stocks** in {len(batches)} batches. Takes ~3–5 min, cached for {CACHE_TTL_HOURS}h.")
    bar = st.progress(0, text="Starting...")
    status = st.empty()
    all_results = {}
    for i, batch in enumerate(batches):
        status.markdown(f"⏳ Batch **{i+1}/{len(batches)}** — {batch[0]} → {batch[-1]}")
        all_results.update(fetch_batch_fundamentals(batch))
        bar.progress((i+1)/len(batches), text=f"Batch {i+1}/{len(batches)}")
        time.sleep(0.3)
    bar.empty(); status.empty()
    sym_lookup = symbols_df.set_index("Yahoo Symbol").to_dict("index")
    rows = []
    for ticker, d in all_results.items():
        meta = sym_lookup.get(ticker, {})
        rows.append({
            "Company Name": meta.get("Company Name", d.get("name", ticker)),
            "NSE Symbol":   meta.get("NSE Symbol",   ticker.replace(".NS", "")),
            "Yahoo Symbol": ticker,
            "Industry":     meta.get("Industry",     d.get("industry", "")),
            "ISIN":         meta.get("ISIN", ""),
            **{k: v for k, v in d.items() if k not in ("ticker","name","industry","error")},
            "fetch_error":  d.get("error"),
        })
    df = pd.DataFrame(rows)
    save_cache(df)
    st.success(f"✅ Fetched and cached {len(df)} stocks!")
    return df


# ─────────────────────────────────────────────
# PRICE HISTORY
# ─────────────────────────────────────────────

@st.cache_data(ttl=3600)
def fetch_price_history(tickers: tuple, period: str = "2y") -> pd.DataFrame:
    try:
        raw = yf.download(" ".join(tickers), period=period, group_by="ticker",
                          auto_adjust=True, progress=False, threads=True)
        if len(tickers) == 1:
            closes = raw[["Close"]].rename(columns={"Close": tickers[0]})
        else:
            closes = pd.DataFrame()
            for t in tickers:
                try:
                    if t in raw.columns.get_level_values(0):
                        closes[t] = raw[t]["Close"]
                except Exception: pass
        return closes.dropna(how="all")
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────
# MPT HELPERS
# ─────────────────────────────────────────────

def portfolio_stats(weights, mean_returns, cov_matrix, rf=RISK_FREE_RATE):
    ret = np.dot(weights, mean_returns) * 252
    vol = np.sqrt(weights @ cov_matrix @ weights) * np.sqrt(252)
    return {"return": ret, "volatility": vol, "sharpe": (ret - rf) / vol if vol > 0 else 0}
