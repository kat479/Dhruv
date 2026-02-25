"""
eod_tracker.py — End-of-Day Portfolio Tracker
===============================================
Fetches closing prices for your selected portfolio stocks,
calculates P&L vs buy prices, and appends a daily snapshot
to portfolio_history.csv.

Designed to be run by GitHub Actions every weekday at 4:30 PM IST
(after NSE market close at 3:30 PM IST).

Usage:
    python eod_tracker.py                # fetch EOD and append to CSV
    python eod_tracker.py --init         # set today's prices as buy prices
    python eod_tracker.py --report       # print P&L summary to console
    python eod_tracker.py --stocks       # print current positions

The portfolio is read from positions.json (created via --init or the
Streamlit tracker's Setup tab).
"""

import yfinance as yf
import pandas as pd
import json
import os
import sys
import argparse
from datetime import datetime, date

# ─────────────────────────────────────────────
# FILES
# ─────────────────────────────────────────────
POSITIONS_FILE   = "positions.json"
HISTORY_FILE     = "portfolio_history.csv"
CACHE_FILE       = "nifty500_cache.csv"

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def load_positions() -> dict:
    if not os.path.exists(POSITIONS_FILE):
        print(f"❌  {POSITIONS_FILE} not found. Run with --init first.")
        sys.exit(1)
    with open(POSITIONS_FILE) as f:
        data = json.load(f)
    # strip comment keys
    return {k: v for k, v in data.items() if not k.startswith("_")}


def save_positions(pos: dict):
    with open(POSITIONS_FILE, "w") as f:
        json.dump(pos, f, indent=2)


def fetch_prices(tickers: list) -> dict:
    """Fetch latest closing price for each ticker."""
    prices = {}
    try:
        raw = yf.download(
            " ".join(tickers),
            period="2d",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        for t in tickers:
            try:
                series = raw["Close"].dropna() if len(tickers) == 1 else raw[t]["Close"].dropna()
                prices[t] = float(series.iloc[-1]) if not series.empty else None
            except Exception:
                prices[t] = None
    except Exception as e:
        print(f"⚠️  Batch fetch failed ({e}), falling back to individual...")
        for t in tickers:
            try:
                h = yf.Ticker(t).history(period="2d")
                prices[t] = float(h["Close"].dropna().iloc[-1]) if not h.empty else None
            except Exception:
                prices[t] = None
    return prices


def load_history() -> pd.DataFrame:
    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame()
    df = pd.read_csv(HISTORY_FILE)
    for c in ["buy_price","shares","allocation","current_price","current_value","pnl_abs","pnl_pct"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ─────────────────────────────────────────────
# COMMAND: --init
# ─────────────────────────────────────────────

def cmd_init(top_n: int = 10):
    """
    Initialise positions from the screener cache.
    Sets today's closing prices as buy prices.
    Allocates equally across top_n stocks by KPI score.
    """
    if not os.path.exists(CACHE_FILE):
        print(f"❌  {CACHE_FILE} not found. Run the Streamlit screener first.")
        sys.exit(1)

    print(f"\n🔍  Loading top {top_n} stocks from {CACHE_FILE}...")
    df = pd.read_csv(CACHE_FILE)

    # Coerce numerics
    num_cols = ["rev_growth","earn_growth","roe","debt_equity","peg","insider_hold","score"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Score if not present
    if "score" not in df.columns or df["score"].isna().all():
        print("   Computing KPI scores...")
        df["score"] = df.apply(_score_row, axis=1)

    df = df[df["current_price"].notna()].sort_values("score", ascending=False).head(top_n)
    tickers    = df["Yahoo Symbol"].tolist()
    alloc_each = 1_000_000 / top_n   # default ₹10L equal split

    print(f"📡  Fetching live prices for {len(tickers)} stocks...")
    prices = fetch_prices(tickers)

    positions = {}
    failed    = []
    today     = date.today().isoformat()

    for _, row in df.iterrows():
        t = row["Yahoo Symbol"]
        p = prices.get(t)
        if p and p > 0:
            shares = alloc_each / p
            positions[t] = {
                "name":       row["Company Name"],
                "nse_symbol": row.get("NSE Symbol", t.replace(".NS", "")),
                "industry":   row.get("Industry", ""),
                "kpi_score":  int(row["score"]),
                "buy_price":  round(p, 4),
                "shares":     round(shares, 6),
                "allocation": round(alloc_each, 2),
                "buy_date":   today,
            }
            print(f"   ✅  {row['Company Name'][:30]:30s} ₹{p:>10,.2f}  ×{shares:>10.4f} shares")
        else:
            failed.append(t)
            print(f"   ❌  {row['Company Name'][:30]:30s} — price unavailable")

    save_positions(positions)
    print(f"\n💾  Saved {len(positions)} positions → {POSITIONS_FILE}")
    if failed:
        print(f"⚠️   Failed: {failed}")


# ─────────────────────────────────────────────
# COMMAND: EOD fetch (default)
# ─────────────────────────────────────────────

def cmd_eod():
    """
    Fetch end-of-day prices, compute P&L, append to portfolio_history.csv.
    Idempotent — re-running on the same day overwrites today's rows.
    """
    positions = load_positions()
    tickers   = list(positions.keys())
    today     = date.today().isoformat()
    now_ts    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n📡  Fetching EOD prices ({today}) for {len(tickers)} stocks...")
    prices = fetch_prices(tickers)

    rows          = []
    total_cost    = 0
    total_value   = 0

    for ticker, pos in positions.items():
        bp  = pos["buy_price"]
        sh  = pos["shares"]
        al  = pos["allocation"]
        cp  = prices.get(ticker)
        cv  = round(sh * cp, 2)   if cp else None
        pnl = round(cv - al, 2)   if cv else None
        pct = round((pnl/al)*100, 4) if pnl is not None else None

        total_cost  += al
        total_value += cv if cv else al

        rows.append({
            "date":          today,
            "timestamp":     now_ts,
            "ticker":        ticker,
            "name":          pos["name"],
            "nse_symbol":    pos.get("nse_symbol", ""),
            "industry":      pos.get("industry", ""),
            "kpi_score":     pos.get("kpi_score", ""),
            "buy_date":      pos["buy_date"],
            "buy_price":     bp,
            "shares":        sh,
            "allocation":    al,
            "current_price": round(cp, 4) if cp else None,
            "current_value": cv,
            "pnl_abs":       pnl,
            "pnl_pct":       pct,
        })

    # Build / update CSV
    df_new = pd.DataFrame(rows)
    if os.path.exists(HISTORY_FILE):
        df_old = pd.read_csv(HISTORY_FILE)
        df_old = df_old[df_old["date"] != today]   # remove today's rows if re-running
        df_out = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_out = df_new

    df_out.to_csv(HISTORY_FILE, index=False)

    # ── Summary ──────────────────────────────
    total_pnl = total_value - total_cost
    total_pct = (total_pnl / total_cost * 100) if total_cost else 0
    sign = "+" if total_pnl >= 0 else ""

    print(f"\n{'─'*58}")
    print(f"  📅  Date             : {today}")
    print(f"  📦  Stocks tracked   : {sum(1 for r in rows if r['current_price'])}/{len(tickers)}")
    print(f"  💰  Total invested   : ₹{total_cost:>12,.0f}")
    print(f"  📈  Current value    : ₹{total_value:>12,.0f}")
    print(f"  💹  Total P&L        : {sign}₹{abs(total_pnl):>10,.0f}  ({sign}{total_pct:.2f}%)")
    print(f"{'─'*58}")
    print(f"  💾  Appended to      : {HISTORY_FILE}")
    print(f"  📊  Total rows       : {len(df_out)}")

    # Top gainer / loser
    valid = [r for r in rows if r["pnl_pct"] is not None]
    if valid:
        best  = max(valid, key=lambda x: x["pnl_pct"])
        worst = min(valid, key=lambda x: x["pnl_pct"])
        print(f"\n  🏆  Top gainer  : {best['name']}  ({best['pnl_pct']:+.2f}%)")
        print(f"  📉  Top loser   : {worst['name']}  ({worst['pnl_pct']:+.2f}%)")

    print()
    return df_out


# ─────────────────────────────────────────────
# COMMAND: --report
# ─────────────────────────────────────────────

def cmd_report():
    """Print a full P&L table from the latest snapshot in history CSV."""
    df = load_history()
    if df.empty:
        print("❌  No history found. Run eod_tracker.py first.")
        return

    latest = df["date"].max()
    snap   = df[df["date"] == latest].sort_values("pnl_pct", ascending=False)

    total_invested = snap["allocation"].sum()
    total_value    = snap["current_value"].sum()
    total_pnl      = total_value - total_invested
    total_pct      = (total_pnl / total_invested * 100) if total_invested else 0
    sign           = "+" if total_pnl >= 0 else ""

    print(f"\n{'═'*78}")
    print(f"  ⭐  DHRUV PORTFOLIO REPORT  —  {latest}")
    print(f"{'═'*78}")
    print(f"  {'Stock':<28} {'Buy ₹':>10} {'Now ₹':>10} {'Shares':>9} {'P&L ₹':>11} {'P&L %':>8}")
    print(f"  {'─'*28} {'─'*10} {'─'*10} {'─'*9} {'─'*11} {'─'*8}")

    for _, r in snap.iterrows():
        pp = r.get("pnl_pct"); pa = r.get("pnl_abs")
        cp = r.get("current_price")
        s  = "+" if (pa or 0) >= 0 else ""
        e  = "🟢" if (pp or 0) >= 0 else "🔴"
        print(f"  {e} {str(r['name']):<26} "
              f"₹{r['buy_price']:>9,.2f} "
              f"₹{float(cp):>9,.2f}  " if cp else f"  {'N/A':>10}  ",
              end="")
        if pd.notna(pa):
            print(f"{r['shares']:>9.3f} {s}₹{abs(pa):>9,.0f} {s}{abs(pp):>7.2f}%")
        else:
            print("N/A")

    print(f"  {'─'*28} {'─'*10} {'─'*10} {'─'*9} {'─'*11} {'─'*8}")
    print(f"  {'TOTAL':<28} {'':>10} {'':>10} {'':>9} "
          f"{sign}₹{abs(total_pnl):>9,.0f} {sign}{abs(total_pct):>7.2f}%")
    print(f"{'═'*78}")
    print(f"  Invested : ₹{total_invested:,.0f}   |   "
          f"Value : ₹{total_value:,.0f}   |   "
          f"P&L : {sign}₹{abs(total_pnl):,.0f} ({sign}{total_pct:.2f}%)")
    days = df["date"].nunique()
    print(f"  Tracking since {df['date'].min()}  ({days} trading days captured)")
    print(f"{'═'*78}\n")


# ─────────────────────────────────────────────
# COMMAND: --stocks
# ─────────────────────────────────────────────

def cmd_stocks():
    """Print current positions."""
    pos = load_positions()
    print(f"\n⭐  DHRUV — {len(pos)} positions in {POSITIONS_FILE}\n")
    print(f"  {'Ticker':<16} {'Name':<30} {'Buy ₹':>10} {'Shares':>10} {'Alloc ₹':>12} {'KPI':>5}")
    print(f"  {'─'*16} {'─'*30} {'─'*10} {'─'*10} {'─'*12} {'─'*5}")
    for t, v in pos.items():
        print(f"  {t:<16} {v['name'][:28]:<30} "
              f"₹{v['buy_price']:>9,.2f} {v['shares']:>10.4f} "
              f"₹{v['allocation']:>11,.0f} {v.get('kpi_score','?'):>5}")
    print()


# ─────────────────────────────────────────────
# KPI SCORER (standalone, mirrors dashboard)
# ─────────────────────────────────────────────

def _score_row(row) -> int:
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


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="⭐ Dhruv — EOD Portfolio Tracker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eod_tracker.py                  # fetch today's EOD prices → CSV
  python eod_tracker.py --init           # init from screener cache (top 10)
  python eod_tracker.py --init --top 15  # init top 15 stocks
  python eod_tracker.py --report         # print P&L table
  python eod_tracker.py --stocks         # list current positions
        """
    )
    parser.add_argument("--init",    action="store_true", help="Initialise positions from screener cache")
    parser.add_argument("--top",     type=int, default=10, help="Number of top stocks to initialise (default: 10)")
    parser.add_argument("--report",  action="store_true", help="Print P&L report")
    parser.add_argument("--stocks",  action="store_true", help="Print current positions")
    args = parser.parse_args()

    if args.init:
        cmd_init(top_n=args.top)
    elif args.report:
        cmd_report()
    elif args.stocks:
        cmd_stocks()
    else:
        cmd_eod()
