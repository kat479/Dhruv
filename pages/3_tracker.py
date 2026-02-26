"""Page 3 — Portfolio Tracker (Daily P&L)"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json, os, sys
from datetime import datetime, date

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from shared.data import load_cache, multibagger_score, score_color

POSITIONS_FILE = "positions.json"
HISTORY_FILE   = "portfolio_history.csv"

# ── Header ───────────────────────────────────
st.markdown("""
<div class="page-header">
    <div class="eyebrow">Daily P&L · Position Tracking</div>
    <h1>📈 Portfolio Tracker</h1>
    <p>Track your actual positions, monitor daily moves, and review cumulative P&L.</p>
</div>
""", unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────
def load_positions() -> dict:
    if not os.path.exists(POSITIONS_FILE): return {}
    with open(POSITIONS_FILE) as f:
        return {k:v for k,v in json.load(f).items() if not k.startswith("_comment")}

def save_positions(pos: dict):
    with open(POSITIONS_FILE, "w") as f:
        json.dump(pos, f, indent=2)

def load_history() -> pd.DataFrame:
    if not os.path.exists(HISTORY_FILE): return pd.DataFrame()
    df = pd.read_csv(HISTORY_FILE)
    for c in ["buy_price","shares","allocation","current_price","current_value","pnl_abs","pnl_pct"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def fetch_prices(tickers: list) -> dict:
    prices = {}
    try:
        data = yf.download(" ".join(tickers), period="2d", group_by="ticker",
                           auto_adjust=True, progress=False, threads=True)
        for t in tickers:
            try:
                series = data["Close"].dropna() if len(tickers)==1 else data[t]["Close"].dropna()
                prices[t] = float(series.iloc[-1]) if not series.empty else None
            except: prices[t] = None
    except:
        for t in tickers:
            try:
                h = yf.Ticker(t).history(period="2d")
                prices[t] = float(h["Close"].dropna().iloc[-1]) if not h.empty else None
            except: prices[t] = None
    return prices

def append_history(positions: dict, prices: dict):
    today  = date.today().isoformat()
    now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows   = []
    for ticker, pos in positions.items():
        bp  = pos["buy_price"]; sh = pos["shares"]; al = pos["allocation"]
        cp  = prices.get(ticker)
        cv  = sh*cp if cp else None
        pnl = cv-al if cv else None
        pct = (pnl/al*100) if pnl is not None else None
        rows.append({"date":today,"timestamp":now_ts,"ticker":ticker,"name":pos["name"],
                     "buy_date":pos["buy_date"],"buy_price":bp,"shares":sh,"allocation":al,
                     "current_price":round(cp,4) if cp else None,
                     "current_value":round(cv,2) if cv else None,
                     "pnl_abs":round(pnl,2) if pnl is not None else None,
                     "pnl_pct":round(pct,4) if pct is not None else None})
    df_new = pd.DataFrame(rows)
    if os.path.exists(HISTORY_FILE):
        df_ex = pd.read_csv(HISTORY_FILE)
        df_ex = df_ex[df_ex["date"]!=today]
        df_out = pd.concat([df_ex, df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.to_csv(HISTORY_FILE, index=False)
    return df_out

# ── Sidebar ──────────────────────────────────
with st.sidebar:
    st.markdown("### 📈 Tracker Settings")
    st.divider()
    total_inv = st.number_input("Total Investment (₹)", min_value=100_000,
                                 max_value=100_000_000, value=1_000_000, step=100_000, format="%d")

    st.divider()
    st.markdown("**Quick actions**")
    if st.button("🔄 Refresh Prices Now"):
        st.cache_data.clear(); st.rerun()
    if st.button("💾 Save Today's Prices to CSV"):
        pos = load_positions()
        if pos:
            with st.spinner("Fetching..."):
                pr = fetch_prices(list(pos.keys()))
                append_history(pos, pr)
            st.success("Saved!"); st.rerun()
        else:
            st.warning("No positions loaded.")

# ── Positions setup ──────────────────────────
positions = load_positions()
history   = load_history()

tab_pos, tab_init, tab_history, tab_pnl = st.tabs([
    "📊 Live Positions",
    "⚙️ Setup / Init",
    "📅 History Log",
    "📉 P&L Analytics",
])

# ══════════════════════════════════════════════
# TAB 1 — LIVE POSITIONS
# ══════════════════════════════════════════════
with tab_pos:
    if not positions:
        st.info("👆 No positions found. Go to **⚙️ Setup** tab to initialise your portfolio.")
    else:
        tickers_list = list(positions.keys())
        with st.spinner("Fetching live prices..."):
            live_prices = fetch_prices(tickers_list)

        total_cost = 0; total_val = 0
        rows = []
        for ticker, pos in positions.items():
            bp = pos["buy_price"]; sh = pos["shares"]; al = pos["allocation"]
            cp = live_prices.get(ticker)
            cv = sh*cp if cp else None
            pnl_a = cv-al if cv else None
            pnl_p = (pnl_a/al*100) if pnl_a is not None else None
            total_cost += al
            total_val  += cv if cv else al
            rows.append({"name":pos["name"],"ticker":ticker,
                         "buy_price":bp,"current_price":cp,
                         "shares":sh,"allocation":al,
                         "current_value":cv,"pnl_abs":pnl_a,"pnl_pct":pnl_p})

        total_pnl = total_val - total_cost
        total_pct = (total_pnl/total_cost*100) if total_cost else 0
        sign      = "+" if total_pnl>=0 else ""
        pnl_col   = "#4ade80" if total_pnl>=0 else "#f87171"

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Invested",  f"₹{total_cost:,.0f}")
        c2.metric("Current Value",   f"₹{total_val:,.0f}")
        c3.metric("Total P&L",       f"{'+'if total_pnl>=0 else''}₹{total_pnl:,.0f}",
                  delta=f"{total_pct:.2f}%")
        c4.metric("Positions",       len(positions))
        st.divider()

        # Position cards
        sorted_rows = sorted(rows, key=lambda x: (x["pnl_pct"] or -999), reverse=True)
        for rank, r in enumerate(sorted_rows):
            pp   = r["pnl_pct"] or 0
            pa   = r["pnl_abs"] or 0
            col  = "#4ade80" if pp>=0 else "#f87171"
            sign = "+" if pp>=0 else ""
            cp_  = r["current_price"]
            cv_  = r["current_value"]

            col1, col2, col3, col4, col5, col6 = st.columns([3,2,2,2,2,2])
            with col1:
                st.markdown(f"""
                <div style="padding:8px 0">
                    <div style="font-size:14px;font-weight:700;color:#f1f5f9">{r['name']}</div>
                    <div style="font-family:'DM Mono',monospace;font-size:11px;color:#6b7280">{r['ticker']}</div>
                </div>""", unsafe_allow_html=True)
            col2.metric("Buy",      f"₹{r['buy_price']:,.2f}")
            col3.metric("Now",      f"₹{cp_:,.2f}" if cp_ else "N/A")
            col4.metric("Value",    f"₹{cv_:,.0f}" if cv_ else "N/A")
            col5.metric("P&L ₹",   f"{'+' if pa>0 else ''}₹{pa:,.0f}" if pa else "N/A")
            col6.metric("P&L %",   f"{'+' if pp>0 else ''}{pp:.2f}%" if pp else "N/A",
                        delta=f"{pp:.2f}%" if pp else None)
            st.markdown('<hr style="border-color:#1e2d45;margin:4px 0">', unsafe_allow_html=True)

        # Waterfall chart
        st.subheader("📊 P&L Waterfall")
        sorted_pnl = sorted(rows, key=lambda x: x["pnl_abs"] or 0, reverse=True)
        fig_wf = go.Figure(go.Bar(
            x=[r["name"][:15] for r in sorted_pnl],
            y=[r["pnl_abs"] or 0 for r in sorted_pnl],
            marker_color=["#4ade80" if (r["pnl_abs"] or 0)>=0 else "#f87171" for r in sorted_pnl],
            text=[f"{'+'if (r['pnl_abs'] or 0)>=0 else ''}₹{(r['pnl_abs'] or 0):,.0f}" for r in sorted_pnl],
            textposition="outside",
        ))
        fig_wf.add_hline(y=0, line_color="#374151", line_width=1)
        fig_wf.update_layout(paper_bgcolor="#080c14", plot_bgcolor="#080c14", font_color="#e2e8f0",
                              height=340, xaxis=dict(gridcolor="#1e2d45",tickangle=-35),
                              yaxis=dict(gridcolor="#1e2d45",title="P&L (₹)"),
                              margin=dict(t=20))
        st.plotly_chart(fig_wf)

# ══════════════════════════════════════════════
# TAB 2 — SETUP / INIT
# ══════════════════════════════════════════════
with tab_init:
    st.markdown("### ⚙️ Initialise Portfolio")
    st.info("This sets today's prices as your **buy prices** and calculates shares based on equal allocation. "
            "Run this once when you first invest.")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Option A — Auto-load from Cache (Top N)")
        st.caption("Picks the top N stocks from your screener cache and initialises them all at once.")
        auto_n = st.slider("Number of stocks", 5, 20, 10, key="auto_n")

        if st.button("🚀 Auto-init from screener top picks"):
            if not os.path.exists("nifty500_cache.csv"):
                st.error("No cache found. Run the Screener first.")
            else:
                df_c = load_cache()
                if "score" not in df_c.columns:
                    df_c["score"] = df_c.apply(lambda r: multibagger_score(r.to_dict()), axis=1)
                df_top = df_c.sort_values("score",ascending=False).head(auto_n)
                tks    = df_top["Yahoo Symbol"].tolist()
                alloc  = total_inv / auto_n
                with st.spinner("Fetching buy prices..."):
                    pr = fetch_prices(tks)
                new_pos = {}
                for _, row in df_top.iterrows():
                    t = row["Yahoo Symbol"]; p = pr.get(t)
                    if p and p>0:
                        new_pos[t] = {"name":row["Company Name"],"buy_price":round(p,4),
                                      "shares":round(alloc/p,6),"allocation":round(alloc,2),
                                      "buy_date":date.today().isoformat()}
                save_positions(new_pos)
                st.success(f"✅ Initialised {len(new_pos)} positions!")
                st.rerun()

    with col_b:
        st.markdown("#### Option B — Manual Entry")
        st.caption("Add individual stocks manually.")
        with st.form("add_stock"):
            m_name   = st.text_input("Display Name", placeholder="Kaynes Technology")
            m_ticker = st.text_input("Yahoo Ticker",  placeholder="KAYNES.NS")
            m_alloc  = st.number_input("Allocation (₹)", min_value=1000, value=50000, step=1000)
            if st.form_submit_button("➕ Add Position"):
                if m_name and m_ticker:
                    with st.spinner("Fetching price..."):
                        pr = fetch_prices([m_ticker.upper()])
                    p = pr.get(m_ticker.upper())
                    if p and p>0:
                        pos = load_positions()
                        pos[m_ticker.upper()] = {"name":m_name,"buy_price":round(p,4),
                                                  "shares":round(m_alloc/p,6),"allocation":float(m_alloc),
                                                  "buy_date":date.today().isoformat()}
                        save_positions(pos)
                        st.success(f"Added {m_name} @ ₹{p:,.2f}"); st.rerun()
                    else:
                        st.error(f"Could not fetch price for {m_ticker}")

    st.divider()
    if positions:
        st.markdown("#### Current Positions")
        pos_rows = [{"Ticker":t,"Name":v["name"],"Buy Price":v["buy_price"],
                     "Shares":v["shares"],"Allocation":v["allocation"],"Buy Date":v["buy_date"]}
                    for t,v in positions.items()]
        st.dataframe(pd.DataFrame(pos_rows), height=300)
        col_del1, col_del2 = st.columns(2)
        with col_del1:
            to_del = st.selectbox("Remove position", [p["name"] for p in positions.values()])
            if st.button("🗑️ Remove"):
                ticker_to_del = next((t for t,v in positions.items() if v["name"]==to_del), None)
                if ticker_to_del:
                    del positions[ticker_to_del]
                    save_positions(positions)
                    st.rerun()
        with col_del2:
            if st.button("🔥 Clear ALL positions", type="secondary"):
                save_positions({})
                st.rerun()

# ══════════════════════════════════════════════
# TAB 3 — HISTORY LOG
# ══════════════════════════════════════════════
with tab_history:
    if history.empty:
        st.info("No history yet. Use the **Save Today's Prices** button in the sidebar daily, "
                "or set up GitHub Actions (see README).")
    else:
        dates = sorted(history["date"].unique(), reverse=True)
        sel_date = st.selectbox("View date", dates)
        df_day = history[history["date"]==sel_date].copy()

        c1,c2,c3 = st.columns(3)
        total_inv_day  = df_day["allocation"].sum()
        total_val_day  = df_day["current_value"].sum()
        total_pnl_day  = total_val_day - total_inv_day
        sign = "+" if total_pnl_day>=0 else ""
        c1.metric("Invested",      f"₹{total_inv_day:,.0f}")
        c2.metric("Value",         f"₹{total_val_day:,.0f}")
        c3.metric("P&L",           f"{'+'if total_pnl_day>=0 else''}₹{total_pnl_day:,.0f}",
                  delta=f"{(total_pnl_day/total_inv_day*100):.2f}%" if total_inv_day else None)

        def style_pnl(val):
            if pd.isna(val): return ""
            return "color:#4ade80" if val>0 else "color:#f87171"

        display = df_day[["name","ticker","buy_price","current_price","shares",
                           "allocation","current_value","pnl_abs","pnl_pct"]].copy()
        display = display.sort_values("pnl_pct", ascending=False)
        st.dataframe(
            display.style.applymap(style_pnl, subset=["pnl_abs","pnl_pct"]).format(na_rep="—"),
            height=400
        )
        st.download_button("⬇️ Download History", history.to_csv(index=False),
                           "portfolio_history.csv", "text/csv")

# ══════════════════════════════════════════════
# TAB 4 — P&L ANALYTICS
# ══════════════════════════════════════════════
with tab_pnl:
    if history.empty:
        st.info("No history data yet.")
    else:
        # Portfolio value over time
        daily_val  = history.groupby("date")["current_value"].sum().reset_index()
        daily_cost = history.groupby("date")["allocation"].sum().reset_index()
        daily_val  = daily_val.merge(daily_cost, on="date", suffixes=("_val","_cost"))
        daily_val["pnl"]     = daily_val["current_value"] - daily_val["allocation"]
        daily_val["pnl_pct"] = daily_val["pnl"] / daily_val["allocation"] * 100

        fig_val = go.Figure()
        fig_val.add_trace(go.Scatter(x=daily_val["date"], y=daily_val["current_value"],
                                      name="Portfolio Value", line=dict(color="#38bdf8",width=2.5),
                                      fill="tozeroy", fillcolor="rgba(56,189,248,0.06)"))
        fig_val.add_trace(go.Scatter(x=daily_val["date"], y=daily_val["allocation"],
                                      name="Invested", line=dict(color="#64748b",width=1.5,dash="dash")))
        fig_val.update_layout(title="Portfolio Value Over Time",
                               paper_bgcolor="#080c14", plot_bgcolor="#080c14", font_color="#e2e8f0",
                               height=340, hovermode="x unified",
                               xaxis=dict(gridcolor="#1e2d45"), yaxis=dict(gridcolor="#1e2d45"),
                               legend=dict(bgcolor="#0d1424",bordercolor="#1e2d45"))
        st.plotly_chart(fig_val)

        # P&L % over time
        fig_pct = go.Figure(go.Scatter(
            x=daily_val["date"], y=daily_val["pnl_pct"],
            mode="lines+markers",
            line=dict(color="#4ade80" if daily_val["pnl_pct"].iloc[-1]>=0 else "#f87171", width=2),
            fill="tozeroy",
            fillcolor="rgba(74,222,128,0.06)" if daily_val["pnl_pct"].iloc[-1]>=0 else "rgba(248,113,113,0.06)",
        ))
        fig_pct.add_hline(y=0, line_color="#374151", line_width=1)
        fig_pct.update_layout(title="Portfolio Return % Over Time",
                               paper_bgcolor="#080c14", plot_bgcolor="#080c14", font_color="#e2e8f0",
                               height=280, xaxis=dict(gridcolor="#1e2d45"),
                               yaxis=dict(gridcolor="#1e2d45",title="Return %"))
        st.plotly_chart(fig_pct)

        # Per-stock performance
        st.subheader("📊 Per-Stock Return (Latest)")
        latest = history[history["date"]==history["date"].max()]
        fig_bar = go.Figure(go.Bar(
            x=latest["name"].str[:15],
            y=latest["pnl_pct"],
            marker_color=["#4ade80" if p>=0 else "#f87171" for p in latest["pnl_pct"].fillna(0)],
            text=[f"{'+'if p>=0 else ''}{p:.1f}%" for p in latest["pnl_pct"].fillna(0)],
            textposition="outside",
        ))
        fig_bar.add_hline(y=0, line_color="#374151")
        fig_bar.update_layout(paper_bgcolor="#080c14", plot_bgcolor="#080c14", font_color="#e2e8f0",
                               height=360, xaxis=dict(gridcolor="#1e2d45",tickangle=-35),
                               yaxis=dict(gridcolor="#1e2d45",title="Return %"), margin=dict(t=20))
        st.plotly_chart(fig_bar)

        # Per-stock line chart over time
        st.subheader("📈 Individual Stock Returns Over Time")
        pivot = history.pivot_table(index="date", columns="name", values="pnl_pct")
        fig_lines = go.Figure()
        for col in pivot.columns:
            last = pivot[col].dropna()
            if not last.empty:
                c = "#4ade80" if last.iloc[-1]>=0 else "#f87171"
                fig_lines.add_trace(go.Scatter(x=pivot.index, y=pivot[col], name=col[:16],
                                                mode="lines", line=dict(width=1.5, color=c), opacity=0.8))
        fig_lines.add_hline(y=0, line_color="#374151", line_width=1)
        fig_lines.update_layout(paper_bgcolor="#080c14", plot_bgcolor="#080c14", font_color="#e2e8f0",
                                  height=400, hovermode="x unified",
                                  xaxis=dict(gridcolor="#1e2d45"), yaxis=dict(gridcolor="#1e2d45",title="P&L %"),
                                  legend=dict(bgcolor="#0d1424",bordercolor="#1e2d45",font=dict(size=9)))
        st.plotly_chart(fig_lines)

st.divider()
st.caption("⚠️ For research purposes only. Not SEBI-registered investment advice.")
