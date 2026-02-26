"""Page 1 — Nifty 500 Screener"""

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from shared.data import (
    load_nifty500_symbols, load_cache, save_cache, cache_is_fresh,
    fetch_all_nifty500, fetch_price_history, multibagger_score,
    multibagger_score_detail, fmt_num, score_color, NUMERIC_COLS,
)

# ── Header ───────────────────────────────────
st.markdown("""
<div class="page-header">
    <div class="eyebrow">Nifty 500 · Live Screener</div>
    <h1>📡 Stock Screener</h1>
    <p>Score all 500 stocks on multibagger KPIs — filtered, ranked, ready to act on.</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────
with st.sidebar:
    st.markdown("### 📡 Screener Settings")
    st.divider()
    st.subheader("📦 Data")
    fresh = cache_is_fresh()
    if fresh:
        mtime = os.path.getmtime("nifty500_cache.csv")
        from datetime import datetime
        st.success(f"✅ Cache fresh\n{datetime.fromtimestamp(mtime).strftime('%d %b %Y, %H:%M')}")
    else:
        st.warning("⚠️ Cache stale or missing")
    if st.button("🔄 Refresh All Data"):
        if os.path.exists("nifty500_cache.csv"): os.remove("nifty500_cache.csv")
        st.cache_data.clear(); st.rerun()

    st.divider()
    st.subheader("🔍 Filters")
    min_score = st.slider("Min Score",    0,   100,  40)
    max_pe    = st.slider("Max PE",       0,   200, 100)
    min_roe   = st.slider("Min ROE %",    0,    50,   0)
    max_de    = st.slider("Max D/E",    0.0,   5.0, 3.0, step=0.1)
    top_n     = st.slider("Scorecard N",  5,    50,  20)

    st.divider()
    st.subheader("⚙️ Chart Period")
    chart_period = st.selectbox("Period", ["1mo","3mo","6mo","1y","2y"], index=3)

# ── Load data ────────────────────────────────
symbols_df = load_nifty500_symbols()
if cache_is_fresh():
    df_cache = load_cache()
else:
    df_cache = fetch_all_nifty500(symbols_df)

if "score" not in df_cache.columns:
    df_cache["score"] = df_cache.apply(lambda r: multibagger_score(r.to_dict()), axis=1)
    save_cache(df_cache)

# ── Sector filter (needs data first) ─────────
with st.sidebar:
    st.divider()
    industries = sorted(df_cache["Industry"].dropna().unique().tolist()) if "Industry" in df_cache.columns else []
    sel_ind = st.multiselect("Sector / Industry", industries, placeholder="All sectors")

# ── Apply filters ────────────────────────────
df = df_cache.copy()
df = df[df["score"] >= min_score]
if "pe" in df.columns:          df = df[df["pe"].isna()          | (df["pe"] <= max_pe)]
if "roe" in df.columns:         df = df[df["roe"].isna()         | (df["roe"] >= min_roe/100)]
if "debt_equity" in df.columns: df = df[df["debt_equity"].isna() | (df["debt_equity"] <= max_de*100)]
if sel_ind:                     df = df[df["Industry"].isin(sel_ind)]
df = df.sort_values("score", ascending=False).reset_index(drop=True)

# ── Save filtered results to session_state for Optimizer ──
st.session_state["screener_filtered_df"]     = df
st.session_state["screener_filter_active"]   = (
    min_score > 0 or max_pe < 200 or min_roe > 0 or max_de < 5.0 or bool(sel_ind)
)
st.session_state["screener_filter_summary"]  = (
    f"Score ≥ {min_score}"
    + (f" · PE ≤ {max_pe}" if max_pe < 200 else "")
    + (f" · ROE ≥ {min_roe}%" if min_roe > 0 else "")
    + (f" · D/E ≤ {max_de}" if max_de < 5.0 else "")
    + (f" · Sectors: {', '.join(sel_ind)}" if sel_ind else "")
)

# ── Quick stats ──────────────────────────────
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Universe",       len(df_cache))
c2.metric("Passing filters",len(df))
c3.metric("Score ≥ 60",     int((df["score"]>=60).sum()) if not df.empty else 0)
c4.metric("Score 40–60",    int(((df["score"]>=40)&(df["score"]<60)).sum()) if not df.empty else 0)
c5.metric("Avg Score",      f"{df['score'].mean():.0f}" if not df.empty else "—")
st.divider()

# ── Top N scorecards ─────────────────────────
st.subheader(f"🏆 Top {min(top_n, len(df))} Candidates")
top_df = df.head(top_n)
for row_start in range(0, len(top_df), 5):
    cols = st.columns(5)
    for ci, (_, row) in enumerate(top_df.iloc[row_start:row_start+5].iterrows()):
        sc    = int(row["score"])
        price = row.get("current_price")
        pfh   = row.get("pct_from_high")
        col   = score_color(sc)
        with cols[ci]:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color:{col}">
                <div style="font-size:11px;color:#6b7280;overflow:hidden;white-space:nowrap;text-overflow:ellipsis">{row['Company Name']}</div>
                <div style="font-size:24px;font-weight:800;color:{col};font-family:'DM Mono',monospace">{sc}<span style="font-size:12px;color:#374151">/100</span></div>
                <div style="font-size:13px;color:#e2e8f0">{"₹{:,.0f}".format(price) if price and pd.notna(price) else "N/A"}</div>
                <div style="font-size:11px;color:#4b5563">{"↓{:.1f}% from high".format(pfh) if pfh and pd.notna(pfh) else ""}</div>
            </div>""", unsafe_allow_html=True)
# ── Send to Optimizer ───────────────────────
col_send1, col_send2, col_send3 = st.columns([2, 2, 4])
with col_send1:
    if st.button("🎯 Send to Optimizer", type="primary", use_container_width=True):
        st.switch_page("pages/2_optimizer.py")
with col_send2:
    st.caption(
        f"**{len(df)} stocks** will flow into the Optimizer "
        + (f"with active filters" if st.session_state.get('screener_filter_active') else "")
    )

st.divider()

# ── Tabs ─────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Full Table","🔬 Deep Dive","📉 Charts","✅ Criteria","📈 Analytics"])

# ── Tab 1: Full table ────────────────────────
with tab1:
    tbl = pd.DataFrame()
    tbl["Company"]          = df["Company Name"]
    tbl["Industry"]         = df["Industry"]
    tbl["Score"]            = df["score"].astype(int)
    tbl["Price (₹)"]        = df["current_price"].apply(lambda x: round(float(x),1)   if pd.notna(x) else None)
    tbl["Mkt Cap"]          = df["market_cap"].apply(lambda x: fmt_num(x, prefix="₹"))
    tbl["PE"]               = df["pe"].apply(lambda x: round(float(x),1)               if pd.notna(x) else None)
    tbl["Fwd PE"]           = df["fwd_pe"].apply(lambda x: round(float(x),1)           if pd.notna(x) else None)
    tbl["PEG"]              = df["peg"].apply(lambda x: round(float(x),2)              if pd.notna(x) else None)
    tbl["ROE %"]            = df["roe"].apply(lambda x: round(float(x)*100,1)          if pd.notna(x) else None)
    tbl["Rev Gr%"]          = df["rev_growth"].apply(lambda x: round(float(x)*100,1)   if pd.notna(x) else None)
    tbl["EPS Gr%"]          = df["earn_growth"].apply(lambda x: round(float(x)*100,1)  if pd.notna(x) else None)
    tbl["D/E"]              = df["debt_equity"].apply(lambda x: round(float(x)/100,2)  if pd.notna(x) else None)
    tbl["Insider %"]        = df["insider_hold"].apply(lambda x: round(float(x)*100,1) if pd.notna(x) else None)
    tbl["↓ 52W High%"]      = df["pct_from_high"].apply(lambda x: round(float(x),1)   if pd.notna(x) else None)

    def sc_style(v):
        if pd.isna(v): return ""
        if v>=60: return "background:#052e16;color:#4ade80;font-weight:bold"
        if v>=40: return "background:#422006;color:#fb923c;font-weight:bold"
        return "background:#3d0d15;color:#f87171"
    def gr_style(v):
        if pd.isna(v): return ""
        return "color:#4ade80" if v>0 else "color:#f87171"

    st.dataframe(
        tbl.style.applymap(sc_style, subset=["Score"])
                 .applymap(gr_style, subset=["Rev Gr%","EPS Gr%","↓ 52W High%"])
                 .format(na_rep="—"),
        height=520
    )
    st.download_button("⬇️ Download CSV", tbl.to_csv(index=False),
                       "nifty500_screener.csv", "text/csv")

# ── Tab 2: Deep Dive ─────────────────────────
with tab2:
    if df.empty:
        st.warning("No stocks match filters.")
    else:
        sel = st.selectbox("Select stock", df["Company Name"].tolist())
        row = df[df["Company Name"]==sel].iloc[0].to_dict()
        sc, checks = multibagger_score_detail(row)
        col = score_color(sc)
        price = row.get("current_price")

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Price",      f"₹{float(price):,.1f}" if price and pd.notna(price) else "N/A")
        c2.metric("Score",      f"{sc}/100")
        c3.metric("Mkt Cap",    fmt_num(row.get("market_cap"), prefix="₹"))
        c4.metric("Industry",   row.get("Industry","N/A"))
        c5.metric("NSE",        row.get("NSE Symbol","N/A"))

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 💰 Valuation")
            for lbl, key, fmt in [("PE (Trailing)","pe",":.1f"),("PE (Forward)","fwd_pe",":.1f"),
                                   ("PEG","peg",":.2f"),("Price/Book","pb",":.2f"),("Price/Sales","ps",":.2f")]:
                v = row.get(key)
                st.markdown(f"- **{lbl}:** {format(float(v), fmt.strip(':'))}" if v and pd.notna(v) else f"- **{lbl}:** N/A")
            st.markdown("#### 📈 Growth")
            rg = row.get("rev_growth"); eg = row.get("earn_growth")
            st.markdown(f"- **Revenue Growth:** {'🟢' if rg and rg>0.15 else '🔴'} {float(rg)*100:.1f}%" if rg and pd.notna(rg) else "- **Revenue Growth:** N/A")
            st.markdown(f"- **Earnings Growth:** {'🟢' if eg and eg>0.20 else '🔴'} {float(eg)*100:.1f}%" if eg and pd.notna(eg) else "- **Earnings Growth:** N/A")
        with col2:
            st.markdown("#### 🏆 Quality")
            roe=row.get("roe"); de=row.get("debt_equity"); roa=row.get("roa")
            st.markdown(f"- **ROE:** {'🟢' if roe and roe>0.18 else '🔴'} {float(roe)*100:.1f}%" if roe and pd.notna(roe) else "- **ROE:** N/A")
            st.markdown(f"- **ROA:** {float(roa)*100:.1f}%" if roa and pd.notna(roa) else "- **ROA:** N/A")
            st.markdown(f"- **D/E:** {'🟢' if de and de<50 else '🔴'} {float(de)/100:.2f}" if de and pd.notna(de) else "- **D/E:** N/A")
            st.markdown(f"- **FCF:** {fmt_num(row.get('free_cashflow'), prefix='₹')}")
            st.markdown("#### 👥 Ownership")
            ins=row.get("insider_hold"); ist=row.get("inst_holding")
            st.markdown(f"- **Promoter:** {'🟢' if ins and ins>0.50 else '🔴'} {float(ins)*100:.1f}%" if ins and pd.notna(ins) else "- **Promoter:** N/A")
            st.markdown(f"- **Institutional:** {float(ist)*100:.1f}%" if ist and pd.notna(ist) else "- **Institutional:** N/A")
            st.markdown("#### 📊 52-Week")
            h52=row.get("52w_high"); l52=row.get("52w_low"); pfh=row.get("pct_from_high")
            st.markdown(f"- **High:** ₹{float(h52):,.1f}" if h52 and pd.notna(h52) else "- **High:** N/A")
            st.markdown(f"- **Low:** ₹{float(l52):,.1f}"  if l52 and pd.notna(l52) else "- **Low:** N/A")
            st.markdown(f"- **From High:** {float(pfh):.1f}%" if pfh and pd.notna(pfh) else "")

        st.divider()
        st.markdown("#### 🎯 Criteria Check")
        for icon, text in checks:
            st.markdown(f"{icon} &nbsp; {text}", unsafe_allow_html=True)

        fig_g = go.Figure(go.Indicator(
            mode="gauge+number", value=sc,
            title={"text":"Multibagger Score","font":{"size":14}},
            gauge={"axis":{"range":[0,100]},"bar":{"color":col},
                   "steps":[{"range":[0,40],"color":"#3d0d15"},{"range":[40,60],"color":"#422006"},{"range":[60,100],"color":"#052e16"}],
                   "threshold":{"line":{"color":"white","width":2},"value":60}}
        ))
        fig_g.update_layout(height=240, paper_bgcolor="#080c14", font_color="white")
        st.plotly_chart(fig_g)

# ── Tab 3: Charts ────────────────────────────
with tab3:
    sel_stocks = st.multiselect("Select stocks", df["Company Name"].tolist(), default=df["Company Name"].head(5).tolist())
    if sel_stocks:
        tick_map = df[df["Company Name"].isin(sel_stocks)].set_index("Company Name")["Yahoo Symbol"].to_dict()
        fig = go.Figure()
        for nm in sel_stocks:
            tk = tick_map.get(nm)
            if not tk: continue
            try:
                h = yf.Ticker(tk).history(period=chart_period)
                if not h.empty:
                    norm = (h["Close"]/h["Close"].iloc[0])*100
                    fig.add_trace(go.Scatter(x=h.index, y=norm, name=nm, mode="lines", line={"width":2}))
            except Exception as e:
                st.warning(f"Could not load chart for {nm}: {e}")
        fig.update_layout(title=f"Normalised Performance (Base=100) — {chart_period}",
                          paper_bgcolor="#080c14", plot_bgcolor="#080c14", font_color="white",
                          height=460, hovermode="x unified",
                          xaxis=dict(gridcolor="#1e2d45"), yaxis=dict(gridcolor="#1e2d45"),
                          legend=dict(bgcolor="#0d1424", bordercolor="#1e2d45"))
        st.plotly_chart(fig)

        st.subheader("📊 Candlestick")
        candle = st.selectbox("Stock", sel_stocks, key="candle")
        try:
            h = yf.Ticker(tick_map[candle]).history(period=chart_period)
            if not h.empty:
                fig2 = go.Figure(go.Candlestick(x=h.index, open=h["Open"], high=h["High"],
                                                low=h["Low"], close=h["Close"], name=candle,
                                                increasing_line_color="#4ade80", decreasing_line_color="#f87171"))
                fig2.add_trace(go.Bar(x=h.index, y=h["Volume"], name="Volume", yaxis="y2",
                                      marker_color="rgba(59,130,246,0.2)"))
                fig2.update_layout(title=candle, paper_bgcolor="#080c14", plot_bgcolor="#080c14",
                                   font_color="white", height=480, xaxis_rangeslider_visible=False,
                                   xaxis=dict(gridcolor="#1e2d45"), yaxis=dict(gridcolor="#1e2d45"),
                                   yaxis2={"overlaying":"y","side":"right","showgrid":False})
                st.plotly_chart(fig2)
        except Exception as e:
            st.warning(str(e))

# ── Tab 4: Criteria ──────────────────────────
with tab4:
    st.markdown(f"### Top {min(50,len(df))} — Criteria Breakdown")
    for _, row in df.head(50).iterrows():
        sc, checks = multibagger_score_detail(row.to_dict())
        with st.expander(f"{row['Company Name']}  ·  {sc}/100  ·  {row.get('Industry','')}"):
            for icon, text in checks:
                st.markdown(f"{icon} &nbsp; {text}", unsafe_allow_html=True)
            st.caption(f"{row.get('Yahoo Symbol','')} | NSE: {row.get('NSE Symbol','')}")

# ── Tab 5: Analytics ─────────────────────────
with tab5:
    if df.empty:
        st.warning("No data.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            fig_h = go.Figure(go.Histogram(x=df_cache["score"], nbinsx=20,
                                           marker_color="#3b82f6", opacity=0.85))
            fig_h.add_vline(x=60, line_dash="dash", line_color="#4ade80", annotation_text="Strong (60+)")
            fig_h.add_vline(x=40, line_dash="dash", line_color="#fb923c", annotation_text="Watch (40+)")
            fig_h.update_layout(title="Score Distribution — All Nifty 500",
                                 paper_bgcolor="#080c14", plot_bgcolor="#080c14",
                                 font_color="white", height=340, showlegend=False,
                                 xaxis=dict(gridcolor="#1e2d45"), yaxis=dict(gridcolor="#1e2d45"))
            st.plotly_chart(fig_h)
        with c2:
            ind_c = df["Industry"].value_counts().head(15)
            fig_b = go.Figure(go.Bar(x=ind_c.values, y=ind_c.index, orientation="h", marker_color="#3b82f6"))
            fig_b.update_layout(title="Industries (Filtered)", paper_bgcolor="#080c14",
                                 plot_bgcolor="#080c14", font_color="white", height=340,
                                 yaxis={"autorange":"reversed"}, xaxis=dict(gridcolor="#1e2d45"))
            st.plotly_chart(fig_b)

        st.subheader("ROE vs Revenue Growth")
        sc_df = df[df["roe"].notna() & df["rev_growth"].notna()].copy()
        fig_s = go.Figure()
        for _, r in sc_df.iterrows():
            s = int(r["score"]); c = "#4ade80" if s>=60 else ("#fb923c" if s>=40 else "#3b82f6")
            ms = max(6, min(28, float(r["market_cap"])/1e9*0.5)) if pd.notna(r.get("market_cap")) else 8
            fig_s.add_trace(go.Scatter(
                x=[float(r["rev_growth"])*100], y=[float(r["roe"])*100], mode="markers",
                marker=dict(size=ms, color=c, opacity=0.7, line=dict(color="white",width=0.5)),
                text=f"{r['Company Name']}<br>Score:{s}", hoverinfo="text", showlegend=False,
            ))
        fig_s.add_hline(y=18, line_dash="dash", line_color="#fb923c", annotation_text="ROE 18%")
        fig_s.add_vline(x=15, line_dash="dash", line_color="#fb923c", annotation_text="RevGr 15%")
        fig_s.update_layout(xaxis_title="Rev Growth %", yaxis_title="ROE %",
                             paper_bgcolor="#080c14", plot_bgcolor="#080c14", font_color="white",
                             height=480, xaxis=dict(gridcolor="#1e2d45"), yaxis=dict(gridcolor="#1e2d45"))
        st.plotly_chart(fig_s)

        top20 = df.head(20)
        fig_t = go.Figure(go.Bar(x=top20["score"], y=top20["Company Name"], orientation="h",
                                  marker_color=[score_color(s) for s in top20["score"]],
                                  text=top20["score"], textposition="outside"))
        fig_t.update_layout(paper_bgcolor="#080c14", plot_bgcolor="#080c14", font_color="white",
                             height=560, xaxis=dict(gridcolor="#1e2d45"),
                             yaxis={"autorange":"reversed"}, margin=dict(l=220))
        st.plotly_chart(fig_t)

st.divider()
st.caption("⚠️ For research purposes only. Not SEBI-registered investment advice.")
