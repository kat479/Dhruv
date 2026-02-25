"""Page 2 — Portfolio Optimizer"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
from datetime import datetime
import os, sys, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from shared.data import (
    load_cache, cache_is_fresh, multibagger_score, kpi_components,
    fetch_price_history, portfolio_stats, fmt_num, score_color, RISK_FREE_RATE,
)

# ── Header ───────────────────────────────────
st.markdown("""
<div class="page-header">
    <div class="eyebrow">Modern Portfolio Theory · KPI Fusion</div>
    <h1>🎯 Portfolio Optimizer</h1>
    <p>Pick your top stocks and find the mathematically optimal allocation.</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────
with st.sidebar:
    st.markdown("### 🎯 Optimizer Settings")
    st.divider()
    total_inv = st.number_input("Total Investment (₹)", min_value=100_000,
                                 max_value=100_000_000, value=1_000_000, step=100_000, format="%d")
    top_n     = st.slider("Number of stocks", 5, 20, 10)
    st.divider()
    strategy = st.radio("Allocation strategy", [
        "🔀 Hybrid (KPI + Sharpe)",
        "🏆 KPI Score Weighted",
        "📐 Sharpe Optimized (MPT)",
        "⚖️ Equal Weight",
        "🛡️ Min Volatility",
    ])
    kpi_blend = 0.5
    if "Hybrid" in strategy:
        kpi_blend = st.slider("KPI weight in blend", 0.1, 0.9, 0.5, step=0.1)
    st.divider()
    min_wt = st.slider("Min allocation %", 1, 15, 3) / 100
    max_wt = st.slider("Max allocation %", 20, 60, 40) / 100
    if st.button("🔄 Reload"):
        st.cache_data.clear(); st.rerun()

# ── Load & pick top N ────────────────────────
if not cache_is_fresh():
    st.warning("⚠️ Cache is stale. Go to **📡 Screener** and refresh data first.")
    st.stop()

df_all = load_cache()
if df_all.empty:
    st.error("No cache found. Run the Screener first to generate data.")
    st.stop()

if "score" not in df_all.columns:
    df_all["score"] = df_all.apply(lambda r: multibagger_score(r.to_dict()), axis=1)

df_all = df_all[df_all["current_price"].notna() & (df_all["current_price"] > 0)]
df_top = df_all.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)

tickers = tuple(df_top["Yahoo Symbol"].tolist())
names   = df_top["Company Name"].tolist()
n       = len(tickers)

# ── Fetch price history ──────────────────────
with st.spinner(f"📡 Fetching 2Y price history for {n} stocks..."):
    prices = fetch_price_history(tickers, "2y")

valid = [t for t in tickers if t in prices.columns and prices[t].notna().sum() > 60]
if len(valid) < 3:
    st.error("Not enough price history. Try refreshing data.")
    st.stop()

prices   = prices[valid].dropna()
df_top   = df_top[df_top["Yahoo Symbol"].isin(valid)].reset_index(drop=True)
names    = df_top["Company Name"].tolist()
tickers  = valid
n        = len(tickers)

daily_ret   = prices.pct_change().dropna()
mean_ret    = daily_ret.mean().values
cov_mat     = daily_ret.cov().values

# ── Optimization helpers ─────────────────────
def opt_sharpe(bounds=(0.03,0.40)):
    def neg_s(w): return -portfolio_stats(w, mean_ret, cov_mat)["sharpe"]
    res = minimize(neg_s, [1/n]*n, method="SLSQP",
                   bounds=[bounds]*n, constraints=[{"type":"eq","fun":lambda w:w.sum()-1}],
                   options={"maxiter":1000})
    return res.x if res.success else np.array([1/n]*n)

def opt_minvol(bounds=(0.03,0.40)):
    def vol(w): return np.sqrt(w@cov_mat@w)*np.sqrt(252)
    res = minimize(vol, [1/n]*n, method="SLSQP",
                   bounds=[bounds]*n, constraints=[{"type":"eq","fun":lambda w:w.sum()-1}],
                   options={"maxiter":1000})
    return res.x if res.success else np.array([1/n]*n)

def kpi_w():
    sc = df_top["score"].fillna(0).values.astype(float)**2
    return sc / sc.sum()

eq_w      = np.array([1/n]*n)
kpi_wts   = kpi_w()
sharpe_wts= opt_sharpe((min_wt, max_wt))
minvol_wts= opt_minvol((min_wt, max_wt))
hybrid_wts= np.clip(0.5*kpi_wts + 0.5*sharpe_wts, min_wt, max_wt)
hybrid_wts= hybrid_wts / hybrid_wts.sum()

if "Hybrid"  in strategy: weights = np.clip(kpi_blend*kpi_wts+(1-kpi_blend)*sharpe_wts, min_wt, max_wt); weights/=weights.sum()
elif "KPI"   in strategy: weights = kpi_wts
elif "Sharpe"in strategy: weights = sharpe_wts
elif "Equal" in strategy: weights = eq_w
else:                     weights = minvol_wts

alloc   = weights * total_inv
shares  = alloc / df_top["current_price"].values
stats   = portfolio_stats(weights, mean_ret, cov_mat)

all_strats = {
    "Equal":   (eq_w,       portfolio_stats(eq_w,       mean_ret, cov_mat)),
    "KPI":     (kpi_wts,    portfolio_stats(kpi_wts,    mean_ret, cov_mat)),
    "Sharpe":  (sharpe_wts, portfolio_stats(sharpe_wts, mean_ret, cov_mat)),
    "Min Vol": (minvol_wts, portfolio_stats(minvol_wts, mean_ret, cov_mat)),
    "Hybrid":  (hybrid_wts, portfolio_stats(hybrid_wts, mean_ret, cov_mat)),
}

# ── Quick stats ──────────────────────────────
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Stocks",         n)
c2.metric("Exp. Return",    f"{stats['return']*100:.1f}% p.a.")
c3.metric("Volatility",     f"{stats['volatility']*100:.1f}% p.a.")
c4.metric("Sharpe Ratio",   f"{stats['sharpe']:.2f}")
c5.metric("Investment",     f"₹{total_inv:,.0f}")
st.divider()

# ── Tabs ─────────────────────────────────────
tab1,tab2,tab3,tab4,tab5 = st.tabs(["💼 Allocations","📊 Compare","🗺️ Frontier","🕸️ Deep Dive","📥 Export"])

# ── Tab 1: Allocations ───────────────────────
with tab1:
    col_left, col_right = st.columns([3, 2])
    sorted_idx = np.argsort(weights)[::-1]

    with col_left:
        st.markdown(f"#### {strategy}")
        for rank, i in enumerate(sorted_idx):
            sc    = int(df_top.iloc[i]["score"])
            price = df_top.iloc[i]["current_price"]
            pct   = weights[i]*100
            bar_w = pct/(weights.max()*100)*100
            sc_cls= "score-high" if sc>=60 else "score-mid"
            st.markdown(f"""
            <div class="alloc-card">
                <div class="rank">#{rank+1}</div>
                <div style="display:flex;justify-content:space-between;align-items:flex-start">
                    <div>
                        <div class="stock-name">{names[i]}</div>
                        <div class="stock-meta">{tickers[i]} · ₹{price:,.1f} · {shares[i]:.3f} shares</div>
                        <span class="score-pill {sc_cls}">KPI {sc}/100</span>
                    </div>
                    <div style="text-align:right">
                        <div class="alloc-amt">₹{alloc[i]:,.0f}</div>
                        <div class="alloc-pct">{pct:.1f}%</div>
                    </div>
                </div>
                <div class="alloc-bar" style="width:{bar_w:.1f}%"></div>
            </div>""", unsafe_allow_html=True)

    with col_right:
        st.markdown("#### Weights")
        fig_pie = go.Figure(go.Pie(
            labels=[names[i][:18] for i in sorted_idx],
            values=[weights[i]*100 for i in sorted_idx],
            hole=0.55, marker_colors=px.colors.qualitative.Set3[:n],
            textinfo="percent",
        ))
        fig_pie.update_layout(paper_bgcolor="#080c14", font_color="#e2e8f0", height=360,
                               legend=dict(bgcolor="#0d1424",bordercolor="#1e2d45",font=dict(size=9)),
                               annotations=[dict(text=f"₹{total_inv/1e5:.0f}L",x=0.5,y=0.5,
                                                  font_size=18,font_color="#38bdf8",showarrow=False)])
        st.plotly_chart(fig_pie)

        st.markdown("#### Risk Contribution")
        port_var = weights @ cov_mat @ weights
        risk_c   = weights * (cov_mat @ weights) / port_var * 100
        fig_r = go.Figure(go.Bar(
            x=[names[i][:12] for i in sorted_idx],
            y=[risk_c[i] for i in sorted_idx],
            marker_color=["#ef4444" if risk_c[i]>15 else "#3b82f6" for i in sorted_idx],
            text=[f"{risk_c[i]:.1f}%" for i in sorted_idx], textposition="outside",
        ))
        fig_r.update_layout(paper_bgcolor="#080c14", plot_bgcolor="#080c14", font_color="#e2e8f0",
                             height=260, margin=dict(t=10),
                             xaxis=dict(gridcolor="#1e2d45",tickangle=-35),
                             yaxis=dict(gridcolor="#1e2d45",title="% of risk"))
        st.plotly_chart(fig_r)

# ── Tab 2: Compare ───────────────────────────
with tab2:
    rows = [{"Strategy":k,"Return %":round(v[1]["return"]*100,2),"Vol %":round(v[1]["volatility"]*100,2),
             "Sharpe":round(v[1]["sharpe"],3),"Max Wt%":round(v[0].max()*100,1),"Min Wt%":round(v[0].min()*100,1)}
            for k,v in all_strats.items()]
    df_cmp = pd.DataFrame(rows)

    def hl(col):
        s = [""]*len(col)
        if col.name in ["Return %","Sharpe"]: s[col.idxmax()] = "background:#052e16;color:#4ade80;font-weight:bold"
        elif col.name == "Vol %":             s[col.idxmin()] = "background:#052e16;color:#4ade80;font-weight:bold"
        return s

    st.dataframe(df_cmp.style.apply(hl), height=220)

    fig_cmp = go.Figure()
    colors  = ["#64748b","#3b82f6","#10b981","#f59e0b","#8b5cf6"]
    for (sn, (sw,_)), col in zip(all_strats.items(), colors):
        fig_cmp.add_trace(go.Bar(name=sn, x=[names[i][:12] for i in range(n)],
                                  y=[round(sw[i]*100,1) for i in range(n)], marker_color=col))
    fig_cmp.update_layout(barmode="group", paper_bgcolor="#080c14", plot_bgcolor="#080c14",
                           font_color="#e2e8f0", height=380,
                           xaxis=dict(gridcolor="#1e2d45",tickangle=-35),
                           yaxis=dict(gridcolor="#1e2d45",title="Weight %"),
                           legend=dict(bgcolor="#0d1424",bordercolor="#1e2d45"))
    st.plotly_chart(fig_cmp)

    fig_rv = go.Figure()
    sc_colors = {"Equal":"#64748b","KPI":"#3b82f6","Sharpe":"#10b981","Min Vol":"#f59e0b","Hybrid":"#8b5cf6"}
    for (sn,(sw,ss)), col in zip(all_strats.items(), sc_colors.values()):
        is_cur = sn.lower() in strategy.lower()
        fig_rv.add_trace(go.Scatter(
            x=[ss["volatility"]*100], y=[ss["return"]*100], mode="markers+text",
            name=sn, text=[sn], textposition="top center",
            marker=dict(size=18 if is_cur else 11, color=col,
                        symbol="star" if is_cur else "circle",
                        line=dict(color="white", width=2 if is_cur else 0))
        ))
    fig_rv.update_layout(xaxis_title="Volatility %", yaxis_title="Return %",
                          paper_bgcolor="#080c14", plot_bgcolor="#080c14", font_color="#e2e8f0",
                          height=380, xaxis=dict(gridcolor="#1e2d45"), yaxis=dict(gridcolor="#1e2d45"),
                          legend=dict(bgcolor="#0d1424",bordercolor="#1e2d45"))
    st.plotly_chart(fig_rv)

# ── Tab 3: Efficient Frontier ────────────────
with tab3:
    st.caption("The curve shows every possible optimal portfolio. Stars = your strategies. "
               "Top-left = better (higher return, lower risk).")

    with st.spinner("Computing efficient frontier..."):
        ef_vols, ef_rets = [], []
        for target in np.linspace(mean_ret.min()*252, mean_ret.max()*252, 150):
            res = minimize(lambda w: np.sqrt(w@cov_mat@w)*np.sqrt(252), [1/n]*n, method="SLSQP",
                           bounds=[(0,1)]*n,
                           constraints=[{"type":"eq","fun":lambda w:w.sum()-1},
                                        {"type":"eq","fun":lambda w,t=target:np.dot(w,mean_ret)*252-t}],
                           options={"maxiter":300})
            if res.success: ef_vols.append(res.fun); ef_rets.append(target)

    fig_ef = go.Figure()
    if ef_vols:
        fig_ef.add_trace(go.Scatter(x=np.array(ef_vols)*100, y=np.array(ef_rets)*100, mode="lines",
                                     name="Efficient Frontier", line=dict(color="#38bdf8",width=2.5),
                                     fill="tozeroy", fillcolor="rgba(56,189,248,0.04)"))
    for i,(t,nm) in enumerate(zip(tickers,names)):
        sr=mean_ret[i]*252; sv=np.sqrt(cov_mat[i,i])*np.sqrt(252)
        fig_ef.add_trace(go.Scatter(x=[sv*100],y=[sr*100],mode="markers+text",text=[nm[:10]],
                                     textposition="top right",textfont=dict(size=8,color="#94a3b8"),
                                     marker=dict(size=7,color="#475569"),showlegend=False))
    for (sn,(sw,ss)),col in zip(all_strats.items(),sc_colors.values()):
        is_cur = sn.lower() in strategy.lower()
        fig_ef.add_trace(go.Scatter(x=[ss["volatility"]*100],y=[ss["return"]*100],mode="markers",
                                     name=sn,marker=dict(size=18 if is_cur else 11,color=col,
                                     symbol="star" if is_cur else "diamond",
                                     line=dict(color="white",width=2 if is_cur else 0.5))))
    fig_ef.update_layout(xaxis_title="Volatility %", yaxis_title="Return %",
                          paper_bgcolor="#080c14", plot_bgcolor="#080c14", font_color="#e2e8f0",
                          height=520, xaxis=dict(gridcolor="#1e2d45"), yaxis=dict(gridcolor="#1e2d45"),
                          legend=dict(bgcolor="#0d1424",bordercolor="#1e2d45",font=dict(size=10)))
    st.plotly_chart(fig_ef)

    st.subheader("🔥 Correlation Heatmap")
    corr = daily_ret.corr()
    short = [nm[:12] for nm in names]
    fig_hm = go.Figure(go.Heatmap(z=corr.values, x=short, y=short, colorscale="RdBu_r",
                                   zmid=0, zmin=-1, zmax=1, text=np.round(corr.values,2),
                                   texttemplate="%{text}", textfont=dict(size=9)))
    fig_hm.update_layout(paper_bgcolor="#080c14", plot_bgcolor="#080c14", font_color="#e2e8f0",
                          height=460, xaxis=dict(tickangle=-35), margin=dict(b=100,t=20))
    st.plotly_chart(fig_hm)

# ── Tab 4: Deep Dive ─────────────────────────
with tab4:
    st.subheader("KPI Radar — All Stocks")
    radar_cats = ["Rev Growth","EPS Growth","ROE","Low Debt","Valuation","Promoter"]
    max_vals   = [20,20,15,15,15,15]
    cols_r = st.columns(min(5,n))
    for i,(_, row) in enumerate(df_top.iterrows()):
        comps = kpi_components(row.to_dict())
        vals  = [comps[c] for c in radar_cats]
        norm  = [v/m*100 for v,m in zip(vals,max_vals)] + [vals[0]/max_vals[0]*100]
        sc    = int(row["score"]); col = "#4ade80" if sc>=60 else "#fb923c"
        rgb   = "68,217,128" if sc>=60 else "251,146,60"
        fig_r = go.Figure(go.Scatterpolar(r=norm, theta=radar_cats+[radar_cats[0]],
                                           fill="toself", fillcolor=f"rgba({rgb},0.12)",
                                           line_color=col, line_width=2))
        fig_r.update_layout(polar=dict(bgcolor="#0d1424",
                                        radialaxis=dict(visible=True,range=[0,100],gridcolor="#1e2d45",tickfont=dict(size=7)),
                                        angularaxis=dict(gridcolor="#1e2d45",tickfont=dict(size=8))),
                             paper_bgcolor="#080c14", font_color="#e2e8f0", height=210,
                             margin=dict(t=32,b=8,l=28,r=28), showlegend=False,
                             title=dict(text=f"<b>{row['Company Name'][:14]}</b>",font=dict(size=10),x=0.5))
        with cols_r[i%min(5,n)]: st.plotly_chart(fig_r)

    st.subheader("📈 Cumulative Returns (2Y)")
    fig_cum = go.Figure()
    for i,(t,nm) in enumerate(zip(tickers,names)):
        if t in prices.columns:
            cum = (1+daily_ret[t]).cumprod()*100
            fig_cum.add_trace(go.Scatter(x=cum.index,y=cum.values,name=nm[:16],mode="lines",
                                          line=dict(width=2 if i<3 else 1),opacity=1.0 if i<3 else 0.5))
    port_cum = (1 + daily_ret[list(tickers)].dot(weights)).cumprod()*100
    fig_cum.add_trace(go.Scatter(x=port_cum.index,y=port_cum.values,name="⭐ Portfolio",
                                  line=dict(color="white",width=3,dash="dash")))
    fig_cum.add_hline(y=100,line_dash="dot",line_color="#374151")
    fig_cum.update_layout(paper_bgcolor="#080c14",plot_bgcolor="#080c14",font_color="#e2e8f0",
                           height=440,hovermode="x unified",
                           xaxis=dict(gridcolor="#1e2d45"),yaxis=dict(gridcolor="#1e2d45",title="Base=100"),
                           legend=dict(bgcolor="#0d1424",bordercolor="#1e2d45",font=dict(size=10)))
    st.plotly_chart(fig_cum)

# ── Tab 5: Export ────────────────────────────
with tab5:
    export = []
    for rank, i in enumerate(sorted_idx):
        row = df_top.iloc[i]
        export.append({
            "Rank": rank+1, "Company": names[i], "NSE Symbol": row.get("NSE Symbol",""),
            "Yahoo Symbol": tickers[i], "Industry": row.get("Industry",""),
            "KPI Score": int(row["score"]), "Buy Price (₹)": round(float(row["current_price"]),2),
            "Weight %": round(weights[i]*100,2), "Allocation (₹)": round(alloc[i],0),
            "Shares to Buy": round(shares[i],4),
            "PE":       row.get("pe"), "PEG": row.get("peg"),
            "ROE %":    round(float(row["roe"])*100,1) if pd.notna(row.get("roe")) else None,
            "Rev Gr%":  round(float(row["rev_growth"])*100,1) if pd.notna(row.get("rev_growth")) else None,
            "EPS Gr%":  round(float(row["earn_growth"])*100,1) if pd.notna(row.get("earn_growth")) else None,
            "D/E":      round(float(row["debt_equity"])/100,2) if pd.notna(row.get("debt_equity")) else None,
            "Exp Return% pa": round(mean_ret[i]*252*100,2),
            "Volatility% pa": round(np.sqrt(cov_mat[i,i])*np.sqrt(252)*100,2),
            "Strategy": strategy, "Generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        })
    df_exp = pd.DataFrame(export)
    st.dataframe(df_exp, height=380)
    st.download_button("⬇️ Download Allocation Plan",
                        df_exp.to_csv(index=False),
                        f"allocation_{datetime.now().strftime('%Y%m%d')}.csv","text/csv")

    st.divider()
    c1,c2,c3 = st.columns(3)
    c1.markdown(f"**Strategy:** {strategy}\n\n**Stocks:** {n}\n\n**Investment:** ₹{total_inv:,.0f}")
    c2.markdown(f"**Return:** {stats['return']*100:.1f}% p.a.\n\n**Volatility:** {stats['volatility']*100:.1f}% p.a.\n\n**Sharpe:** {stats['sharpe']:.2f}")
    c3.markdown(f"**Top holding:** {names[sorted_idx[0]]} ({weights[sorted_idx[0]]*100:.1f}%)\n\n**Avg KPI:** {df_top['score'].mean():.0f}/100\n\n**Date:** {datetime.now().strftime('%d %b %Y')}")

st.divider()
st.caption("⚠️ For research purposes only. Not SEBI-registered investment advice. Past returns ≠ future performance.")
