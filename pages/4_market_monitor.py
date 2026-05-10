"""Page 4 — Market Monitor · Sector PE Heatmap + Stock Drill-down"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from shared.data import (
    load_cache, cache_is_fresh, multibagger_score,
    save_cache, fmt_num, score_color, kpi_components,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <div class="eyebrow">Nifty 500 · Sector Intelligence</div>
    <h1>🌐 Market Monitor</h1>
    <p>Sector-wise PE valuation vs Nifty benchmark — click any sector to drill into stocks.</p>
</div>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
# Nifty 500 trailing PE — fetched live or fallback
NIFTY_PE_FALLBACK = 24.0

@st.cache_data(ttl=3600)
def fetch_nifty_pe() -> float:
    """Try to get live Nifty 500 PE from ^CRSLDX or fall back."""
    try:
        # Nifty 500 index
        info = yf.Ticker("^CRSLDX").info
        pe   = info.get("trailingPE")
        if pe and 10 < pe < 80:
            return round(float(pe), 1)
    except Exception:
        pass
    try:
        # Fallback: Nifty 50
        info = yf.Ticker("^NSEI").info
        pe   = info.get("trailingPE")
        if pe and 10 < pe < 80:
            return round(float(pe), 1)
    except Exception:
        pass
    return NIFTY_PE_FALLBACK

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌐 Market Monitor")
    st.divider()

    nifty_pe_live = fetch_nifty_pe()
    nifty_pe = st.number_input(
        "Nifty PE Benchmark",
        min_value=5.0, max_value=100.0,
        value=float(nifty_pe_live),
        step=0.5,
        help="Sector PE is compared against this. Auto-fetched from Nifty 500 index."
    )

    st.divider()
    st.markdown("**Colour thresholds**")
    amber_thresh = st.slider(
        "Amber above Nifty PE by %", 0, 50, 20,
        help="Sectors with PE up to this % above Nifty = Amber"
    )
    # Green = ≤ nifty_pe
    # Amber = nifty_pe → nifty_pe * (1 + amber_thresh/100)
    # Red   = above amber threshold

    st.divider()
    st.markdown("**Filters**")
    min_stocks = st.slider("Min stocks in sector", 1, 20, 3,
                            help="Hide sectors with fewer than N stocks")
    show_no_pe = st.checkbox("Show sectors with no PE data", value=False)

    st.divider()
    if st.button("🔄 Refresh data"):
        st.cache_data.clear(); st.rerun()

# ── Load data ──────────────────────────────────────────────────────────────────
if not cache_is_fresh():
    st.warning("⚠️ Cache is stale. Go to **📡 Screener** and refresh data first.")
    st.stop()

df_all = load_cache()
if df_all.empty:
    st.error("No data. Run the Screener first to generate the cache.")
    st.stop()

if "score" not in df_all.columns:
    df_all["score"] = df_all.apply(lambda r: multibagger_score(r.to_dict()), axis=1)
    save_cache(df_all)

# Coerce key columns
for col in ["pe", "roe", "rev_growth", "earn_growth", "debt_equity",
            "insider_hold", "peg", "market_cap", "score", "current_price"]:
    if col in df_all.columns:
        df_all[col] = pd.to_numeric(df_all[col], errors="coerce")

# ── Sector aggregation ─────────────────────────────────────────────────────────
def pe_colour(sector_pe: float, nifty: float, amber_pct: int) -> tuple:
    """Returns (bg_hex, text_hex, label)"""
    if sector_pe <= nifty:
        return "#052e16", "#4ade80", "green"      # below nifty = green (cheaper)
    elif sector_pe <= nifty * (1 + amber_pct / 100):
        return "#422006", "#fb923c", "amber"       # slightly above = amber
    else:
        return "#3d0d15", "#f87171", "red"         # well above = expensive

# Valid PE rows only (positive, reasonable)
df_pe = df_all[df_all["pe"].notna() & (df_all["pe"] > 0) & (df_all["pe"] < 500)].copy()

sector_stats = (
    df_pe.groupby("Industry")
    .agg(
        median_pe   = ("pe",           "median"),
        mean_pe     = ("pe",           "mean"),
        stock_count = ("Company Name", "count"),
        avg_score   = ("score",        "mean"),
        avg_roe     = ("roe",          "mean"),
        avg_rev_gr  = ("rev_growth",   "mean"),
        total_mcap  = ("market_cap",   "sum"),
    )
    .reset_index()
    .rename(columns={"Industry": "sector"})
)

# Also add total stock count (including those without PE)
total_counts = df_all.groupby("Industry")["Company Name"].count().reset_index()
total_counts.columns = ["sector", "total_stocks"]
sector_stats = sector_stats.merge(total_counts, on="sector", how="left")

# Filter min stocks
sector_stats = sector_stats[sector_stats["total_stocks"] >= min_stocks]
sector_stats = sector_stats.sort_values("median_pe", ascending=False).reset_index(drop=True)

if not show_no_pe:
    sector_stats = sector_stats[sector_stats["stock_count"] >= 1]

# Add colour info
sector_stats["bg"]    = sector_stats["median_pe"].apply(lambda p: pe_colour(p, nifty_pe, amber_thresh)[0])
sector_stats["fg"]    = sector_stats["median_pe"].apply(lambda p: pe_colour(p, nifty_pe, amber_thresh)[1])
sector_stats["label"] = sector_stats["median_pe"].apply(lambda p: pe_colour(p, nifty_pe, amber_thresh)[2])

n_green = (sector_stats["label"] == "green").sum()
n_amber = (sector_stats["label"] == "amber").sum()
n_red   = (sector_stats["label"] == "red").sum()

# ── Top metrics row ────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Nifty PE",         f"{nifty_pe:.1f}",  help="Benchmark (auto-fetched or manual)")
c2.metric("Sectors tracked",  len(sector_stats))
c3.metric("🟢 Cheap sectors", n_green,  help=f"Median PE ≤ {nifty_pe:.1f}")
c4.metric("🟡 Fair sectors",  n_amber,  help=f"PE within {amber_thresh}% above Nifty")
c5.metric("🔴 Pricey sectors",n_red,    help=f"PE > {nifty_pe*(1+amber_thresh/100):.1f}")
c6.metric("Stocks universe",  len(df_all))
st.divider()

# ── Legend ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex;gap:20px;margin-bottom:16px;font-family:'DM Mono',monospace;font-size:11px;">
    <span style="background:#052e16;color:#4ade80;padding:3px 12px;border-radius:20px;border:1px solid #166534;">
        🟢 CHEAP — PE ≤ {nifty_pe:.1f}
    </span>
    <span style="background:#422006;color:#fb923c;padding:3px 12px;border-radius:20px;border:1px solid #9a3412;">
        🟡 FAIR — PE {nifty_pe:.1f}–{nifty_pe*(1+amber_thresh/100):.1f}
    </span>
    <span style="background:#3d0d15;color:#f87171;padding:3px 12px;border-radius:20px;border:1px solid #991b1b;">
        🔴 PRICEY — PE &gt; {nifty_pe*(1+amber_thresh/100):.1f}
    </span>
    <span style="color:#64748b;">Benchmark: Nifty PE {nifty_pe:.1f}</span>
</div>
""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_heat, tab_chart, tab_drill = st.tabs([
    "🗺️ Sector Heatmap",
    "📊 PE Chart",
    "🔬 Sector Drill-down",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — HEATMAP GRID
# ══════════════════════════════════════════════════════════════════════════════
with tab_heat:
    st.markdown("#### Click a sector card to drill into its stocks →  🔬 Sector Drill-down tab")
    st.caption("Cards sorted by median PE (highest first). Size reflects market cap.")

    cols_per_row = 4
    rows = [sector_stats.iloc[i:i+cols_per_row]
            for i in range(0, len(sector_stats), cols_per_row)]

    for row_df in rows:
        cols = st.columns(cols_per_row)
        for ci, (_, sec) in enumerate(row_df.iterrows()):
            pe_val   = sec["median_pe"]
            bg, fg, lbl = pe_colour(pe_val, nifty_pe, amber_thresh)
            diff_pct = ((pe_val - nifty_pe) / nifty_pe) * 100
            diff_str = f"{diff_pct:+.1f}% vs Nifty"
            mcap_str = fmt_num(sec["total_mcap"], prefix="₹") if pd.notna(sec["total_mcap"]) else "N/A"
            roe_str  = f"{sec['avg_roe']*100:.1f}%" if pd.notna(sec["avg_roe"]) else "N/A"
            sc_str   = f"{sec['avg_score']:.0f}/100"

            with cols[ci]:
                # Clickable card — pressing button sets session state
                clicked = st.button(
                    f"{sec['sector'][:28]}",
                    key=f"sec_{sec['sector']}",
                    width='stretch',
                    help=f"Click to drill into {sec['sector']} stocks"
                )
                if clicked:
                    st.session_state["selected_sector"] = sec["sector"]
                    st.session_state["drill_tab_trigger"] = True

                st.markdown(f"""
                <div style="
                    background:{bg};
                    border:1px solid {fg}33;
                    border-top:3px solid {fg};
                    border-radius:10px;
                    padding:12px 14px;
                    margin-top:-8px;
                    margin-bottom:8px;
                ">
                    <div style="font-family:'DM Mono',monospace;font-size:20px;
                                font-weight:700;color:{fg};">{pe_val:.1f}x</div>
                    <div style="font-family:'DM Mono',monospace;font-size:9px;
                                color:{fg}99;margin-bottom:6px;">{diff_str}</div>
                    <div style="display:flex;justify-content:space-between;
                                font-size:10px;color:#94a3b8;">
                        <span>📦 {int(sec['total_stocks'])} stocks</span>
                        <span>⭐ {sc_str}</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;
                                font-size:10px;color:#64748b;margin-top:3px;">
                        <span>ROE {roe_str}</span>
                        <span>{mcap_str}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    if st.session_state.get("drill_tab_trigger"):
        st.session_state["drill_tab_trigger"] = False
        st.info(f"📌 **{st.session_state['selected_sector']}** selected → switch to 🔬 **Sector Drill-down** tab to see stocks.", icon="👆")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PE BAR CHART
# ══════════════════════════════════════════════════════════════════════════════
with tab_chart:
    st.markdown("#### Sector Median PE vs Nifty Benchmark")

    sort_by = st.radio("Sort by", ["PE (highest first)", "PE (lowest first)",
                                    "Avg KPI Score", "Stock Count"],
                        horizontal=True)
    if sort_by == "PE (highest first)":
        plot_df = sector_stats.sort_values("median_pe", ascending=True)
    elif sort_by == "PE (lowest first)":
        plot_df = sector_stats.sort_values("median_pe", ascending=False)
    elif sort_by == "Avg KPI Score":
        plot_df = sector_stats.sort_values("avg_score", ascending=True)
    else:
        plot_df = sector_stats.sort_values("total_stocks", ascending=True)

    bar_colors = plot_df["fg"].tolist()

    fig_bar = go.Figure()

    # Sector PE bars
    fig_bar.add_trace(go.Bar(
        y=plot_df["sector"],
        x=plot_df["median_pe"],
        orientation="h",
        marker_color=bar_colors,
        marker_line_width=0,
        text=[f"{v:.1f}x" for v in plot_df["median_pe"]],
        textposition="outside",
        textfont=dict(size=9, color="#e2e8f0"),
        name="Sector PE",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Median PE: %{x:.1f}x<br>"
            "<extra></extra>"
        )
    ))

    # Nifty PE reference line
    fig_bar.add_vline(
        x=nifty_pe, line_dash="dash", line_color="#38bdf8", line_width=1.5,
        annotation_text=f"Nifty {nifty_pe:.1f}x",
        annotation_font_color="#38bdf8", annotation_font_size=10,
        annotation_position="top right"
    )

    # Amber threshold line
    amber_line = nifty_pe * (1 + amber_thresh / 100)
    fig_bar.add_vline(
        x=amber_line, line_dash="dot", line_color="#fb923c", line_width=1,
        annotation_text=f"Amber {amber_line:.1f}x",
        annotation_font_color="#fb923c", annotation_font_size=9,
        annotation_position="bottom right"
    )

    fig_bar.update_layout(
        paper_bgcolor="#080c14", plot_bgcolor="#080c14",
        font_color="#e2e8f0", height=max(420, len(plot_df) * 22),
        margin=dict(l=10, r=60, t=20, b=20),
        xaxis=dict(gridcolor="#1e2d45", title="Median PE",
                   title_font_color="#64748b"),
        yaxis=dict(gridcolor="#1e2d45", tickfont=dict(size=9)),
        showlegend=False,
        bargap=0.3,
    )
    st.plotly_chart(fig_bar, width='stretch')

    # Bubble chart: PE vs ROE sized by market cap
    st.markdown("#### PE vs ROE — Bubble = Market Cap")
    st.caption("Ideal: bottom-right (low PE, high ROE). Top-left is expensive with poor returns.")

    bubble_df = sector_stats[sector_stats["avg_roe"].notna()].copy()
    bubble_df["mcap_norm"] = bubble_df["total_mcap"].fillna(bubble_df["total_mcap"].median())

    fig_bub = go.Figure()
    for _, row in bubble_df.iterrows():
        bg, fg, lbl = pe_colour(row["median_pe"], nifty_pe, amber_thresh)
        size = max(12, min(55, row["mcap_norm"] / 1e11))
        fig_bub.add_trace(go.Scatter(
            x=[row["median_pe"]],
            y=[row["avg_roe"] * 100 if pd.notna(row["avg_roe"]) else 0],
            mode="markers+text",
            text=[row["sector"][:14]],
            textposition="top center",
            textfont=dict(size=7, color="#94a3b8"),
            marker=dict(size=size, color=fg, opacity=0.75,
                        line=dict(color=fg, width=1)),
            name=row["sector"],
            showlegend=False,
            hovertemplate=(
                f"<b>{row['sector']}</b><br>"
                f"Median PE: {row['median_pe']:.1f}x<br>"
                f"Avg ROE: {row['avg_roe']*100:.1f}%<br>"
                f"Stocks: {int(row['total_stocks'])}<br>"
                f"Avg Score: {row['avg_score']:.0f}/100<br>"
                "<extra></extra>"
            )
        ))

    fig_bub.add_vline(x=nifty_pe, line_dash="dash", line_color="#38bdf8",
                       line_width=1, annotation_text=f"Nifty PE {nifty_pe:.1f}x",
                       annotation_font_color="#38bdf8", annotation_font_size=9)
    fig_bub.add_hline(y=15, line_dash="dot", line_color="#4ade80",
                       line_width=1, annotation_text="ROE 15%",
                       annotation_font_color="#4ade80", annotation_font_size=9)

    fig_bub.update_layout(
        paper_bgcolor="#080c14", plot_bgcolor="#080c14",
        font_color="#e2e8f0", height=480,
        xaxis=dict(gridcolor="#1e2d45", title="Median PE",
                   title_font_color="#64748b"),
        yaxis=dict(gridcolor="#1e2d45", title="Avg ROE %",
                   title_font_color="#64748b"),
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_bub, width='stretch')

    # Full sector summary table
    st.markdown("#### Full Sector Summary Table")
    tbl = sector_stats[[
        "sector", "median_pe", "stock_count", "total_stocks",
        "avg_score", "avg_roe", "avg_rev_gr", "total_mcap", "label"
    ]].copy()
    tbl.columns = ["Sector", "Median PE", "Stocks w/ PE",
                    "Total Stocks", "Avg KPI", "Avg ROE %",
                    "Avg Rev Gr %", "Total MCap", "Valuation"]
    tbl["Avg ROE %"]    = tbl["Avg ROE %"].apply(lambda x: round(x*100,1) if pd.notna(x) else None)
    tbl["Avg Rev Gr %"] = tbl["Avg Rev Gr %"].apply(lambda x: round(x*100,1) if pd.notna(x) else None)
    tbl["Total MCap"]   = tbl["Total MCap"].apply(lambda x: fmt_num(x, prefix="₹"))
    tbl["Avg KPI"]      = tbl["Avg KPI"].apply(lambda x: round(x,0) if pd.notna(x) else None)
    tbl["Median PE"]    = tbl["Median PE"].apply(lambda x: round(x,1) if pd.notna(x) else None)

    def style_valuation(val):
        if val == "green": return "background:#052e16;color:#4ade80;font-weight:bold"
        if val == "amber": return "background:#422006;color:#fb923c;font-weight:bold"
        if val == "red":   return "background:#3d0d15;color:#f87171;font-weight:bold"
        return ""

    def style_pe(val):
        if pd.isna(val): return ""
        if val <= nifty_pe: return "color:#4ade80"
        if val <= nifty_pe*(1+amber_thresh/100): return "color:#fb923c"
        return "color:#f87171"

    st.dataframe(
        tbl.sort_values("Median PE", ascending=False)
           .style.map(style_valuation, subset=["Valuation"])
                 .map(style_pe, subset=["Median PE"])
                 .format(na_rep="—"),
        height=440, width='stretch'
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SECTOR DRILL-DOWN
# ══════════════════════════════════════════════════════════════════════════════
with tab_drill:
    # Sector selector
    all_sectors = sorted(sector_stats["sector"].tolist())
    default_idx = 0
    if "selected_sector" in st.session_state:
        sel = st.session_state["selected_sector"]
        if sel in all_sectors:
            default_idx = all_sectors.index(sel)

    selected = st.selectbox(
        "Select sector", all_sectors, index=default_idx,
        help="Or click any card in the 🗺️ Heatmap tab to jump here"
    )

    # Sector summary banner
    sec_row = sector_stats[sector_stats["sector"] == selected]
    if not sec_row.empty:
        sr = sec_row.iloc[0]
        bg, fg, lbl = pe_colour(sr["median_pe"], nifty_pe, amber_thresh)
        diff_pct = ((sr["median_pe"] - nifty_pe) / nifty_pe) * 100
        emoji = "🟢" if lbl=="green" else ("🟡" if lbl=="amber" else "🔴")
        val_label = {"green":"CHEAP","amber":"FAIR","red":"PRICEY"}[lbl]

        st.markdown(f"""
        <div style="background:{bg};border:1px solid {fg}44;border-left:4px solid {fg};
                    border-radius:12px;padding:16px 20px;margin-bottom:16px;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <div style="font-size:11px;font-family:'DM Mono',monospace;
                                color:{fg}99;letter-spacing:2px;text-transform:uppercase;">
                        {emoji} {val_label} · Median PE
                    </div>
                    <div style="font-size:32px;font-weight:800;color:{fg};
                                font-family:'DM Mono',monospace;line-height:1.1;">
                        {sr['median_pe']:.1f}x
                    </div>
                    <div style="font-size:11px;color:#64748b;margin-top:4px;">
                        {diff_pct:+.1f}% vs Nifty {nifty_pe:.1f}x benchmark
                    </div>
                </div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;text-align:right;">
                    <div>
                        <div style="font-size:9px;color:#64748b;font-family:'DM Mono',monospace;">STOCKS</div>
                        <div style="font-size:20px;font-weight:700;color:#e2e8f0;">{int(sr['total_stocks'])}</div>
                    </div>
                    <div>
                        <div style="font-size:9px;color:#64748b;font-family:'DM Mono',monospace;">AVG KPI</div>
                        <div style="font-size:20px;font-weight:700;color:#e2e8f0;">{sr['avg_score']:.0f}</div>
                    </div>
                    <div>
                        <div style="font-size:9px;color:#64748b;font-family:'DM Mono',monospace;">AVG ROE</div>
                        <div style="font-size:20px;font-weight:700;color:#e2e8f0;">
                            {f"{sr['avg_roe']*100:.1f}%" if pd.notna(sr['avg_roe']) else "N/A"}
                        </div>
                    </div>
                    <div>
                        <div style="font-size:9px;color:#64748b;font-family:'DM Mono',monospace;">MCap</div>
                        <div style="font-size:16px;font-weight:700;color:#e2e8f0;">
                            {fmt_num(sr['total_mcap'], prefix="₹")}
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Stocks in sector
    df_sector = df_all[df_all["Industry"] == selected].copy()
    df_sector = df_sector.sort_values("score", ascending=False).reset_index(drop=True)

    if df_sector.empty:
        st.warning("No stocks found for this sector.")
        st.stop()

    st.markdown(f"#### {len(df_sector)} stocks in **{selected}**")

    # ── Stock KPI cards ───────────────────────────────────────────────────────
    card_cols = st.columns(4)
    for ci, (_, row) in enumerate(df_sector.iterrows()):
        sc     = int(row["score"])
        col    = score_color(sc)
        sc_cls = "score-high" if sc >= 60 else ("score-mid" if sc >= 40 else "score-low")
        pe_v   = row.get("pe")
        roe_v  = row.get("roe")
        rg_v   = row.get("rev_growth")
        de_v   = row.get("debt_equity")
        price  = row.get("current_price")

        pe_col = "#4ade80"
        if pd.notna(pe_v):
            if pe_v > nifty_pe * (1 + amber_thresh/100): pe_col = "#f87171"
            elif pe_v > nifty_pe:                         pe_col = "#fb923c"

        with card_cols[ci % 4]:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#0d1424,#111827);
                        border:1px solid #1e2d45;border-top:3px solid {col};
                        border-radius:12px;padding:14px 16px;margin-bottom:10px;">

                <div style="font-size:13px;font-weight:700;color:#f1f5f9;
                            white-space:nowrap;overflow:hidden;text-overflow:ellipsis;"
                     title="{row['Company Name']}">
                    {row['Company Name'][:22]}
                </div>
                <div style="font-family:'DM Mono',monospace;font-size:9px;
                            color:#6b7280;margin-bottom:8px;">
                    {row.get('NSE Symbol','')}&nbsp;·&nbsp;
                    {"₹{:,.0f}".format(price) if pd.notna(price) else "N/A"}
                </div>

                <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;
                            font-family:'DM Mono',monospace;font-size:9px;">
                    <div style="color:#64748b;">KPI Score</div>
                    <div style="color:{col};font-weight:700;text-align:right;">{sc}/100</div>

                    <div style="color:#64748b;">PE</div>
                    <div style="color:{pe_col};text-align:right;">
                        {"—" if pd.isna(pe_v) else f"{pe_v:.1f}x"}
                    </div>

                    <div style="color:#64748b;">ROE</div>
                    <div style="color:#e2e8f0;text-align:right;">
                        {"—" if pd.isna(roe_v) else f"{roe_v*100:.1f}%"}
                    </div>

                    <div style="color:#64748b;">Rev Gr</div>
                    <div style="color:{"#4ade80" if pd.notna(rg_v) and rg_v>0.15 else "#e2e8f0"};text-align:right;">
                        {"—" if pd.isna(rg_v) else f"{rg_v*100:.1f}%"}
                    </div>

                    <div style="color:#64748b;">D/E</div>
                    <div style="color:{"#4ade80" if pd.notna(de_v) and de_v<50 else "#f87171" if pd.notna(de_v) and de_v>=50 else "#e2e8f0"};text-align:right;">
                        {"—" if pd.isna(de_v) else f"{de_v/100:.2f}"}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ── KPI table for sector ───────────────────────────────────────────────────
    st.markdown("#### Full KPI Table")
    kpi_tbl = pd.DataFrame({
        "Company":    df_sector["Company Name"],
        "NSE":        df_sector.get("NSE Symbol", pd.Series(["N/A"]*len(df_sector))),
        "Score":      df_sector["score"].astype(int),
        "Price ₹":    df_sector["current_price"].apply(lambda x: round(float(x),1) if pd.notna(x) else None),
        "PE":         df_sector["pe"].apply(lambda x: round(float(x),1) if pd.notna(x) else None),
        "ROE %":      df_sector["roe"].apply(lambda x: round(float(x)*100,1) if pd.notna(x) else None),
        "Rev Gr %":   df_sector["rev_growth"].apply(lambda x: round(float(x)*100,1) if pd.notna(x) else None),
        "EPS Gr %":   df_sector["earn_growth"].apply(lambda x: round(float(x)*100,1) if pd.notna(x) else None),
        "D/E":        df_sector["debt_equity"].apply(lambda x: round(float(x)/100,2) if pd.notna(x) else None),
        "Promoter %": df_sector["insider_hold"].apply(lambda x: round(float(x)*100,1) if pd.notna(x) else None),
        "PEG":        df_sector["peg"].apply(lambda x: round(float(x),2) if pd.notna(x) else None),
        "MCap":       df_sector["market_cap"].apply(lambda x: fmt_num(x, prefix="₹")),
    })

    def style_score(v):
        if pd.isna(v): return ""
        if v >= 60: return "background:#052e16;color:#4ade80;font-weight:bold"
        if v >= 40: return "background:#422006;color:#fb923c;font-weight:bold"
        return "background:#3d0d15;color:#f87171"

    def style_pe_cell(v):
        if pd.isna(v): return ""
        if v <= nifty_pe: return "color:#4ade80"
        if v <= nifty_pe*(1+amber_thresh/100): return "color:#fb923c"
        return "color:#f87171"

    def style_pos(v):
        if pd.isna(v): return ""
        return "color:#4ade80" if v > 0 else "color:#f87171"

    st.dataframe(
        kpi_tbl.style
               .map(style_score,   subset=["Score"])
               .map(style_pe_cell, subset=["PE"])
               .map(style_pos,     subset=["Rev Gr %", "EPS Gr %"])
               .format(na_rep="—"),
        height=420, width='stretch'
    )

    # ── PE distribution chart for sector ──────────────────────────────────────
    pe_valid = df_sector["pe"].dropna()
    pe_valid = pe_valid[(pe_valid > 0) & (pe_valid < 300)]
    if len(pe_valid) >= 3:
        st.markdown("#### PE Distribution within Sector")
        fig_pe = go.Figure()
        fig_pe.add_trace(go.Histogram(
            x=pe_valid, nbinsx=15,
            marker_color="#3b82f6", opacity=0.8, name="Stock PE"
        ))
        fig_pe.add_vline(x=nifty_pe, line_dash="dash", line_color="#38bdf8",
                          annotation_text=f"Nifty {nifty_pe:.1f}x",
                          annotation_font_color="#38bdf8")
        fig_pe.add_vline(x=pe_valid.median(), line_dash="dot", line_color="#4ade80",
                          annotation_text=f"Sector median {pe_valid.median():.1f}x",
                          annotation_font_color="#4ade80",
                          annotation_position="top left")
        fig_pe.update_layout(
            paper_bgcolor="#080c14", plot_bgcolor="#080c14",
            font_color="#e2e8f0", height=280,
            xaxis=dict(gridcolor="#1e2d45", title="PE Ratio"),
            yaxis=dict(gridcolor="#1e2d45", title="# Stocks"),
            margin=dict(t=20, b=20), showlegend=False
        )
        st.plotly_chart(fig_pe, width='stretch')

    # ── Radar comparison of top 5 in sector ───────────────────────────────────
    top5 = df_sector.head(5)
    if len(top5) >= 2:
        st.markdown("#### KPI Radar — Top 5 by Score")
        radar_cats = ["Rev Growth","EPS Growth","ROE","Low Debt","Valuation","Promoter"]
        max_vals   = [20, 20, 15, 15, 15, 15]
        radar_cols = st.columns(min(5, len(top5)))
        colors_r   = ["#3b82f6","#10b981","#f59e0b","#8b5cf6","#ef4444"]

        for i, (_, row) in enumerate(top5.iterrows()):
            comps = kpi_components(row.to_dict())
            vals  = [comps[c] for c in radar_cats]
            norm  = [v/m*100 for v,m in zip(vals,max_vals)]
            norm_closed = norm + [norm[0]]
            cats_closed = radar_cats + [radar_cats[0]]
            c     = colors_r[i % len(colors_r)]
            fig_r = go.Figure(go.Scatterpolar(
                r=norm_closed, theta=cats_closed, fill="toself",
                fillcolor="rgba({},{},{},0.13)".format(int(c[1:3],16),int(c[3:5],16),int(c[5:7],16)), line_color=c, line_width=2,
            ))
            fig_r.update_layout(
                polar=dict(
                    bgcolor="#0d1424",
                    radialaxis=dict(visible=True, range=[0,100],
                                    gridcolor="#1e2d45", tickfont=dict(size=7)),
                    angularaxis=dict(gridcolor="#1e2d45", tickfont=dict(size=8))
                ),
                paper_bgcolor="#080c14", font_color="#e2e8f0",
                height=200, margin=dict(t=36,b=8,l=24,r=24),
                showlegend=False,
                title=dict(text=f"<b>{row['Company Name'][:14]}</b>",
                           font=dict(size=9), x=0.5)
            )
            with radar_cols[i % min(5, len(top5))]:
                st.plotly_chart(fig_r, width='stretch')

st.divider()
st.caption("⚠️ For research purposes only. Not SEBI-registered investment advice. PE comparisons use trailing 12-month data.")
