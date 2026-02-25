"""
Dhruv — The North Star of Indian Stock Research
Named after Dhruva, the unwavering North Star.
Run:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(
    page_title="Dhruv — Multibagger Research",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
    section[data-testid="stSidebar"] { background: #080c14; border-right: 1px solid #1e2d45; }
    section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
    div[data-testid="metric-container"] { background: #0d1424; border-radius: 12px; padding: 12px 18px; border: 1px solid #1e2d45; }
    .metric-card { background: linear-gradient(135deg, #0d1424 0%, #111827 100%); border-radius: 14px; padding: 16px 20px; border-left: 4px solid #3b82f6; margin-bottom: 10px; }
    .alloc-card { background: linear-gradient(135deg, #0d1424 0%, #111827 100%); border: 1px solid #1e2d45; border-radius: 16px; padding: 20px 24px; margin-bottom: 12px; transition: border-color 0.2s; }
    .alloc-card:hover { border-color: #3b82f6; }
    .alloc-card .rank { font-family:'DM Mono',monospace; font-size:11px; color:#4b5563; letter-spacing:2px; text-transform:uppercase; margin-bottom:4px; }
    .alloc-card .stock-name { font-size:16px; font-weight:700; color:#f1f5f9; }
    .alloc-card .stock-meta { font-family:'DM Mono',monospace; font-size:11px; color:#6b7280; }
    .alloc-card .alloc-amt  { font-family:'DM Mono',monospace; font-size:22px; font-weight:500; color:#38bdf8; margin-top:8px; }
    .alloc-card .alloc-pct  { font-family:'DM Mono',monospace; font-size:13px; color:#64748b; }
    .alloc-bar { height:4px; border-radius:2px; margin-top:12px; background:linear-gradient(90deg,#3b82f6,#06b6d4); }
    .score-pill { display:inline-block; padding:2px 10px; border-radius:20px; font-family:'DM Mono',monospace; font-size:11px; font-weight:500; margin-top:6px; }
    .score-high { background:#052e16; color:#4ade80; border:1px solid #166534; }
    .score-mid  { background:#422006; color:#fb923c; border:1px solid #9a3412; }
    .score-low  { background:#3d0d15; color:#f87171; border:1px solid #991b1b; }
    .page-header { padding:24px 0 4px 0; }
    .page-header .eyebrow { font-family:'DM Mono',monospace; font-size:10px; letter-spacing:4px; color:#4b5563; text-transform:uppercase; margin-bottom:6px; }
    .page-header h1 { font-family:'Syne',sans-serif; font-size:38px; font-weight:800; color:#f8fafc; margin:0; line-height:1.1; }
    .page-header p  { color:#64748b; margin-top:8px; font-size:14px; }
    .stProgress > div > div { border-radius:8px; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div style="padding:20px 8px 8px 8px;text-align:center;">
        <div style="font-size:36px;margin-bottom:4px;">⭐</div>
        <div style="font-family:'Syne',sans-serif;font-size:22px;font-weight:800;color:#f8fafc;letter-spacing:2px;">DHRUV</div>
        <div style="font-family:'DM Mono',monospace;font-size:9px;color:#4b5563;letter-spacing:3px;text-transform:uppercase;margin-top:2px;">North Star · Nifty 500</div>
    </div>
    <hr style="border-color:#1e2d45;margin:8px 0 16px 0">
    """, unsafe_allow_html=True)

pages = {"⭐ Dhruv": [
    st.Page("pages/1_screener.py",  title="📡 Nifty 500 Screener", icon="📡"),
    st.Page("pages/2_optimizer.py", title="🎯 Portfolio Optimizer", icon="🎯"),
    st.Page("pages/3_tracker.py",   title="📈 Portfolio Tracker",   icon="📈"),
]}
pg = st.navigation(pages)
pg.run()
