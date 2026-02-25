# ⭐ Dhruv — The North Star of Indian Stock Research

> *Named after Dhruva, the unwavering North Star of Hindu mythology.*
> *Like the star that never moves, Dhruv helps you find stocks with*
> *unshakeable fundamentals and hold conviction through the noise.*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

A free, open-source research suite for Indian multibagger investing.
Screens all **500 Nifty stocks**, scores them on 6 KPI criteria, optimises portfolio allocation using **Modern Portfolio Theory + KPI fusion**, and tracks daily P&L automatically via GitHub Actions.

---

## 🚀 Features

| Module | What it does |
|--------|-------------|
| 📡 **Nifty 500 Screener** | Scores all 500 stocks on 6 multibagger KPIs, deep-dive, charts, analytics |
| 🎯 **Portfolio Optimizer** | 5 allocation strategies (KPI, Sharpe, Hybrid, Equal, Min Vol), efficient frontier, correlation heatmap |
| 📈 **Portfolio Tracker** | Live P&L, position cards, daily history log, P&L analytics |
| 🤖 **EOD Auto-tracker** | GitHub Actions fetches closing prices every weekday, appends to CSV |

---

## 📁 Project Structure

```
dhruv/
├── app.py                          ← Streamlit entry point
├── eod_tracker.py                  ← EOD price tracker (CLI + GitHub Actions)
├── requirements.txt
├── positions.json                  ← Your portfolio positions (init once)
├── portfolio_history.csv           ← Daily EOD P&L log (auto-updated)
├── pages/
│   ├── 1_screener.py               ← Nifty 500 Screener page
│   ├── 2_optimizer.py              ← Portfolio Optimizer page
│   └── 3_tracker.py                ← Portfolio Tracker page
├── shared/
│   ├── __init__.py
│   └── data.py                     ← Shared data loading, scoring, helpers
├── .streamlit/
│   └── config.toml                 ← Dark theme config
└── .github/
    └── workflows/
        └── eod_tracker.yml         ← Daily GitHub Actions workflow
```

---

## ⚡ Quick Start

### Local

```bash
git clone https://github.com/YOUR_USERNAME/dhruv.git
cd dhruv
pip install -r requirements.txt
streamlit run app.py
```

### First run
1. Go to **📡 Nifty 500 Screener** — click **🔄 Refresh All Data** (takes ~3–5 min, cached for 24h)
2. Go to **📈 Portfolio Tracker → ⚙️ Setup / Init** — click **Auto-init from screener top picks**
3. Go to **🎯 Portfolio Optimizer** — pick your strategy and download your allocation plan

---

## ☁️ Deploy to Streamlit Cloud (Free)

1. Fork this repo to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your forked repo
4. Set **Main file path** to `app.py`
5. Click **Deploy** — live in ~2 minutes

> **Note:** Streamlit Cloud runs in a read-only filesystem. The screener cache and positions are regenerated each session. For persistent tracking, use the GitHub Actions EOD tracker instead.

---

## 🤖 GitHub Actions — Automatic EOD Tracking

The `eod_tracker.yml` workflow runs **every weekday at 4:30 PM IST** (1 hour after NSE closes).

### One-time setup

**Step 1 — Enable write permissions:**
```
Repo → Settings → Actions → General → Workflow permissions
→ Select "Read and write permissions" → Save
```

**Step 2 — Initialise positions:**

Option A — locally:
```bash
python eod_tracker.py --init         # top 10 by KPI score
python eod_tracker.py --init --top 15  # top 15
```
Then `git add positions.json && git commit -m "init positions" && git push`

Option B — via GitHub Actions UI:
```
Repo → Actions → "Dhruv · Daily EOD Tracker" → Run workflow → mode: init
```

**Step 3 — Done!** The action runs automatically. Each day it:
- Fetches NSE closing prices for your positions
- Calculates P&L vs buy prices
- Appends 1 row per stock to `portfolio_history.csv`
- Commits and pushes the updated CSV to your repo
- Prints a P&L summary in the Actions log

### Manual CLI usage

```bash
python eod_tracker.py                  # fetch today's EOD → append to CSV
python eod_tracker.py --init           # initialise from screener cache (top 10)
python eod_tracker.py --init --top 15  # initialise top 15
python eod_tracker.py --report         # print P&L table
python eod_tracker.py --stocks         # list current positions
```

---

## 📊 The Multibagger Scoring System

Each stock is scored out of **100** across 6 KPI criteria:

| Metric | Max | Full Score | Partial |
|--------|-----|-----------|---------|
| Revenue Growth (YoY TTM) | 20 | > 25% | > 15% = 12pts |
| Earnings Growth (YoY TTM) | 20 | > 25% | > 15% = 10pts |
| ROE (TTM) | 15 | > 20% | > 15% = 8pts |
| Debt/Equity (Latest Qtr) | 15 | < 0.3x | < 0.5x = 8pts |
| PEG Ratio | 15 | < 1.0 | < 1.5 = 8pts |
| Promoter Holding | 15 | > 50% | > 35% = 8pts |

**Score ≥ 60** → 🟢 Strong multibagger candidate
**Score 40–59** → 🟡 Watch list
**Score < 40** → 🔴 Doesn't meet criteria

> PEG and ROE are calculated manually when Yahoo Finance doesn't return them, using a 3-tier fallback from financial statements.

---

## 🎯 Portfolio Optimizer Strategies

| Strategy | How it works |
|----------|-------------|
| 🔀 **Hybrid (KPI + Sharpe)** | Blends KPI score weighting with Sharpe optimization (recommended) |
| 🏆 **KPI Score Weighted** | Allocates proportional to score² — top scorer gets most weight |
| 📐 **Sharpe Optimized (MPT)** | Maximises risk-adjusted return using 2Y price history |
| ⚖️ **Equal Weight** | Simple 1/N benchmark |
| 🛡️ **Min Volatility** | Minimises portfolio variance |

---

## 📈 portfolio_history.csv Schema

| Column | Description |
|--------|-------------|
| `date` | Trading date (YYYY-MM-DD) |
| `ticker` | Yahoo Finance ticker (e.g. `KAYNES.NS`) |
| `name` | Company display name |
| `kpi_score` | Multibagger score at time of init |
| `buy_price` | Price on initialisation date |
| `shares` | Units held |
| `allocation` | ₹ invested in this stock |
| `current_price` | EOD closing price |
| `current_value` | shares × current_price |
| `pnl_abs` | Absolute P&L in ₹ |
| `pnl_pct` | % return since buy date |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| [Streamlit](https://streamlit.io) | Dashboard UI |
| [yfinance](https://github.com/ranaroussi/yfinance) | NSE price & fundamentals data |
| [SciPy](https://scipy.org) | Portfolio optimization (SLSQP) |
| [Plotly](https://plotly.com) | Interactive charts |
| [GitHub Actions](https://github.com/features/actions) | Automated EOD tracking |

All free. No API keys. No subscriptions.

---

## ⚠️ Disclaimer

For **research and educational purposes only**.
Not SEBI-registered investment advice.
Past returns do not guarantee future performance.
Always consult a qualified financial advisor before investing.

---

## 📄 Licence

MIT — use freely, attribution appreciated.
