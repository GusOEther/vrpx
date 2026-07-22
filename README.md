# VRPX — Volatility Risk Premium Analyzer (open data)

A self-contained research dashboard for the S&P 500 **volatility risk premium** (VRP = implied vol − subsequently realized vol), built entirely from **free public data** — no API keys, no paid feeds.

One Python script downloads ~15 years of SPX + VIX-family history, computes everything, and writes a single interactive `vrpx_report.html` (no server needed; tabs, lookback switching, tooltips and a help page all run client-side).

## What it computes — all measured, nothing decorative

- **Overview** — per-DTE scorecards (7/14/21/30): interpolated IV, nowcast VRP, percentile, hit rate, worst episode, and a documented 5-axis composite (richness · carry/day · reliability · tail-safety · stability)
- **Trade context** — option-strategy families assigned to Favored / Acceptable / Caution **from measured per-regime backtest stats** with explicit thresholds (not opinion)
- **VRP matrix** — monthly VRP by DTE, 24-month heatmap
- **Charts** — IV vs realized vol, VRP by DTE, VIX term-structure history
- **Regime analysis / timeline / forecast** — rule-based regimes (rules disclosed), per-regime forward-VRP evidence, empirical Markov transition matrix (descriptive base-rate, not a prediction)
- **Validation** — does a richer signal pay more forward? Quintile test + equity proxy
- **Backtest** — five Black-Scholes-reconstructed structures over ~15 years (short straddle, 16Δ strangle, 16/5Δ iron condor, long straddle, 9/30 calendar) with skew-approximation and regime-gate toggles, per-regime and per-quintile breakdowns
- **Help** — how to read every number and derive trade inputs, with live-injected figures

## Quick start

```bash
pip install -r requirements.txt
python vrpx.py
# open vrpx_report.html
```

## Data

| Series | Source | Ticker |
|---|---|---|
| S&P 500 | Yahoo Finance | `^GSPC` |
| VIX / VIX9D / VIX3M / VIX6M | Cboe via Yahoo | `^VIX` `^VIX9D` `^VIX3M` `^VIX6M` |
| Risk-free rate | 13-week T-bill via Yahoo | `^IRX` |

Per-source freshness is measured before any forward-fill; the dashboard visibly degrades (`DATA n/5`) if a source goes stale.

## Methodology in one paragraph

IV per tenor comes from total-variance interpolation of the VIX term structure (7 DTE is flat below the 9-day node and flagged as indicative). Realized vol is forward and horizon-matched for all statistics; the "current VRP" is a clearly labelled nowcast (IV − trailing RV). Backtest prices are Black-Scholes reconstructions (parametric skew toggle, flat 2% cost, P&L in % of spot notional — not margin returns). Composite weights, sub-score definitions and regime rules are fully disclosed in the Settings tab.

## Honest limitations

History, not prediction. No per-strike option quotes (none exist for free historically) — the skew is a parametric approximation. No bid-ask microstructure. Returns are on spot notional, not margin. **This is research evidence, not investment advice.**

## Deployment

A GitHub Action (`.github/workflows/build.yml`) rebuilds the report every weekday after US close and publishes it to GitHub Pages. If a data source fails, the previous day's page stays live.

## License

MIT — see [LICENSE](LICENSE).
