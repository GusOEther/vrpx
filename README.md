# VRPX — Volatility Risk Premium Analyzer (open data)

A self-contained research dashboard for the S&P 500 **volatility risk premium** (VRP = implied vol − subsequently realized vol), built entirely from **free public data** — no API keys, no paid feeds.

One Python script downloads ~15 years of SPX + VIX-family history, computes everything, and writes a single interactive `vrpx_report.html` (no server needed; tabs, lookback switching, tooltips and a help page all run client-side).

## What it computes — all measured, nothing decorative

- **Overview** — per-DTE scorecards (7/14/21/30): interpolated IV, nowcast VRP, percentile, hit rate, worst forward VRP, and a documented composite over five axes (richness · carry/day · safety · path · stability) that ranks premium *quality*, not expected P&L
- **Trade context** — each option structure assigned to Favored / Acceptable / Caution **from its own measured per-regime backtest stats** with disclosed, hand-set cut-offs, plus a measured realized-vol-acceleration caution in the carry regimes
- **Expected move** — IV-implied ±1σ range and ≈16Δ strikes per DTE, next to the measured share of past entries that finished inside (80–85 % vs the nominal 68 %) and the size of the misses
- **VRP matrix** — monthly VRP by DTE, 24-month heatmap
- **Charts** — IV vs realized vol, VRP by DTE, VIX term-structure history
- **Regime analysis / timeline** — rule-based regimes (rules disclosed), per-regime forward-VRP evidence (full history + selected window with a minimum-sample gate), run length and change log
- **Regime forecast** — 5-day streak-adjusted persistence model, a carry/neutral/stress family readout, a retroactive causal calibration curve, and a live forward test logged by the daily build
- **Validation** — does a richer signal pay more forward? Quintile test with an honest monotonicity verdict, overlap-aware sample size, equity proxy, and a k-NN analog engine (context, not alpha)
- **Backtest** — five Black-Scholes-reconstructed structures over ~15 years (short straddle, 16Δ strangle, 16/5Δ iron condor, long straddle, 9/30 calendar) with skew-approximation and regime-gate toggles; totals and drawdowns as the median over all entry phases
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
| VIX / VIX9D / VIX3M / VIX6M | Cboe daily-price CSVs (Yahoo only as emergency fallback) | `^VIX` `^VIX9D` `^VIX3M` `^VIX6M` |
| Risk-free rate | 13-week T-bill via Yahoo | `^IRX` |

The sources publish a day's close at different hours (Cboe posts the VIX close only the next morning). The report is therefore held at the last session for which **every** close is published — nothing is forward-filled at the trailing edge — and per-source freshness is measured separately; the dashboard visibly degrades (`DATA n/5`) if a source goes stale.

## Methodology in one paragraph

IV per tenor comes from total-variance interpolation of the VIX term structure (7 DTE is flat below the 9-day node and flagged as indicative). Realized vol is forward and horizon-matched for all statistics; the "current VRP" is a clearly labelled nowcast (IV − trailing RV). Backtest prices are Black-Scholes reconstructions with the VIX as ATM-vol input (it sits above true ATM vol under skew, so short-vol P&L is biased upward), parametric skew toggle, flat 2% cost, P&L in % of spot notional — not margin returns. Composite weights, sub-score definitions and regime rules are fully disclosed in the Settings tab.

## Honest limitations

History, not prediction. No per-strike option quotes (none exist for free historically) — the skew is a parametric approximation. No bid-ask microstructure. Returns are on spot notional, not margin. **This is research evidence, not investment advice.**

## Deployment

A GitHub Action (`.github/workflows/build.yml` on `main`) rebuilds the report Monday–Saturday at 11:00 UTC for the previous US session and publishes it to GitHub Pages: the stable version at `/vrpx/`, this branch as a preview at `/vrpx/preview/`. If a data source fails, the previous page stays live and flags itself as stale.

## Credits

The dashboard concept and visual language pay homage to **Tomas Byron's "VRPX PRO"** — [homepage (ATM Options Edge)](https://atmoptionsedge.ch/) · [YouTube channel](https://www.youtube.com/@tombyr4907) · [the original walkthrough video](https://www.youtube.com/watch?v=cHPew0fKbgQ). This is an independent rebuild: free public data only, and all scoring/regime/backtest logic is our own and fully disclosed in the Settings tab.

Some feature *ideas* were prompted by that public walkthrough (an analog/k-NN engine, a regime forecast with self-calibration, a path-risk score). The implementation and **every parameter** here were derived and validated independently against our own measurements — nothing was copied. Where a measurement did not support an idea (e.g. a nine-state behavioural taxonomy, a weighted forecast ensemble, a per-strategy weighting lens), it was left out on purpose.

## License

MIT — see [LICENSE](LICENSE).
