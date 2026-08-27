#!/usr/bin/env python3
"""
VRPX  -  Volatility Risk Premium Term-Structure Analyzer  (open-data rebuild)

A working rebuild of the *functionality* behind the VRPX PRO dashboard - every
tab does a real computation on real, free data (yfinance: SPX + VIX family).
Python computes everything up front and embeds it as JSON; client-side JS makes
the whole UI interactive (tabs + lookback), no server, works offline.

Tabs / functions:
  OVERVIEW        - DTE composite cards (IV, VRP, percentile, hit, worst, sub-scores)
  VRP MATRIX      - monthly VRP by DTE, last 24 months (heatmap)
  CHARTS          - IV vs RV, VRP by DTE, VIX term-structure history (line charts)
  REGIME ANALYSIS - regime distribution + per-regime forward-VRP / hit stats
  VALIDATION      - does the signal pay? quintile buckets + overlapping equity curve
  REGIME TIMELINE - historical regime ribbon over time
  REGIME FORECAST - empirical Markov transition matrix + h-step forecast
  SETTINGS        - data-source status, parameters, weights, rules
  EXPORT          - print / save as PDF

Reproducible core:
  IV per DTE = total-variance interpolation of the VIX term structure (VIX9D->VIX).
  RV = realized vol from SPX log returns (forward & horizon-matched for stats,
       trailing for the nowcast).
  VRP = IV - RV, in vol points.
Descriptive research, not a trade signal. Not investment advice.
"""
from __future__ import annotations
import sys, os, math, json
import numpy as np
import pandas as pd
import yfinance as yf

_erf = np.vectorize(math.erf)
def _ncdf(x):                       # standard normal CDF, no scipy needed
    return 0.5 * (1 + _erf(np.asarray(x, float) / math.sqrt(2)))

START      = "2011-01-01"
DTES       = [7, 14, 21, 30]
TRADING_YR = 252
LOOKBACKS  = {63: "3M", 126: "6M", 252: "1Y", 504: "2Y", 1260: "5Y"}
DEFAULT_LB = 504
CHART_TAIL = 1260
FORECAST_H = 20                       # trading days ahead for the Markov forecast
BT_DTE   = 30                         # backtest: tenor (calendar days)
BT_SLIP  = 0.02                       # modeled cost = 2% of premium (bid-ask/commission)
BT_Q     = 0.018                      # assumed SPX dividend yield (constant)
# strike placement via z = Phi^-1(delta), off the ATM vol
Z16C, Z16P = -0.9945,  0.9945         # 16-delta call / put
Z05C, Z05P = -1.6449,  1.6449         #  5-delta wings (iron-condor longs)
# parametric equity skew: IV(K) = IV_atm * (1 + slope * (-m)), m = ln(K/F)/(IV_atm*sqrtT)
# put wing (m<0) steeper than call wing (m>0) — typical SPX smirk. Calibrated to a
# ~25d risk-reversal of ~4-5 vol pts on ~16 ATM. Constants are editable.
SKEW_DN, SKEW_UP = 0.28, 0.10
CALM_REGIMES = {"CALM CARRY", "STEADY CARRY"}
# Composite weights over 5 INDEPENDENT axes (our definition, editable). Sum = 1.0
#   rich   = richness   : percentile of current VRP within the DTE's own history
#   carry  = carry/day  : VRP per calendar day, ranked across the DTE set (relative)
#   safety = reliability+tail-safety, MERGED: the old RELIAB (info ratio) and TAIL
#            (1-|CVaR5|) axes measure the same thing — Spearman +0.885 on the real
#            definitions (measured 2026-08-26) — so they are one axis, not two, to
#            avoid double-counting. safety = mean of both 0-100 sub-scores.
#   path   = path-safety: cross-sectional rank of trailing mean Max-Adverse-Excursion
#            of the SPX path over the forward window (mild path now = high score). A
#            genuine PATH axis (the price journey), distinct from tail (VRP outcome).
#   stab   = stability  : steadiness of recent VRP vs its window dispersion
W = {"rich": 0.30, "carry": 0.15, "safety": 0.35, "path": 0.10, "stab": 0.10}
CARD_COLORS = ["--pink", "--yellow", "--blue", "--green"]

REG_ORDER = ["CALM CARRY", "STEADY CARRY", "NEUTRAL RANGE",
             "TRANSITION", "STRESS / BACKW", "VOL SHOCK"]
REG_COLOR = {"CALM CARRY": "--yellow", "STEADY CARRY": "--green",
             "NEUTRAL RANGE": "--blue", "TRANSITION": "--amber",
             "STRESS / BACKW": "--red", "VOL SHOCK": "--pink"}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def jnum(x, d=2):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return None
    return round(float(x), d)

def arr(s, d=2):
    return [jnum(x, d) for x in s]

def pct_rank(win, value):
    w = win[~np.isnan(win)]
    if len(w) == 0 or value is None or np.isnan(value):
        return float("nan")
    return float((w < value).mean() * 100)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load():
    tk = ["^GSPC", "^VIX", "^VIX9D", "^VIX3M", "^VIX6M"]
    px = yf.download(tk, start=START, progress=False, auto_adjust=True)["Close"]
    px = px.rename(columns={"^GSPC": "SPX"})
    # real per-source freshness, measured BEFORE ffill (ffill would mask staleness)
    src = []
    for name, col in [("SPX", "SPX"), ("VIX", "^VIX"), ("VIX9D", "^VIX9D"),
                      ("VIX3M", "^VIX3M"), ("VIX6M", "^VIX6M")]:
        s = px[col].dropna()
        lag = int((px.index >= s.index[-1]).sum() - 1) if len(s) else 9999
        src.append(dict(name=name, ticker=col if col != "SPX" else "^GSPC",
                        last=s.index[-1].strftime("%d %b %Y") if len(s) else "—",
                        lag=lag, ok=bool(lag <= 3)))
    # close interior gaps (Cboe drops the odd day) via ffill; COUNT how many rows were
    # carried so the freshness layer can flag the series as degraded rather than pretend.
    fill_counts = {}
    for c in ["^VIX", "^VIX9D", "^VIX3M", "^VIX6M"]:
        s = px[c]
        interior = s.loc[s.first_valid_index():].isna().sum() if s.first_valid_index() is not None else 0
        fill_counts[c] = int(interior)
        px[c] = px[c].ffill(limit=3)
    for s in src:
        s["filled"] = fill_counts.get(s["ticker"], 0)
    px = px.dropna(subset=["SPX", "^VIX", "^VIX9D"])
    px.attrs["sources"] = src
    return px

def trading_days(dte):
    return max(1, round(dte / 365 * TRADING_YR))

def iv_series(px, dte):
    v9, v30 = px["^VIX9D"], px["^VIX"]
    if dte <= 9:
        return v9.copy()
    if dte == 30:
        return v30.copy()
    T1, T2 = 9, 30
    tv1, tv2 = (v9 ** 2) * T1, (v30 ** 2) * T2
    w = (dte - T1) / (T2 - T1)
    return np.sqrt((tv1 + w * (tv2 - tv1)) / dte)

def rv_forward(px, dte):
    n = trading_days(dte)
    r = np.log(px["SPX"] / px["SPX"].shift(1))
    return r.rolling(n).std().shift(-n) * math.sqrt(TRADING_YR) * 100

def rv_trailing(px, dte):
    n = trading_days(dte)
    r = np.log(px["SPX"] / px["SPX"].shift(1))
    return r.rolling(n).std() * math.sqrt(TRADING_YR) * 100


# --------------------------------------------------------------------------- #
# per-DTE series & cards
# --------------------------------------------------------------------------- #
def mae_series(px, dte):
    """Max Adverse Excursion (%) of the SPX path over the forward window, per entry day.
    Resolved (shift by horizon) so the value is causally available. For a short-vol
    position the adverse move is large in either direction."""
    n = trading_days(dte)
    p = px["SPX"]
    hi = p.rolling(n).max().shift(-n)
    lo = p.rolling(n).min().shift(-n)
    mae = np.maximum(hi / p - 1.0, 1.0 - lo / p) * 100.0
    return mae.shift(n)                      # causal: known only n days later

def dte_series(px, dte):
    iv  = iv_series(px, dte)
    rvf = rv_forward(px, dte)
    rvt = rv_trailing(px, dte)
    vrp_hist = iv - rvf                      # forward outcome (for stats)
    vrp_now  = iv - rvt                      # nowcast (no forward yet)
    mae = mae_series(px, dte)                # causal path risk
    rv_acc = (rvt.iloc[-1] / rvt.iloc[-22]) if not np.isnan(rvt.iloc[-22]) and rvt.iloc[-22] else 1.0
    gamma  = float(np.clip(100 * (1 - (rv_acc - 1)), 0, 100))
    return dict(dte=dte, iv=iv, rvt=rvt, cur_iv=float(iv.iloc[-1]),
                cur_vrp=float(vrp_now.iloc[-1]), vrp_hist=vrp_hist,
                vrp_now=vrp_now, mae=mae, gamma=gamma)

def cards_for(series, N):
    """Compute all DTE cards for a lookback together (needed for cross-sectional carry)."""
    raw = {}
    for d in DTES:
        ds  = series[d]
        win = ds["vrp_hist"].dropna().tail(N).to_numpy()
        cur = ds["cur_vrp"]
        mean, sd = (np.nanmean(win), np.nanstd(win)) if len(win) else (np.nan, np.nan)
        ir   = mean / sd if sd else np.nan
        p5   = np.nanpercentile(win, 5) if len(win) else np.nan
        tailv = win[win <= p5]
        cvar = float(np.nanmean(tailv)) if len(tailv) else np.nan
        recent = ds["vrp_now"].dropna().tail(21).std()
        stab = 100 * (1 - min(recent / sd, 1)) if sd and sd > 0 else 50.0
        # PATH: self-relative percentile of the recent path risk vs this DTE's own
        # window history (cross-sectional MAE is trivially monotone in horizon, so it
        # would only measure window length — the percentile normalises that away).
        mwin = ds["mae"].dropna().tail(N).to_numpy()
        recent_mae = float(np.nanmean(mwin[-21:])) if len(mwin) else np.nan
        path = (100 - pct_rank(mwin, recent_mae)) if len(mwin) and not np.isnan(recent_mae) else 50.0
        raw[d] = dict(cur_iv=ds["cur_iv"], cur_vrp=cur, rich=pct_rank(win, cur), ir=ir,
                      cvar=cvar, stab=stab, carry_pd=cur / d, path=path,
                      hit=float((win > 0).mean() * 100) if len(win) else np.nan,
                      worst=float(np.nanmin(win)) if len(win) else np.nan)
    cps = [raw[d]["carry_pd"] for d in DTES]
    cmin, cmax = min(cps), max(cps)
    rng = cmax - cmin
    cards = []
    for d in DTES:
        r = raw[d]
        carry  = 100 * (r["carry_pd"] - cmin) / rng if rng > 1e-9 else 50.0
        reliab = float(np.clip((0 if np.isnan(r["ir"]) else r["ir"]) / 0.8 * 100, 0, 100))
        tail   = float(np.clip(100 * (1 - abs(r["cvar"]) / 50), 0, 100)) if not np.isnan(r["cvar"]) else 50.0
        safety = (reliab + tail) / 2.0                       # merged axis (rho +0.885)
        path   = r["path"]
        subs = {"rich": r["rich"], "carry": carry, "safety": safety, "path": path, "stab": r["stab"]}
        comp = sum(W[k] * (0 if (subs[k] is None or np.isnan(subs[k])) else subs[k]) for k in W)
        cards.append(dict(dte=d, cur_iv=jnum(r["cur_iv"]), cur_vrp=jnum(r["cur_vrp"]),
                          vrp_pct=jnum(r["rich"]), hit=jnum(r["hit"]), worst=jnum(r["worst"]),
                          composite=jnum(comp), subs={k: jnum(v) for k, v in subs.items()}))
    return cards


# --------------------------------------------------------------------------- #
# regime (vectorised over full history)
# --------------------------------------------------------------------------- #
def regime_series(px):
    vix   = px["^VIX"]
    v3m   = px["^VIX3M"].fillna(px["^VIX"])
    slope = v3m / vix
    up    = px["SPX"] > px["SPX"].rolling(200).mean()
    cond = [vix >= 40,
            (vix >= 28) | (slope < 0.97),
            vix >= 20,
            (slope >= 1.05) & up & (vix < 17),
            (slope >= 1.0) & up]
    lab = ["VOL SHOCK", "STRESS / BACKW", "TRANSITION", "CALM CARRY", "STEADY CARRY"]
    labels = pd.Series(np.select(cond, lab, default="NEUTRAL RANGE"), index=px.index)
    return labels, slope, up

def transition_matrix(labels):
    idx = {l: i for i, l in enumerate(REG_ORDER)}
    n = len(REG_ORDER)
    M = np.zeros((n, n))
    a = labels.to_numpy()
    for i in range(len(a) - 1):
        M[idx[a[i]], idx[a[i + 1]]] += 1
    rs = M.sum(1, keepdims=True)
    P = np.divide(M, rs, out=np.zeros_like(M), where=rs > 0)
    return M, P

def regime_block(px, labels, slope, up, vrp_hist30, vrp_now30):
    cur = labels.iloc[-1]
    vix = float(px["^VIX"].iloc[-1])
    v3m = float(px["^VIX3M"].fillna(px["^VIX"]).iloc[-1])
    sl  = float(slope.iloc[-1])
    ret20 = float(px["SPX"].iloc[-1] / px["SPX"].iloc[-21] - 1) * 100

    # per-regime forward-VRP stats (full history)
    per = {}
    vh = vrp_hist30
    total = len(labels)
    for lab in REG_ORDER:
        mask = (labels == lab)
        days = int(mask.sum())
        vals = vh[mask.reindex(vh.index, fill_value=False)].dropna()
        per[lab] = dict(days=days, share=jnum(days / total * 100, 1),
                        avg_vrp=jnum(vals.mean(), 2) if len(vals) else None,
                        hit=jnum((vals > 0).mean() * 100, 0) if len(vals) else None)

    # per-regime, per-lookback (window) stats with a hard min-n gate. Below MIN_REG_N
    # eligible observations a cell reads "insufficient" rather than a misleading number.
    MIN_REG_N = 60
    per_lb = {}
    for N in LOOKBACKS:
        if N > len(labels):
            continue
        wl = labels.iloc[-N:]
        cell = {}
        for lab in REG_ORDER:
            m = (wl == lab)
            vals = vh[m.reindex(vh.index, fill_value=False)].dropna()
            days = int(m.sum())
            if len(vals) >= MIN_REG_N:
                cell[lab] = dict(days=days, n=int(len(vals)),
                                 avg_vrp=jnum(vals.mean(), 2),
                                 hit=jnum((vals > 0).mean() * 100, 0), ok=True)
            else:
                cell[lab] = dict(days=days, n=int(len(vals)), avg_vrp=None, hit=None, ok=False)
        per_lb[str(N)] = cell

    # transition matrix + forecast
    M, P = transition_matrix(labels)
    Ph = np.linalg.matrix_power(P, FORECAST_H)
    ci = REG_ORDER.index(cur)
    fc_1  = P[ci]
    fc_h  = Ph[ci]

    # diagnostics: where do we sit vs history
    vix_pct = float((px["^VIX"].to_numpy() < vix).mean() * 100)
    sl_pct  = float((slope.dropna().to_numpy() < sl).mean() * 100)
    front   = float(px["^VIX9D"].iloc[-1] / px["^VIX"].iloc[-1])   # <1 = steep front contango
    rvt30   = rv_trailing(px, 30)
    rv_acc  = float(rvt30.iloc[-1] / rvt30.iloc[-22]) if rvt30.iloc[-22] and not np.isnan(rvt30.iloc[-22]) else 1.0
    # rv_composite = rv10 - rv63 (vol points, trailing): >0 = realized vol accelerating vs
    # its quarter norm. Measured (Ticket 02, 2026-08-26): in CALM/STEADY a positive value
    # precedes weaker short-vol P&L (-0.7pp straddle). TRANSITION flips sign -> no rule there.
    r_ = np.log(px["SPX"] / px["SPX"].shift(1))
    rvc = float(((r_.rolling(10).std() - r_.rolling(63).std()) * math.sqrt(TRADING_YR) * 100).iloc[-1])

    return dict(
        current=dict(label=cur, color=REG_COLOR[cur], slope=jnum(sl, 3),
                     slope_txt="contango" if sl >= 1 else "backwardation",
                     vix=jnum(vix, 1), v3m=jnum(v3m, 1),
                     spx_state="Uptrend" if bool(up.iloc[-1]) else "Range/Down",
                     ret20=jnum(ret20, 1), vix_pct=jnum(vix_pct, 0), slope_pct=jnum(sl_pct, 0),
                     front=jnum(front, 3), reg_hit=per[cur]["hit"], reg_avg=per[cur]["avg_vrp"],
                     reg_days=per[cur]["days"], rv_acc=jnum(rv_acc, 2), rv_calm=bool(rv_acc <= 1.05),
                     rvc=jnum(rvc, 2), rvc_accel=bool(rvc > 0)),
        order=REG_ORDER, colors=[REG_COLOR[l] for l in REG_ORDER],
        per_regime=per, per_regime_lb=per_lb, min_reg_n=MIN_REG_N,
        transition=[[jnum(P[i, j] * 100, 0) for j in range(len(REG_ORDER))] for i in range(len(REG_ORDER))],
        counts=[int((labels == l).sum()) for l in REG_ORDER],
        forecast_1=[jnum(x * 100, 0) for x in fc_1],
        forecast_h=[jnum(x * 100, 0) for x in fc_h],
        horizon=FORECAST_H,
    )


# --------------------------------------------------------------------------- #
# regime forecast (C2: streak-adjusted persistence) + retroactive calibration
#   Chosen over a 4-component ensemble by measurement (Ticket 04): streak-adjusted
#   persistence alone beat every combination (LogLoss 0.923, top-1 67.5%). Horizon
#   is 5 trading days — behaviour features have ~zero persistence at 21d (Ticket 04b).
# --------------------------------------------------------------------------- #
FC_H = 5          # forecast horizon, trading days
CAL_START = "2016-01-01"

def _streak_bucket(s):
    return 0 if s <= 3 else (1 if s <= 10 else 2)

def forecast_block(px, labels):
    K = len(REG_ORDER)
    idx = {l: i for i, l in enumerate(REG_ORDER)}
    L = np.array([idx[x] for x in labels.to_numpy()])
    n = len(L)
    streak = np.ones(n, int)
    for i in range(1, n):
        streak[i] = streak[i - 1] + 1 if L[i] == L[i - 1] else 1

    M = np.ones((K, K))            # Laplace-smoothed transition counts (day+H)
    PS = np.ones((K, 3, 2))        # [regime, streak-bucket, {stay, go}]
    nxt = 0
    dates = labels.index
    cal_start_pos = dates.get_loc(dates[dates >= CAL_START][0]) if (dates >= CAL_START).any() else n

    def dist_at(t):
        """C2 forecast distribution from counters as they stand (expanding)."""
        pstay = PS[L[t], _streak_bucket(streak[t]), 0] / PS[L[t], _streak_bucket(streak[t])].sum()
        off = M[L[t]].copy(); off[L[t]] = 0
        off = off / off.sum() if off.sum() > 0 else np.full(K, 1.0 / K)
        pr = off * (1 - pstay); pr[L[t]] += pstay
        return pr / pr.sum()

    journal, cal = [], []
    for t in range(n):
        while nxt + FC_H <= t:                 # fold in every resolved pair s -> s+H
            s = nxt
            M[L[s], L[s + FC_H]] += 1
            PS[L[s], _streak_bucket(streak[s]), 0 if L[s + FC_H] == L[s] else 1] += 1
            nxt += 1
        if t >= cal_start_pos and t + FC_H < n:
            pr = dist_at(t)
            top = int(np.argmax(pr))
            cal.append((float(pr[top]), int(top == L[t + FC_H])))

    # current forecast (counters now hold every resolved pair up to today)
    t = n - 1
    pr = dist_at(t)
    order = np.argsort(pr)[::-1]
    top, second = int(order[0]), int(order[1])
    fc = [jnum(x * 100, 1) for x in pr]

    # retroactive calibration curve
    cdf = pd.DataFrame(cal, columns=["conf", "hit"])
    edges = [0.30, 0.50, 0.70, 0.90, 1.01]
    labels_b = ["30-50%", "50-70%", "70-90%", "90-100%"]
    buckets = []
    for lo, hi, nm in zip([0.30, 0.50, 0.70, 0.90], edges[1:], labels_b):
        m = (cdf["conf"] >= lo) & (cdf["conf"] < hi)
        sub = cdf[m]
        buckets.append(dict(bucket=nm, n=int(len(sub)),
                            conf=jnum(sub["conf"].mean() * 100, 0) if len(sub) else None,
                            acc=jnum(sub["hit"].mean() * 100, 0) if len(sub) else None,
                            err=jnum((sub["hit"].mean() - sub["conf"].mean()) * 100, 0) if len(sub) else None))
    ov_conf = float(cdf["conf"].mean()) if len(cdf) else 0.0
    ov_acc = float(cdf["hit"].mean()) if len(cdf) else 0.0
    bias_pp = (ov_acc - ov_conf) * 100
    quality = int(round(100 - min(abs(bias_pp) + cdf.assign(e=(cdf["hit"] - cdf["conf"]).abs())["e"].mean() * 100 if len(cdf) else 100, 100)))

    # journal: last ~15 days, each forecast made with counters strictly up to that day
    # (last FC_H are PENDING — no resolved actual yet). Fresh causal replay.
    journal = []
    Mj = np.ones((K, K)); PSj = np.ones((K, 3, 2)); nj = 0
    def dist_j(t):
        ps = PSj[L[t], _streak_bucket(streak[t]), 0] / PSj[L[t], _streak_bucket(streak[t])].sum()
        off = Mj[L[t]].copy(); off[L[t]] = 0
        off = off / off.sum() if off.sum() > 0 else np.full(K, 1.0 / K)
        pr = off * (1 - ps); pr[L[t]] += ps
        return pr / pr.sum()
    for t in range(n):
        while nj + FC_H <= t:
            s = nj; Mj[L[s], L[s + FC_H]] += 1
            PSj[L[s], _streak_bucket(streak[s]), 0 if L[s + FC_H] == L[s] else 1] += 1; nj += 1
        if t >= n - 15:
            prj = dist_j(t); tp = int(np.argmax(prj))
            actual = REG_ORDER[L[t + FC_H]] if t + FC_H < n else None
            status = "PENDING" if actual is None else ("HIT" if tp == L[t + FC_H] else "MISS")
            journal.append(dict(date=dates[t].strftime("%Y-%m-%d"), regime=REG_ORDER[L[t]],
                                pred=REG_ORDER[tp], prob=jnum(float(prj[tp]) * 100, 0),
                                actual=actual, status=status))
    journal.reverse()

    return dict(horizon=FC_H, order=REG_ORDER, colors=[REG_COLOR[l] for l in REG_ORDER],
                dist=fc, top=REG_ORDER[top], top_prob=jnum(float(pr[top]) * 100, 1),
                second=REG_ORDER[second], second_prob=jnum(float(pr[second]) * 100, 1),
                calib=dict(buckets=buckets, bias_pp=jnum(bias_pp, 0), quality=quality,
                           overall_conf=jnum(ov_conf * 100, 0), overall_acc=jnum(ov_acc * 100, 0),
                           bias="Overconfident" if bias_pp < 0 else "Underconfident", n=int(len(cdf))),
                journal=journal, cal_from=CAL_START[:4])


# --------------------------------------------------------------------------- #
# forward test — a genuine, non-circular test: log today's forecast, resolve it
#   FC_H trading days later. Grows forward only; persisted in forecast_forward.csv
#   which the CI commits back to the repo each run. Distinct from the retroactive
#   calibration (which re-derives history) — this one cannot be curve-fit.
# --------------------------------------------------------------------------- #
FORWARD_CSV = "forecast_forward.csv"

def forward_test(px, labels, fc):
    import csv
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), FORWARD_CSV)
    rows = []
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    idx = labels.index
    lab = labels.to_numpy()
    date_pos = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(idx)}
    today = idx[-1].strftime("%Y-%m-%d")

    # resolve any pending row whose horizon has now elapsed
    for row in rows:
        if row.get("status", "PENDING") == "PENDING":
            pos = date_pos.get(row["date"])
            if pos is not None and pos + FC_H < len(idx):
                actual = lab[pos + FC_H]
                row["actual"] = actual
                row["resolved_on"] = idx[pos + FC_H].strftime("%Y-%m-%d")
                row["status"] = "HIT" if actual == row["pred"] else "MISS"

    # append today's forecast once (idempotent per as-of date)
    if today not in {r["date"] for r in rows}:
        rows.append(dict(date=today, regime=labels.iloc[-1],
                         pred=fc["top"], prob=fc["top_prob"],
                         actual="", resolved_on="", status="PENDING"))

    # write back (CI commits this)
    cols = ["date", "regime", "pred", "prob", "actual", "resolved_on", "status"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})

    resolved = [r for r in rows if r["status"] in ("HIT", "MISS")]
    hits = sum(1 for r in resolved if r["status"] == "HIT")
    pending = [r for r in rows if r["status"] == "PENDING"]
    recent = list(reversed(rows[-15:]))
    return dict(since=rows[0]["date"] if rows else today, n_logged=len(rows),
                n_resolved=len(resolved), n_pending=len(pending),
                hits=hits, hit_rate=jnum(hits / len(resolved) * 100, 0) if resolved else None,
                horizon=FC_H, recent=recent)


# --------------------------------------------------------------------------- #
# validation: does high current VRP predict higher captured premium?
# --------------------------------------------------------------------------- #
def validation_block(vrp_now30, vrp_hist30):
    df = pd.DataFrame({"sig": vrp_now30, "out": vrp_hist30}).dropna()
    q = pd.qcut(df["sig"], 5, labels=False, duplicates="drop")
    buckets = []
    for b in sorted(pd.unique(q.dropna())):
        m = q == b
        buckets.append(dict(bucket=int(b) + 1, n=int(m.sum()),
                            sig=jnum(df["sig"][m].mean(), 2),
                            out=jnum(df["out"][m].mean(), 2),
                            hit=jnum((df["out"][m] > 0).mean() * 100, 0)))
    out = df["out"]
    mean, sd = out.mean(), out.std()
    return dict(buckets=buckets,
                mean=jnum(mean, 2), std=jnum(sd, 2),
                info_ratio=jnum(mean / sd, 2) if sd else None,
                hit=jnum((out > 0).mean() * 100, 0),
                worst=jnum(out.min(), 1), best=jnum(out.max(), 1),
                n=int(len(out)))


# --------------------------------------------------------------------------- #
# analog engine (k-NN over the vol/term/price state) — Ticket 03
#   5-dim vector, causal percentile-rank normalisation, k=120, full history.
#   Measured fwd-RV21 Spearman ~0.60 vs 0.55 baseline; return prediction ~0 -> context,
#   not alpha. Neighbours restricted to days with a complete forward window.
# --------------------------------------------------------------------------- #
ANALOG_K = 120
ANALOG_HORIZONS = [7, 14, 21, 30]

def analog_block(px, labels):
    p = px["SPX"]
    r = np.log(p / p.shift(1))
    ann = math.sqrt(TRADING_YR) * 100
    feats = pd.DataFrame({
        "vix":  px["^VIX"],
        "v9v":  px["^VIX9D"] / px["^VIX"],
        "vv3m": px["^VIX"] / px["^VIX3M"].fillna(px["^VIX"]),
        "rvc":  (r.rolling(10).std() - r.rolling(63).std()) * ann,
        "er21": (p - p.shift(21)).abs() / p.diff().abs().rolling(21).sum(),
    }).dropna()
    # causal percentile-rank normalisation (expanding). For today's query the final
    # expanding state == full-history rank, which is correct for the last row.
    Z = feats.rank(pct=True).to_numpy()
    idx = feats.index
    n = len(idx)
    maxh = max(ANALOG_HORIZONS)
    q = Z[-1]
    # eligible neighbours: complete forward window, exclude the last maxh rows and today
    elig = np.arange(0, n - maxh)
    d = np.linalg.norm(Z[elig] - q, axis=1)
    order = elig[np.argsort(d)][:ANALOG_K]
    dd = np.linalg.norm(Z[order] - q, axis=1)
    sim = 1.0 / (1.0 + dd)                       # display transform only
    sim = (sim - sim.min()) / (sim.max() - sim.min()) if sim.max() > sim.min() else sim

    P = p.reindex(idx).to_numpy()
    VIX = px["^VIX"].reindex(idx).to_numpy()
    lab = labels.reindex(idx).to_numpy()
    rv21_fwd = (r.rolling(21).std().shift(-21) * ann).reindex(idx).to_numpy()

    per_h = []
    for h in ANALOG_HORIZONS:
        rets = np.array([P[s + h] / P[s] - 1 for s in order]) * 100
        per_h.append(dict(h=h, matches=len(order),
                          avg=jnum(np.nanmean(rets), 2), med=jnum(np.nanmedian(rets), 2),
                          worst=jnum(np.nanmin(rets), 1), best=jnum(np.nanmax(rets), 1),
                          pos=jnum((rets > 0).mean() * 100, 0)))
    fwd_rv = np.array([rv21_fwd[s] for s in order])
    vix_chg = np.array([VIX[s + 21] - VIX[s] for s in order])

    top = []
    for j in np.argsort(dd)[:12]:
        s = order[j]
        top.append(dict(date=idx[s].strftime("%Y-%m-%d"), regime=lab[s],
                        dist=jnum(float(dd[j]), 3), sim=jnum(float(sim[j]) * 100, 0),
                        vix=jnum(float(VIX[s]), 1),
                        ret21=jnum(float(P[s + 21] / P[s] - 1) * 100, 1),
                        fwd_rv=jnum(float(rv21_fwd[s]), 1)))

    return dict(k=ANALOG_K, n_eligible=int(len(elig)),
                window_from=idx[0].strftime("%d %b %Y"), window_to=idx[-1].strftime("%d %b %Y"),
                per_h=per_h, avg_fwd_rv=jnum(float(np.nanmean(fwd_rv)), 1),
                avg_vix_chg=jnum(float(np.nanmean(vix_chg)), 1), top=top,
                cur=dict(vix=jnum(float(feats["vix"].iloc[-1]), 1),
                         v9v=jnum(float(feats["v9v"].iloc[-1]), 3),
                         vv3m=jnum(float(feats["vv3m"].iloc[-1]), 3),
                         rvc=jnum(float(feats["rvc"].iloc[-1]), 2),
                         er21=jnum(float(feats["er21"].iloc[-1]), 2)))


# --------------------------------------------------------------------------- #
# VRP matrix: monthly nowcast VRP by DTE (last 24 months)
# --------------------------------------------------------------------------- #
def vrp_matrix_block(series):
    months = None
    rows = {}
    for d in DTES:
        m = series[d]["vrp_now"].dropna().resample("ME").mean().tail(24)
        rows[str(d)] = arr(m, 2)
        months = [ix.strftime("%b %y") for ix in m.index]
    flat = [v for r in rows.values() for v in r if v is not None]
    return dict(months=months, rows=rows, dtes=DTES,
                vmin=jnum(min(flat), 1), vmax=jnum(max(flat), 1))


# --------------------------------------------------------------------------- #
# historical backtest: short ATM straddle, BS-reconstructed, hold to expiry
# --------------------------------------------------------------------------- #
def load_rate(index):
    try:
        r = yf.download("^IRX", start=START, progress=False, auto_adjust=True)["Close"]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[:, 0]
        r = r.reindex(index).ffill().bfill() / 100.0
        return r.fillna(0.03)
    except Exception:
        return pd.Series(0.03, index=index)

def bs_call_put(S, K, T, r, q, sig):
    """Black-Scholes call & put price (index points), vectorised."""
    sig = np.maximum(sig, 1e-6)
    d1 = (np.log(S / K) + (r - q + 0.5 * sig ** 2) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    call = S * np.exp(-q * T) * _ncdf(d1) - K * np.exp(-r * T) * _ncdf(d2)
    put  = K * np.exp(-r * T) * _ncdf(-d2) - S * np.exp(-q * T) * _ncdf(-d1)
    return call, put

def _stats(pnl):
    pnl = pnl[~np.isnan(pnl)]
    if not len(pnl):
        return {}
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    return dict(n=int(len(pnl)), mean=jnum(pnl.mean(), 2), std=jnum(pnl.std(), 2),
                hit=jnum((pnl > 0).mean() * 100, 0), worst=jnum(pnl.min(), 1),
                best=jnum(pnl.max(), 1), p5=jnum(np.percentile(pnl, 5), 1),
                avg_win=jnum(wins.mean(), 2) if len(wins) else None,
                avg_loss=jnum(losses.mean(), 2) if len(losses) else None)

def _equity(pnl_steps, dates_steps, mask=None):
    eq, out, peak, mdd = 0.0, [], 0.0, 0.0
    for i, p in enumerate(pnl_steps):
        if p is not None and not np.isnan(p) and (mask is None or mask[i]):
            eq += p
        peak = max(peak, eq); mdd = min(mdd, eq - peak)
        out.append(round(eq, 2))
    return out, jnum(eq, 1), jnum(mdd, 1)

def backtest_block(px, iv30, labels, vrp_now30, rate):
    n_exp = trading_days(BT_DTE)
    T  = BT_DTE / 365.0
    sq = math.sqrt(T)
    S   = px["SPX"].to_numpy()
    sig = (iv30 / 100.0).to_numpy()
    r   = rate.to_numpy()
    idx = px.index

    s0    = S[:-n_exp]                  # entry spot
    s_exp = S[n_exp:]                   # spot at expiry
    sig0  = sig[:-n_exp]
    r0    = r[:-n_exp]
    lab   = labels.to_numpy()[:-n_exp]
    vrp   = vrp_now30.to_numpy()[:-n_exp]
    dts   = [d.strftime("%Y-%m-%d") for d in idx[:-n_exp]]

    F = s0 * np.exp((r0 - BT_Q) * T)
    def strike(z):  return s0 * np.exp((r0 - BT_Q + 0.5 * sig0 ** 2) * T - z * sig0 * sq)
    def sk_iv(K):                                   # skew-adjusted IV at strike K
        m = np.log(K / F) / (sig0 * sq)
        return sig0 * (1 + np.where(m < 0, SKEW_DN * (-m), SKEW_UP * (-m)))
    def cp(K, skew):  return bs_call_put(s0, K, T, r0, BT_Q, sk_iv(K) if skew else sig0)

    def straddle(skew):                             # ATM -> skew has no effect
        c, p = cp(s0, skew)
        return ((c + p) * (1 - BT_SLIP) - np.abs(s_exp - s0)) / s0 * 100
    def strangle(skew):
        Kc, Kp = strike(Z16C), strike(Z16P)
        cc, _ = cp(Kc, skew); _, pp = cp(Kp, skew)
        pay = np.maximum(s_exp - Kc, 0) + np.maximum(Kp - s_exp, 0)
        return ((cc + pp) * (1 - BT_SLIP) - pay) / s0 * 100
    def condor(skew):
        Ksc, Klc, Ksp, Klp = strike(Z16C), strike(Z05C), strike(Z16P), strike(Z05P)
        csc, _ = cp(Ksc, skew); clc, _ = cp(Klc, skew); _, psp = cp(Ksp, skew); _, plp = cp(Klp, skew)
        ncr  = ((csc - clc) + (psp - plp)) * (1 - BT_SLIP)
        cpay = np.maximum(s_exp - Ksc, 0) - np.maximum(s_exp - Klc, 0)
        ppay = np.maximum(Ksp - s_exp, 0) - np.maximum(Klp - s_exp, 0)
        return (ncr - (cpay + ppay)) / s0 * 100

    yrs    = (idx[-1] - idx[0]).days / 365.25
    steps  = list(range(0, len(s0), n_exp))
    ds     = [dts[i] for i in steps]
    gated  = np.array([l in CALM_REGIMES for l in lab])
    gm     = [bool(gated[i]) for i in steps]

    def one(pnl, mask_steps=None, mask_full=None):
        ps = [pnl[i] for i in steps]
        eq, tot, mdd = _equity(ps, ds, mask_steps)
        st = _stats(pnl if mask_full is None else pnl[mask_full])
        n  = len(steps) if mask_steps is None else int(np.sum(mask_steps))
        return dict(stats=st, eq=eq, tot=tot, mdd=mdd, ann=jnum(tot / yrs, 1), n=n)

    def make(fn):
        out = {}
        for sk_key, skew in (("flat", False), ("skew", True)):
            pnl = fn(skew)
            out[sk_key] = dict(all=one(pnl), gated=one(pnl, gm, gated))
        return out

    structures = {"Straddle (ATM)": make(straddle),
                  "Strangle (16Δ)": make(strangle),
                  "Iron Condor (16/5Δ)": make(condor)}

    # --- Long vol: BUY the ATM straddle (mirror trade; buyer pays the spread) ---
    c0, p0 = cp(s0, False)
    pnl_lv = (np.abs(s_exp - s0) - (c0 + p0) * (1 + BT_SLIP)) / s0 * 100
    lv = dict(all=one(pnl_lv), gated=one(pnl_lv, gm, gated))
    structures["Long Vol (ATM buy)"] = {"flat": lv, "skew": lv}   # ATM -> skew-invariant

    # --- Calendar 9/30 (ATM call): short 9d, long 30d, exit at front expiry.
    #     Back leg marked with the REAL then-current interpolated 21d IV. ---
    n_f = trading_days(9)
    Tf, Tb, Tr = 9 / 365.0, 30 / 365.0, 21 / 365.0
    iv21 = (iv_series(px, 21) / 100.0).to_numpy()
    sig9 = (px["^VIX9D"] / 100.0).to_numpy()
    S0c, STc, r0c = S[:-n_f], S[n_f:], r[:-n_f]
    cf, _ = bs_call_put(S0c, S0c, Tf, r0c, BT_Q, sig9[:-n_f])
    cb, _ = bs_call_put(S0c, S0c, Tb, r0c, BT_Q, sig[:-n_f])
    debit = cb * (1 + BT_SLIP) - cf * (1 - BT_SLIP)
    vb, _ = bs_call_put(STc, S0c, Tr, r0c, BT_Q, iv21[n_f:])
    pnl_cal = (vb * (1 - BT_SLIP) - np.maximum(STc - S0c, 0) - debit) / S0c * 100
    lab_c   = labels.to_numpy()[:-n_f]
    steps_c = list(range(0, len(pnl_cal), n_f))
    ds_c    = [idx[i].strftime("%Y-%m-%d") for i in steps_c]
    gm_c    = [bool(lab_c[i] in CALM_REGIMES) for i in steps_c]
    gated_c = np.array([l in CALM_REGIMES for l in lab_c])

    def cal_entry(mask_steps=None, mask_full=None):
        """Calendar runs on its own 6-session cadence; totals/mdd are cadence-true,
        the display equity is resampled onto the shared 21-session date grid."""
        ps = [pnl_cal[i] for i in steps_c]
        _, tot, mdd = _equity(ps, ds_c, mask_steps)
        eq, run, j = [], 0.0, 0
        for pos in steps:
            while j < len(steps_c) and steps_c[j] <= pos:
                p = pnl_cal[steps_c[j]]
                if not np.isnan(p) and (mask_steps is None or mask_steps[j]):
                    run += p
                j += 1
            eq.append(round(run, 2))
        st = _stats(pnl_cal if mask_full is None else pnl_cal[mask_full])
        n = len(steps_c) if mask_steps is None else int(np.sum(mask_steps))
        return dict(stats=st, eq=eq, tot=tot, mdd=mdd, ann=jnum(tot / yrs, 1), n=n)

    calv = dict(all=cal_entry(), gated=cal_entry(gm_c, gated_c))
    structures["Calendar 9/30 (ATM call)"] = {"flat": calv, "skew": calv}  # ATM -> skew-invariant

    # unified per-strategy breakdowns (flat pricing, all entries)
    def per_regime(pnl, labarr):
        out = []
        for lr in REG_ORDER:
            m = (labarr == lr) & ~np.isnan(pnl)
            if m.sum():
                x = pnl[m]
                out.append(dict(regime=lr, n=int(m.sum()), mean=jnum(x.mean(), 2),
                                hit=jnum((x > 0).mean() * 100, 0),
                                p5=jnum(np.percentile(x, 5), 1), worst=jnum(x.min(), 1)))
        return out

    def per_quintile(pnl, sigarr):
        dfq = pd.DataFrame({"sig": sigarr, "pnl": pnl}).dropna()
        qq = pd.qcut(dfq["sig"], 5, labels=False, duplicates="drop")
        return [dict(q=int(b) + 1, sig=jnum(dfq["sig"][qq == b].mean(), 2),
                     pnl=jnum(dfq["pnl"][qq == b].mean(), 2),
                     hit=jnum((dfq["pnl"][qq == b] > 0).mean() * 100, 0))
                for b in sorted(pd.unique(qq.dropna()))]

    pnl_sd, pnl_stg, pnl_ic = straddle(False), strangle(False), condor(False)
    vrp_c = vrp_now30.to_numpy()[:-n_f]
    breakdowns = {
        "Straddle (ATM)":           dict(regime=per_regime(pnl_sd,  lab),   quintile=per_quintile(pnl_sd,  vrp)),
        "Strangle (16Δ)":           dict(regime=per_regime(pnl_stg, lab),   quintile=per_quintile(pnl_stg, vrp)),
        "Iron Condor (16/5Δ)":      dict(regime=per_regime(pnl_ic,  lab),   quintile=per_quintile(pnl_ic,  vrp)),
        "Long Vol (ATM buy)":       dict(regime=per_regime(pnl_lv,  lab),   quintile=per_quintile(pnl_lv,  vrp)),
        "Calendar 9/30 (ATM call)": dict(regime=per_regime(pnl_cal, lab_c), quintile=per_quintile(pnl_cal, vrp_c)),
    }

    return dict(
        params=dict(dte=BT_DTE, slip=int(BT_SLIP * 100), q=BT_Q, exp_td=n_exp,
                    rate="^IRX 13w T-bill", unit="% of spot notional", years=jnum(yrs, 1),
                    skew_dn=SKEW_DN, skew_up=SKEW_UP, cal_front_td=n_f),
        eq_dates=ds, structures=structures, breakdowns=breakdowns,
    )


# --------------------------------------------------------------------------- #
# assemble
# --------------------------------------------------------------------------- #
def build_data(px):
    series = {d: dte_series(px, d) for d in DTES}
    labels, slope, up = regime_series(px)
    rate = load_rate(px.index)

    lookbacks = {}
    for N, name in LOOKBACKS.items():
        if N > len(px):
            continue
        cards = cards_for(series, N)
        top = sorted(cards, key=lambda c: (c["composite"] is not None, c["composite"]), reverse=True)[0]
        lookbacks[str(N)] = dict(name=name, sessions=min(N, len(px)), cards=cards, top=top,
                                 win_start=px.index[-min(N, len(px))].strftime("%d %b %Y"),
                                 win_end=px.index[-1].strftime("%d %b %Y"))

    term = [{"lbl": lbl, "v": jnum(float(px[col].iloc[-1]), 2)}
            for lbl, col in [("9d", "^VIX9D"), ("30d", "^VIX"), ("3M", "^VIX3M"), ("6M", "^VIX6M")]
            if not np.isnan(px[col].iloc[-1])]

    tail = px.tail(CHART_TAIL)
    ti   = tail.index
    ch = dict(
        dates=[d.strftime("%Y-%m-%d") for d in ti],
        iv30 = arr(series[30]["iv"].reindex(ti)),
        rv30 = arr(series[30]["rvt"].reindex(ti)),
        vrp7 = arr(series[7]["vrp_now"].reindex(ti)),
        vrp14= arr(series[14]["vrp_now"].reindex(ti)),
        vrp21= arr(series[21]["vrp_now"].reindex(ti)),
        vrp30= arr(series[30]["vrp_now"].reindex(ti)),
        vrp30_fwd = arr(series[30]["vrp_hist"].reindex(ti)),
        vix9d= arr(px["^VIX9D"].reindex(ti)),
        vix  = arr(px["^VIX"].reindex(ti)),
        vix3m= arr(px["^VIX3M"].reindex(ti)),
        reg  = [REG_ORDER.index(l) for l in labels.reindex(ti)],
    )

    fc_block = forecast_block(px, labels)
    src = px.attrs.get("sources", [])
    return dict(
        asof=px.index[-1].strftime("%d %b %Y"),
        asof_iso=px.index[-1].strftime("%Y-%m-%d"),
        first=px.index[0].strftime("%d %b %Y"),
        n_total=len(px),
        sources=src, n_ok=sum(1 for s in src if s["ok"]),
        spx=jnum(float(px["SPX"].iloc[-1]), 2),
        iv30=jnum(float(px["^VIX"].iloc[-1]), 1),
        vrp30=jnum(series[30]["cur_vrp"], 1),
        term=term,
        lookbacks=lookbacks, default_lb=str(DEFAULT_LB),
        weights={k: int(v * 100) for k, v in W.items()},
        colors=CARD_COLORS,
        charts=ch,
        regime=regime_block(px, labels, slope, up, series[30]["vrp_hist"], series[30]["vrp_now"]),
        forecast=fc_block,
        forward=forward_test(px, labels, fc_block),
        analog=analog_block(px, labels),
        validation=validation_block(series[30]["vrp_now"], series[30]["vrp_hist"]),
        backtest=backtest_block(px, series[30]["iv"], labels, series[30]["vrp_now"], rate),
        matrix=vrp_matrix_block(series),
        node_days={"9d": 9, "30d": 30, "3M": 93, "6M": 186},
        dtes=DTES,
    )


def main():
    print("Downloading SPX + VIX family (free) ...", file=sys.stderr)
    px = load()
    print(f"  {len(px)} sessions, {px.index[0].date()} -> {px.index[-1].date()}", file=sys.stderr)
    data = build_data(px)
    for c in data["lookbacks"][data["default_lb"]]["cards"]:
        print(f"  {c['dte']:>2}DTE  IV {c['cur_iv']:5.1f}  VRP {c['cur_vrp']:+5.1f}  "
              f"pct {c['vrp_pct']:4.0f}  hit {c['hit']:3.0f}%  comp {c['composite']:4.1f}", file=sys.stderr)
    r = data["regime"]["current"]
    v = data["validation"]
    print(f"  regime: {r['label']}  slope {r['slope']}  (VIX {r['vix_pct']:.0f}th pct)", file=sys.stderr)
    print(f"  validation: mean VRP {v['mean']}  hit {v['hit']}%  IR {v['info_ratio']}  n {v['n']}", file=sys.stderr)
    b = data["backtest"]
    for nm, s in b["structures"].items():
        f, k = s["flat"]["all"], s["skew"]["all"]
        print(f"  backtest {nm:<20} flat: mean {f['stats']['mean']:>5}% hit {f['stats']['hit']:>3}% worst {f['stats']['worst']:>6}% tot {f['tot']:>6}% | "
              f"skew: mean {k['stats']['mean']:>5}% tot {k['tot']:>6}% maxDD {k['mdd']}%", file=sys.stderr)

    html = TEMPLATE.replace("__DATA__", json.dumps(data))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vrpx_report.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nWrote {out}  ({len(html)//1024} KB)", file=sys.stderr)


# --------------------------------------------------------------------------- #
# HTML  (all tabs interactive; JS renders from embedded JSON)
# --------------------------------------------------------------------------- #
TEMPLATE = r"""<meta charset="utf-8">
<title>VRPX — Volatility Risk Premium Analyzer (open-data rebuild)</title>
<style>
  :root{
    --bg:#05070a;--panel:#0a0e14;--panel-2:#0c1119;--line:#141c26;--line-2:#1c2733;
    --ink:#c8d3de;--ink-dim:#7b8aa0;--ink-faint:#4d5a6d;
    --green:#3ddc84;--green-dim:#1f7a4d;--yellow:#ffd645;--amber:#f0a93b;
    --red:#ff5d6c;--pink:#ff5d8f;--blue:#4aa8ff;--purple:#a98bff;
    --mono:ui-monospace,"Cascadia Code","Cascadia Mono",Consolas,"SFMono-Regular",Menlo,monospace;
  }
  *{box-sizing:border-box} html,body{margin:0}
  body{background:radial-gradient(1200px 500px at 70% -10%,rgba(61,220,132,.05),transparent 60%),var(--bg);
    color:var(--ink);font-family:var(--mono);font-size:13px;line-height:1.35;letter-spacing:.2px;-webkit-font-smoothing:antialiased}
  .app{max-width:1560px;margin:0 auto;padding:14px 18px 40px}
  .topbar{display:flex;align-items:baseline;gap:18px;padding:6px 2px 14px;flex-wrap:wrap}
  .brand{font-size:26px;font-weight:700;letter-spacing:2px;color:#eef4fa} .brand b{color:var(--green)}
  .subtitle{color:var(--ink-faint);letter-spacing:4px;font-size:11px;text-transform:uppercase}
  .subtitle .homage{color:var(--ink-faint);text-decoration:none;border-bottom:1px dotted var(--ink-faint);
    letter-spacing:1px;text-transform:none;transition:.15s}
  .subtitle .homage:hover{color:var(--green);border-color:var(--green)}
  .top-right{margin-left:auto;display:flex;align-items:center;gap:16px;font-size:12px;flex-wrap:wrap}
  .kv{display:flex;gap:6px;align-items:baseline} .kv .k{color:var(--ink-faint);letter-spacing:1px}
  .kv .v{color:#eef4fa;font-weight:600;font-variant-numeric:tabular-nums}
  .badge{border:1px solid var(--green-dim);color:var(--green);padding:2px 8px;font-size:11px;letter-spacing:1px;border-radius:3px;background:rgba(61,220,132,.06)}
  .date{color:var(--ink-dim)}
  .tabs{display:flex;align-items:center;gap:6px;border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:9px 2px;flex-wrap:wrap}
  .tab{color:var(--ink-dim);padding:5px 11px;border:1px solid transparent;border-radius:4px;letter-spacing:.8px;font-size:11px;white-space:nowrap;cursor:pointer;transition:.12s;background:none;font-family:inherit}
  .tab:hover{color:var(--ink);background:var(--panel)}
  .tab.active{color:#05070a;background:var(--yellow);font-weight:700}
  .tab.export{color:var(--green);border-color:var(--green-dim);background:rgba(61,220,132,.06)}
  .tab.export.active{color:#05070a;background:var(--green)}
  .spacer{margin-left:auto}
  .lookback{display:flex;align-items:center;gap:6px;color:var(--ink-faint);font-size:11px;letter-spacing:1px}
  .lb{padding:3px 9px;border:1px solid var(--line-2);border-radius:4px;color:var(--ink-dim);font-size:11px;cursor:pointer;font-family:inherit;background:none;transition:.12s}
  .lb:hover{color:var(--ink)} .lb.active{background:var(--panel-2);color:var(--green);border-color:var(--green-dim)}
  .strip{border:1px solid var(--line);border-radius:6px;margin-top:12px;background:linear-gradient(180deg,var(--panel),var(--panel-2))}
  .strip.window{border-left:3px solid var(--purple);padding:10px 14px}
  .strip.status{border-left:3px solid var(--green);padding:10px 14px;display:flex;gap:22px;flex-wrap:wrap;align-items:center}
  .win-row{display:flex;gap:26px;flex-wrap:wrap;align-items:baseline}
  .lbl{color:var(--ink-faint);letter-spacing:1px;font-size:10.5px;margin-right:6px}
  .val{color:var(--ink);font-variant-numeric:tabular-nums} .val.dim{color:var(--ink-dim)}
  .win-sub{color:var(--ink-faint);margin-top:7px;font-size:11.5px}
  .status .live{color:var(--green);letter-spacing:1px;font-weight:600}
  .status .regime{letter-spacing:1px} .chk{color:var(--green)} .chx{color:var(--red)}
  .cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:14px}
  @media(max-width:1100px){.cards{grid-template-columns:repeat(2,1fr)}}
  @media(max-width:620px){.cards{grid-template-columns:1fr}}
  .card{--c:var(--ink);border:1px solid var(--line-2);border-top:2px solid var(--c);border-radius:6px;background:linear-gradient(180deg,rgba(255,255,255,.015),transparent),var(--panel);padding:14px 16px 16px;position:relative}
  .card .tag{position:absolute;top:12px;right:12px;font-size:10px;letter-spacing:1px;padding:2px 7px;border-radius:3px;border:1px solid}
  .tag.best{color:var(--pink);border-color:rgba(255,93,143,.5);background:rgba(255,93,143,.08)}
  .tag.weak{color:var(--green);border-color:rgba(61,220,132,.4);background:rgba(61,220,132,.06)}
  .card .dte{font-size:30px;font-weight:700;color:var(--c);line-height:1}
  .card .dte small{font-size:12px;font-weight:600;color:var(--c);opacity:.75;margin-left:3px;letter-spacing:1px}
  .card .composite{font-size:46px;font-weight:700;color:#eef4fa;line-height:1;margin:12px 0 2px;font-variant-numeric:tabular-nums}
  .card .comp-lbl{color:var(--ink-faint);font-size:10px;letter-spacing:1px}
  .metrics{margin:14px 0 12px;display:flex;flex-direction:column;gap:5px}
  .m{display:flex;justify-content:space-between;font-size:11.5px}
  .m .mk{color:var(--ink-dim)} .m .mv{color:var(--ink);font-variant-numeric:tabular-nums} .m .mv.neg{color:var(--red)}
  .bars{display:flex;flex-direction:column;gap:7px;border-top:1px solid var(--line);padding-top:12px}
  .bar{display:grid;grid-template-columns:50px 1fr 26px;align-items:center;gap:9px;font-size:10px}
  .bar .bk{color:var(--ink-faint);letter-spacing:.5px}
  .bar .track{display:block;height:8px;background:#0e141d;border:1px solid var(--line-2);border-radius:4px;overflow:hidden}
  .bar .fill{display:block;min-width:2px;height:100%;background:var(--c);border-radius:4px;box-shadow:0 0 7px -1px var(--c);transition:width .35s ease}
  .bar .bv{color:var(--c);text-align:right;font-variant-numeric:tabular-nums;font-weight:600}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}
  @media(max-width:900px){.grid2{grid-template-columns:1fr}}
  .box{border:1px solid var(--line);border-radius:6px;padding:14px 16px;background:linear-gradient(180deg,var(--panel),var(--panel-2));margin-top:12px}
  .box-title{color:var(--ink-dim);letter-spacing:1px;font-size:11px;margin-bottom:10px}
  .box-note{color:var(--ink-faint);font-size:10.5px;margin-top:8px;line-height:1.6}
  .chart{width:100%;height:auto;display:block}
  .legend{display:flex;gap:14px;flex-wrap:wrap;font-size:10.5px;color:var(--ink-dim);margin-top:8px}
  .legend .lg{display:flex;align-items:center;gap:5px} .legend .sw{width:10px;height:10px;border-radius:2px;display:inline-block}
  .context{border:1px solid var(--line);border-left:3px solid var(--green);border-radius:6px;margin-top:12px;padding:14px 16px;background:linear-gradient(180deg,var(--panel),var(--panel-2))}
  .ctx-title{color:var(--green);letter-spacing:1px;font-size:12px} .ctx-sub{color:var(--ink-faint);font-size:11.5px;margin-top:4px;max-width:640px}
  .ctx-head{display:flex;justify-content:space-between;gap:20px;flex-wrap:wrap;align-items:flex-start}
  .ctx-meta{text-align:right;color:var(--ink-dim);font-size:11px;line-height:1.7} .ctx-meta b{color:var(--yellow)}
  .ctx-cols{display:grid;grid-template-columns:repeat(3,1fr);gap:24px;margin-top:16px}
  @media(max-width:900px){.ctx-cols{grid-template-columns:1fr}}
  .col-h{font-size:10.5px;letter-spacing:1px;padding-bottom:8px;border-bottom:1px solid var(--line)}
  .col.fav .col-h{color:var(--green)} .col.acc .col-h{color:var(--yellow)} .col.cau .col-h{color:var(--red)}
  .citem{margin-top:12px} .citem .name{color:#dbe4ee;font-size:12px} .citem .desc{color:var(--ink-dim);font-size:11px}
  .citem .ev{color:var(--ink-faint);font-size:10.5px;margin-top:2px;font-variant-numeric:tabular-nums}
  .col.cau .citem .desc{color:var(--ink-faint)}
  .hist{font-size:12px;line-height:1.6;color:var(--ink-dim);margin-top:14px;border-top:1px solid var(--line);padding-top:12px} .hist b{color:var(--green)} .hist .neg{color:var(--red)}
  .panel{display:none} .panel.active{display:block}
  table.tbl{width:100%;border-collapse:collapse;font-size:12px;margin-top:6px}
  table.tbl th,table.tbl td{text-align:right;padding:6px 10px;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}
  table.tbl th{color:var(--ink-faint);font-weight:400;letter-spacing:1px;font-size:10.5px}
  table.tbl td:first-child,table.tbl th:first-child{text-align:left;color:var(--ink)}
  .heat{border-collapse:separate;border-spacing:2px;font-size:10px;width:100%;overflow-x:auto;display:block}
  .heat td,.heat th{padding:5px 4px;text-align:center;font-variant-numeric:tabular-nums;white-space:nowrap}
  .heat th{color:var(--ink-faint);font-weight:400}
  .heat td.k{text-align:left;color:var(--ink);padding-right:8px}
  .heat td.c{border-radius:3px;color:#05070a;font-weight:600}
  .reg-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px}
  .reg-cell{border:1px solid var(--line-2);border-radius:6px;padding:12px 14px;background:var(--panel-2)}
  .reg-cell .rk{color:var(--ink-faint);font-size:10px;letter-spacing:1px}
  .reg-cell .rv{color:#eef4fa;font-size:20px;font-weight:700;font-variant-numeric:tabular-nums;margin-top:4px}
  .mtx{border-collapse:separate;border-spacing:2px;font-size:10.5px;width:100%}
  .mtx td,.mtx th{padding:6px;text-align:center;font-variant-numeric:tabular-nums}
  .mtx th{color:var(--ink-faint);font-weight:400;font-size:9.5px} .mtx td.k{text-align:left;color:var(--ink)}
  .mtx td.c{border-radius:3px;color:#eef4fa}
  .fcbar{display:grid;grid-template-columns:130px 1fr 40px;align-items:center;gap:8px;font-size:11px;margin-top:6px}
  .fcbar .fk{color:var(--ink-dim)} .fcbar .track{height:14px;background:var(--line-2);border-radius:3px;overflow:hidden}
  .fcbar .track{display:block} .fcbar .fill{display:block;height:100%;border-radius:3px} .fcbar .fv{text-align:right;color:var(--ink);font-variant-numeric:tabular-nums}
  .bt-ctrls{display:flex;align-items:center;gap:7px;flex-wrap:wrap;margin-bottom:12px}
  .bt-ctrls .lb{cursor:pointer}
  .helptext{font-size:12px;color:var(--ink-dim);line-height:1.75;max-width:960px}
  .helptext b{color:var(--ink)}
  .helptext .good,table.tbl td.good{color:var(--green)}
  .helptext .bad,table.tbl td.bad{color:var(--red)}
  .helptext .warn,table.tbl td.warn{color:var(--amber)}
  .h4{display:block;color:var(--green);font-size:10.5px;letter-spacing:1.5px;margin:14px 0 4px}
  [data-tip]{cursor:help}
  #tip{position:fixed;z-index:99;max-width:330px;background:#0d1420;border:1px solid var(--line-2);
    border-left:2px solid var(--green);padding:8px 11px;font-size:11px;line-height:1.55;color:var(--ink);
    border-radius:4px;pointer-events:none;display:none;box-shadow:0 6px 20px rgba(0,0,0,.55)}
  .rules{font-size:11.5px;color:var(--ink-dim);line-height:1.7;margin-top:12px} .rules code{color:var(--amber)}
  .kvlist{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:8px 26px;font-size:12px}
  .kvlist .row{display:flex;justify-content:space-between;border-bottom:1px solid var(--line);padding:6px 0}
  .kvlist .rk{color:var(--ink-dim)} .kvlist .rv{color:var(--ink);font-variant-numeric:tabular-nums}
  .foot{color:var(--ink-faint);font-size:10.5px;margin-top:22px;line-height:1.6} .foot b{color:var(--ink-dim)}
  button.pbtn{padding:6px 14px;border:1px solid var(--line-2);border-radius:4px;color:var(--green);background:var(--panel-2);cursor:pointer;font-family:inherit;font-size:12px}
  @media print{.tabs,.top-right{display:none} .panel{display:block!important} .box{break-inside:avoid}}
</style>
<div id="stale-overlay" style="display:none;position:fixed;top:0;left:0;right:0;z-index:9999;
  background:#3a1417;border-bottom:2px solid var(--red);color:#ffdfe3;font-family:var(--mono);
  font-size:12.5px;letter-spacing:.5px;padding:9px 16px;text-align:center;box-shadow:0 2px 12px rgba(0,0,0,.5)">
  <span id="stale-msg"></span>
  <button id="stale-x" style="margin-left:14px;background:none;border:1px solid var(--red);color:#ffdfe3;
    cursor:pointer;padding:1px 7px;border-radius:3px;font-family:inherit">dismiss</button>
</div>
<div class="app">
  <div class="topbar">
    <div class="brand"><b>VRPX</b></div>
    <div class="subtitle">Volatility Risk Premium Term Structure Analyzer · open-data rebuild
      · inspired by <a class="homage" href="https://atmoptionsedge.ch/" target="_blank" rel="noopener"
           data-tip="Tomas Byron's homepage — ATM Options Edge. This dashboard is an independent open-data rebuild of his VRPX PRO; free public data only, all scoring logic our own and disclosed in Settings.">Tomas Byron</a>'s
      <a class="homage" href="https://www.youtube.com/watch?v=cHPew0fKbgQ" target="_blank" rel="noopener"
           data-tip="The original VRPX PRO walkthrough video that inspired this rebuild.">VRPX PRO</a>
      · <a class="homage" href="https://www.youtube.com/@tombyr4907" target="_blank" rel="noopener"
           data-tip="Tomas Byron's YouTube channel.">YT ↗</a></div>
    <div class="top-right">
      <span class="badge" id="t-badge" data-tip="Data status: green when all five free sources (SPX + VIX family) are fresh; degrades to n/5 with a warning if a source goes stale.">●</span>
      <span class="kv" data-tip="S&P 500 close (^GSPC, Yahoo) on the as-of date."><span class="k">SPX</span><span class="v" id="t-spx"></span></span>
      <span class="kv" data-tip="30-day implied vol = VIX on the as-of date."><span class="k">IV30</span><span class="v" id="t-iv30"></span></span>
      <span class="kv" data-tip="Nowcast 30-day volatility risk premium = IV30 minus trailing 21-session realized vol. Positive = options priced rich vs recent movement."><span class="k">VRP30</span><span class="v" id="t-vrp30"></span></span>
      <span class="date" id="t-asof"></span>
    </div>
  </div>

  <div class="tabs" id="tabs">
    <button class="tab active" data-tab="overview" data-tip="DTE scorecards + evidence-based option-strategy context.">OVERVIEW</button>
    <button class="tab" data-tab="matrix" data-tip="Monthly average VRP per DTE — 24-month heatmap.">VRP MATRIX</button>
    <button class="tab" data-tab="charts" data-tip="IV vs realized vol, VRP by DTE, VIX term-structure history.">CHARTS</button>
    <button class="tab" data-tab="regime" data-tip="Current regime diagnostics + per-regime forward-VRP evidence.">REGIME ANALYSIS</button>
    <button class="tab" data-tab="validation" data-tip="Does the VRP signal carry forward information? Quintile test + equity proxy + analog engine (k-NN over state).">VALIDATION</button>
    <button class="tab" data-tab="backtest" data-tip="Black-Scholes-reconstructed option structures over ~15 years, with skew & regime-gate toggles.">BACKTEST</button>
    <button class="tab" data-tab="timeline" data-tip="Rule-based regime label for every session, as a colored ribbon.">REGIME TIMELINE</button>
    <button class="tab" data-tab="forecast" data-tip="Streak-adjusted persistence forecast (5 trading days) + retroactive self-calibration. Regime-state estimate, not a market prediction.">REGIME FORECAST</button>
    <button class="tab" data-tab="help" data-tip="How to read every number, derive trade inputs, size from the tail — and the red flags.">HELP</button>
    <button class="tab" data-tab="settings" data-tip="Data sources, parameters, composite weights, sub-score definitions, regime rules.">SETTINGS</button>
    <button class="tab export" data-tab="export" data-tip="Print or save the report as PDF.">EXPORT PDF</button>
    <div class="spacer"></div>
    <div class="lookback" data-tip="Window for percentiles, hit rate, composite, charts and validation: 3M=63 · 6M=126 · 1Y=252 · 2Y=504 · 5Y=1260 sessions.">LOOKBACK:<span id="lbs"></span></div>
  </div>

  <div class="strip window">
    <div class="win-row">
      <span><span class="lbl">📅 DATA WINDOW</span><span class="val" id="w-range"></span> <span class="val dim" id="w-sess"></span></span>
      <span><span class="lbl">DATA SOURCE:</span><span class="val dim">SPX ^GSPC (Yahoo) + VIX/VIX9D/VIX3M/VIX6M (Cboe via Yahoo)</span></span>
    </div>
    <div class="win-sub"><span class="lbl">IV:</span><span class="val">Total-variance interpolation VIX9D→VIX</span>
      <span class="lbl" style="margin-left:14px">RV:</span><span class="val">Forward realized vol, horizon-matched</span></div>
  </div>

  <div class="strip status">
    <span class="live" id="s-live"></span>
    <span id="s-src"></span>
    <span>REGIME <span class="regime" id="s-regime"></span></span>
    <span class="lbl" id="s-slope" style="margin:0" data-tip="Slope = VIX3M ÷ VIX: >1 contango (calm, carry-friendly), <1 backwardation (stress). Plus VIX percentile vs full history, SPX vs its 200-day average, and the 20-day return."></span>
  </div>

  <div class="panel active" data-panel="overview">
    <div class="cards" id="cards"></div>
    <div class="context">
      <div class="ctx-head">
        <div><div class="ctx-title">TRADE CONTEXT LAYER · EVIDENCE ONLY</div>
          <div class="ctx-sub" id="ctx-sub"></div></div>
        <div class="ctx-meta" id="ctx-meta"></div>
      </div>
      <div class="ctx-cols">
        <div class="col fav"><div class="col-h" data-tip="Historically paid in the current regime — measured thresholds: mean ≥0.3%/trade and win ≥65% (short vol), analogous per family. Not a trade instruction.">FAVORED / WATCH</div><div id="col-fav"></div></div>
        <div class="col acc"><div class="col-h" data-tip="Mixed or ≈zero historical record in this regime — usable, but context dependent.">ACCEPTABLE / CONTEXT DEPENDENT</div><div id="col-acc"></div></div>
        <div class="col cau"><div class="col-h" data-tip="Historically weak in this regime, or the measured risk profile argues against aggression.">CAUTION / AVOID AGGRESSION</div><div id="col-cau"></div></div>
      </div>
      <div class="hist" id="ctx-hist"></div>
    </div>
  </div>

  <div class="panel" data-panel="matrix">
    <div class="box"><div class="box-title">▶ VRP MATRIX · monthly nowcast VRP by DTE · last 24 months</div>
      <div style="overflow-x:auto"><table class="heat" id="matrix-tbl"></table></div>
      <div class="box-note">Cell = average nowcast VRP (IV − trailing RV) that month, in vol points. Green = IV rich vs realized, red = IV cheap.</div></div>
  </div>

  <div class="panel" data-panel="charts">
    <div class="box"><div class="box-title" id="c1-title"></div><div id="chart-ivrv"></div><div class="legend" id="c1-leg"></div></div>
    <div class="box"><div class="box-title" id="c2-title"></div><div id="chart-vrp"></div><div class="legend" id="c2-leg"></div></div>
    <div class="box"><div class="box-title" id="c3-title"></div><div id="chart-term"></div><div class="legend" id="c3-leg"></div></div>
  </div>

  <div class="panel" data-panel="regime">
    <div class="box"><div class="box-title">▶ CURRENT REGIME</div><div class="reg-grid" id="reg-grid"></div></div>
    <div class="box"><div class="box-title" id="reg-tbl-title"></div>
      <table class="tbl" id="reg-tbl"><thead><tr><th>Regime</th><th>Days (full)</th><th>Avg VRP (full)</th><th>Hit (full)</th><th>Avg VRP (window)</th><th>Hit (window)</th></tr></thead><tbody></tbody></table>
      <div class="box-note" id="reg-tbl-note"></div></div>
  </div>

  <div class="panel" data-panel="validation">
    <div class="box"><div class="box-title">▶ SIGNAL VALIDATION · does a richer current VRP pay more forward? · quintiles, full history</div>
      <table class="tbl" id="val-tbl"><thead><tr><th>Quintile</th><th>N</th><th>Avg signal (nowcast VRP)</th><th>Avg forward VRP</th><th>Hit rate</th></tr></thead><tbody></tbody></table>
      <div class="box-note" id="val-note"></div></div>
    <div class="box"><div class="box-title" id="eq-title"></div><div id="chart-eq"></div>
      <div class="box-note">Overlapping ~21-day short-vol trades, 1 per day, P&L in vol points (illustrative — overlapping windows, no costs/vega scaling).</div></div>
    <div class="grid2" style="margin-top:0">
      <div class="box"><div class="box-title">▶ SUMMARY (full history)</div><div class="kvlist" id="val-sum"></div></div>
      <div class="box"><div class="box-title">▶ READING IT</div><div class="box-note" style="font-size:11.5px">A monotone rise in "Avg forward VRP" across quintiles means the current-VRP signal has genuine forward information: when IV looks rich vs recent realized, subsequent captured premium is higher. A flat/noisy pattern means little edge.</div></div>
    </div>

    <div class="box"><div class="box-title" id="an-title"></div>
      <div class="box-note" style="margin-top:0" id="an-sub"></div>
      <div class="reg-grid" id="an-state" style="margin-top:10px"></div></div>
    <div class="box"><div class="box-title">▶ HISTORICAL MATCHES · what followed the analog days · by horizon</div>
      <table class="tbl" id="an-h"><thead><tr><th>Horizon</th><th>Matches</th><th>Avg SPX return</th><th>Median</th><th>Worst</th><th>Best</th><th>Positive %</th></tr></thead><tbody></tbody></table>
      <div class="box-note" id="an-h-note"></div></div>
    <div class="box"><div class="box-title">▶ TOP ANALOG DAYS · nearest neighbours in state space</div>
      <table class="tbl" id="an-top"><thead><tr><th>Date</th><th>Regime</th><th>Distance</th><th>Similarity</th><th>VIX</th><th>21d SPX ret</th><th>21d fwd RV</th></tr></thead><tbody></tbody></table>
      <div class="box-note">Similarity is a display transform of distance, <b>not a probability</b>. Neighbours use only observable state (VIX level, term structure, RV composite, 21-day efficiency) and are restricted to days with a complete forward window.</div></div>
  </div>

  <div class="panel" data-panel="backtest">
    <div class="box"><div class="box-title" id="bt-title"></div>
      <div class="box-note" id="bt-sub" style="margin-top:0;margin-bottom:10px"></div>
      <div class="bt-ctrls" id="bt-ctrls">
        <span class="lbl">SKEW</span><span id="bt-skew"></span>
        <span class="lbl" style="margin-left:18px">REGIME GATE</span><span id="bt-gate"></span>
      </div>
      <div id="bt-eq"></div><div class="legend" id="bt-leg"></div></div>
    <div class="box"><div class="box-title" id="bt-cmp-title"></div>
      <table class="tbl" id="bt-cmp"><thead><tr><th>Metric</th></tr></thead><tbody></tbody></table>
      <div class="box-note">Straddle = max premium, max tail. Strangle 16Δ = less premium, wider safe zone. Iron Condor = defined risk — the wing <b>caps the worst case</b>. Long Vol = the mirror (insurance cost). Calendar 9/30 runs on its own 6-session cadence — its equity line is resampled onto the shared date grid, totals are cadence-true. Toggle SKEW to shift the OTM legs (ATM structures unaffected); REGIME GATE applies to all.</div></div>
    <div class="box" id="bt-strat-box" style="padding:10px 16px">
      <div class="bt-ctrls" style="margin:0"><span class="lbl">BREAKDOWN TABLES · STRATEGY</span><span id="bt-strat"></span></div></div>
    <div class="grid2" style="margin-top:0">
      <div class="box"><div class="box-title" id="bt-q-title"></div>
        <table class="tbl" id="bt-q"><thead><tr><th>Quintile</th><th>Avg nowcast VRP</th><th>Avg P&L %</th><th>Win rate</th></tr></thead><tbody></tbody></table></div>
      <div class="box"><div class="box-title" id="bt-reg-title"></div>
        <table class="tbl" id="bt-reg"><thead><tr><th>Regime</th><th>Trades</th><th>Avg P&L %</th><th>Win rate</th><th>5th pct</th><th>Worst</th></tr></thead><tbody></tbody></table></div>
    </div>
    <div class="box"><div class="box-title">▶ METHOD & CAVEATS</div><div class="box-note" id="bt-note" style="font-size:11px"></div></div>
  </div>

  <div class="panel" data-panel="timeline">
    <div class="box"><div class="box-title" id="tl-title"></div><div id="ribbon"></div><div class="legend" id="tl-leg"></div>
      <div class="box-note">Each vertical band = the rule-based regime on that day. Rules are shown in the Regime Analysis / Settings tabs.</div></div>
  </div>

  <div class="panel" data-panel="forecast">
    <div class="box"><div class="box-title" id="fc-head"></div>
      <div class="box-note" style="margin-top:0" id="fc-sub"></div>
      <div class="reg-grid" id="fc-cards" style="margin-top:10px"></div></div>
    <div class="box"><div class="box-title" id="fc-title"></div><div id="fc-bars"></div>
      <div class="box-note">Next-regime probabilities over the forecast horizon. Model = streak-adjusted persistence (P(stay | regime, streak bucket), remainder from the transition off-diagonal) — chosen over a 4-component ensemble by measurement (LogLoss 0.923). A regime-state estimate, <b>not</b> a forecast of SPX, VIX, returns or trade profitability.</div></div>
    <div class="box"><div class="box-title" id="fc-cal-title"></div>
      <table class="tbl" id="fc-cal"><thead><tr><th>Forecast confidence</th><th>N</th><th>Avg confidence</th><th>Actual accuracy</th><th>Calibration error</th></tr></thead><tbody></tbody></table>
      <div class="box-note" id="fc-cal-note"></div></div>
    <div class="box"><div class="box-title" id="fw-title"></div>
      <div class="box-note" style="margin-top:0" id="fw-sub"></div>
      <div class="reg-grid" id="fw-cards" style="margin-top:10px"></div>
      <table class="tbl" id="fw-tbl" style="margin-top:10px"><thead><tr><th>Logged</th><th>Regime</th><th>Predicted next</th><th>Prob</th><th>Resolved</th><th>Actual</th><th>Status</th></tr></thead><tbody></tbody></table></div>
    <div class="box"><div class="box-title">▶ FORECAST JOURNAL (retroactive) · last 15 days · resolved HIT/MISS after the horizon</div>
      <table class="tbl" id="fc-jrn"><thead><tr><th>Date</th><th>Regime that day</th><th>Predicted next</th><th>Prob</th><th>Actual</th><th>Status</th></tr></thead><tbody></tbody></table></div>
    <div class="box"><div class="box-title">▶ EMPIRICAL TRANSITION MATRIX · P(next-day regime | today) % · full history</div>
      <div style="overflow-x:auto"><table class="mtx" id="mtx"></table></div>
      <div class="box-note">Descriptive only — day→next-day transition counts, row-normalised. The forecast above uses the 5-day streak-adjusted model, not this 1-day matrix.</div></div>
  </div>

  <div class="panel" data-panel="help">
    <div class="box"><div class="box-title">▶ WHAT THIS TOOL MEASURES — THE ONE IDEA</div>
      <div class="helptext">
        S&P-500 options have historically been priced for <b>more movement than then occurred</b>. The difference — implied vol minus subsequently realized vol — is the <b>volatility risk premium (VRP)</b>. Across <b><span id="h-n"></span> trading days</b> in this dataset it averaged <b class="good">+<span id="h-vrp"></span> vol points</b> and was positive on <b><span id="h-hitv"></span>%</b> of days: the payment option sellers collect for insuring other people's risk.<br><br>
        Each tab answers one question about that premium: <b>Overview</b> — where is it rich right now? · <b>Validation</b> — is the signal real? · <b>Backtest</b> — what did harvesting it earn and cost? · <b>Regime tabs</b> — under which market conditions? · <b>Trade Context</b> — which structures fit the current conditions, by measurement.
      </div></div>

    <div class="box"><div class="box-title">▶ THE 60-SECOND DAILY READ</div>
      <div class="helptext">
        <b>1 ·</b> Status strip: all five sources fresh? Which <b>regime</b>? (hover any regime name for its rule)<br>
        <b>2 ·</b> Cards: is <b>CUR VRP</b> positive? Where sits <b>VRP PCT</b>? Which DTE ranks #1 — and <i>why</i> (its five bars)?<br>
        <b>3 ·</b> Trade Context: which families are <b>Favored</b> — read the measured evidence line, not just the column.<br>
        <b>4 ·</b> Risk check: the ranked DTE's <b>WORST</b>, plus <b>p5 / Worst</b> for the current regime in the Backtest tab.<br>
        <b>5 ·</b> Cross-check: VRP Matrix for the recent trend, Charts for IV-vs-RV divergence.
      </div></div>

    <div class="box"><div class="box-title">▶ READING THE NUMBERS — WHAT MATTERS, WHAT DOESN'T</div>
      <table class="tbl"><thead><tr><th>Value</th><th>What it tells you</th><th>Rule of thumb</th></tr></thead><tbody>
        <tr><td>CUR VRP</td><td>Is there premium to sell at all?</td><td class="good">&gt; 0 required — below 0, selling premium fights the data</td></tr>
        <tr><td>VRP PCT</td><td>Richness vs its own history</td><td>≥ 60th = rich · ≤ 30th = thin, patience is a position</td></tr>
        <tr><td>HIT RATE</td><td>How often selling won historically</td><td class="warn">~80% is NORMAL here — never read it as safety; losses are far larger than wins</td></tr>
        <tr><td>WORST / p5</td><td>Size of the left tail</td><td class="bad">THE critical number — position sizing derives from it (next box)</td></tr>
        <tr><td>COMPOSITE</td><td>Premium quality/safety across DTEs</td><td>A ranking of <i>safety</i>, NOT expected P&amp;L — measured, calm conditions precede weaker short-vol returns (complacency). Read <i>why</i> via the five bars</td></tr>
        <tr><td>Slope VIX3M/VIX</td><td>Term-structure state</td><td>≥ 1 contango = carry-friendly · &lt; 0.97 backwardation = stress dynamics</td></tr>
        <tr><td>Validation quintiles</td><td>Does a richer signal pay more forward?</td><td>Rising Q1→Q5 = the signal carries real information</td></tr>
        <tr><td>RV COMPOSITE</td><td>rv10 − rv63: realized vol accelerating?</td><td class="warn">&gt; 0 in CALM/STEADY = measured drag on short vol (Ticket 02). No signal in TRANSITION (sign flips)</td></tr>
        <tr><td>Forecast reliability</td><td>Is the regime forecast well-calibrated?</td><td>Confidence vs actual accuracy per bucket; the tool shows its own bias openly (retroactive replay)</td></tr>
        <tr><td>Analog engine</td><td>What followed similar historical states?</td><td>Context, not alpha — measured fwd-RV skill only; return prediction ≈ 0</td></tr>
        <tr><td>Regime</td><td>Which historical playbook applies</td><td class="bad">NEUTRAL RANGE is the only regime where short vol lost on average (<span id="h-nr"></span>)</td></tr>
      </tbody></table></div>

    <div class="box"><div class="box-title">▶ FROM EVIDENCE TO TRADE INPUTS</div>
      <div class="helptext">
        <span class="h4">WHICH STRUCTURE</span>
        Take it from the measured comparison, not from preference. The <b>Straddle</b> earned the most, but its worst trade cost <b class="bad"><span id="h-wsd"></span>%</b> of notional. The <b>Iron Condor</b> gave up roughly half the return and capped the worst trade at <b class="good"><span id="h-wic"></span>%</b> — if you cannot hold through the straddle's tail, the condor IS your trade. <b>Long vol</b> is insurance costing ≈<span id="h-lv"></span>%/trade on average; the <b>calendar</b> earned ≈<span id="h-cal"></span>%/trade after costs — neither is a carry engine.
        <span class="h4">WHICH TENOR (DTE)</span>
        The #1-ranked card — but check <i>why</i>: a rank driven by <b>CARRY</b> (most premium per day) is a different trade than one driven by <b>RICH</b> (statistically stretched). Treat <b>7 DTE</b> as indicative only (short-end interpolation).
        <span class="h4">WHEN</span>
        Best measured conditions: CUR VRP positive, VRP PCT elevated, and a regime with a positive measured record (hover it). Staying flat in NEUTRAL RANGE costs nothing and skipped the only measured negative pocket.
        <span class="h4">HOW MUCH — THE CRITICAL ONE</span>
        Size from the tail, not the average: assume the worst historical episode repeats <i>tomorrow</i>. It cost <b><span id="h-wsd2"></span>%</b> of notional; if you accept losing at most 2% of your account on that repeat, short-straddle notional must stay below ≈ <b class="warn"><span id="h-size"></span>% of the account</b>. All figures are % of spot notional — margin leverage multiplies both gain and loss.
        <span class="h4">WHAT THE DATA SAYS NOT TO DO</span>
        · Don't equate calm with safe — the largest losses started in CALM/STEADY regimes.<br>
        · Don't read the ~80% hit rate as safety — the payoff is asymmetric by construction.<br>
        · Don't run permanent long-vol hedges expecting them to be free — measured cost ≈ 1%/month.
      </div></div>

    <div class="box"><div class="box-title">▶ RED FLAGS — WHEN NOT TO TRUST, WHEN NOT TO TRADE</div>
      <div class="helptext">
        <span class="bad">✗</span> Badge shows <b>DATA n/5</b> / DEGRADED → the snapshot is unreliable; fix data before reading anything.<br>
        <span class="bad">✗</span> <b>CUR VRP &lt; 0</b> → there is no premium; short vol is negative-EV territory right now.<br>
        <span class="bad">✗</span> Regime <b>NEUTRAL RANGE</b> → the only measured losing regime for short vol.<br>
        <span class="bad">✗</span> <b>Slope &lt; 0.97</b> (backwardation) → stress dynamics; premium is rich but paths are violent.<br>
        <span class="warn">△</span> <b>STAB low</b> or RV accel elevated → premium unstable; size down.<br>
        <span class="warn">△</span> <b>VOL SHOCK</b> statistics → thin sample (~51 days); treat loosely.
      </div></div>

    <div class="box"><div class="box-title">▶ WHAT THIS TOOL CANNOT TELL YOU</div>
      <div class="helptext">
        It describes history — it does not predict. The Markov forecast is a base-rate, not a signal. Backtest prices are Black-Scholes reconstructions (parametric skew, flat 2% cost, % of spot notional — not margin returns, no per-strike quotes). 7 DTE is interpolated. All numbers on this page are injected live from the current dataset and update with every refresh. None of this is investment advice — it is evidence to reason from.
      </div></div>
  </div>

  <div class="panel" data-panel="settings">
    <div class="box"><div class="box-title">▶ DATA SOURCES</div><div class="kvlist" id="set-src"></div></div>
    <div class="box"><div class="box-title">▶ PARAMETERS</div><div class="kvlist" id="set-par"></div></div>
    <div class="box"><div class="box-title">▶ COMPOSITE WEIGHTS (edit W in vrpx.py)</div><div class="kvlist" id="set-w"></div></div>
    <div class="box"><div class="box-title">▶ SUB-SCORES · 5 independent axes (0–100)</div><div class="rules" id="set-subs"></div></div>
    <div class="box"><div class="box-title">▶ REGIME RULES</div><div class="rules" id="set-rules"></div></div>
  </div>

  <div class="panel" data-panel="export">
    <div class="box"><div class="box-title">▶ EXPORT</div>
      <div style="color:var(--ink-dim);font-size:12px;line-height:1.8">Single self-contained HTML — no server, works offline.<br>
      <button class="pbtn" style="margin-top:8px" onclick="window.print()">🖨 Print / Save as PDF</button><br><br>
      Refresh with live data: <code style="color:var(--amber)">python vrpx.py</code></div></div>
  </div>

  <div class="foot" id="foot"></div>
</div>
<div id="tip"></div>

<script>
const DATA = __DATA__;
let state = {tab:"overview", lb:DATA.default_lb, skew:false, gate:false, btStrat:"Straddle (ATM)"};
const num=(x,d=1,s="")=>x==null?"—":x.toFixed(d)+s;
const ord=x=>{if(x==null)return"—";const n=Math.round(x),t=n%100;if(t>=11&&t<=13)return n+"th";return n+({1:"st",2:"nd",3:"rd"}[n%10]||"th");};
const cv=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const el=id=>document.getElementById(id);

/* ---- hover tooltips ---- */
const TIP={
  RICH:"Richness — percentile of the current VRP within this DTE's own trailing history. High = implied vol rich vs its own norm.",
  CARRY:"Carry per day — VRP ÷ calendar days, ranked across the four DTEs (min–max). Which tenor pays the most premium per day right now.",
  SAFETY:"Safety — merged reliability (info ratio) + tail-safety (1−|CVaR₅|) of forward VRP. These two measured the same thing (Spearman +0.89) so they are one axis. Higher = consistent edge with a milder left tail.",
  PATH:"Path-safety — trailing mean Max Adverse Excursion of the SPX path over the forward window, percentile of recent (21d) path risk within this DTE's own window history, inverted. High = the price path has been mild lately vs its norm (the path, not the VRP outcome).",
  STAB:"Stability — how steady recent VRP (21 sessions) is vs the whole window's dispersion. High = calm, predictable premium.",
  "CUR IV":"Implied vol for this DTE, interpolated from the VIX term structure (total-variance method). 7 DTE holds flat below the 9-day node.",
  "CUR VRP":"Nowcast VRP = current IV − trailing realized vol. Positive = options currently priced rich vs recent market movement.",
  "VRP PCT":"Where today's VRP sits in the trailing lookback distribution of forward VRP. 80th = richer than 80% of history.",
  "HIT RATE":"Share of lookback history where IV exceeded subsequently realized vol — how often short premium ended ahead, in vol points.",
  WORST:"Worst episode in the window: most negative forward VRP (realized vol exploded past implied). This DTE's left tail.",
  STABILITY:"0–100: steadiness of recent VRP vs the window's dispersion (same measure as the STAB bar).",
  COMPOSITE:"Weighted blend of the five sub-score axes (weights & definitions in Settings). Higher = historically more attractive premium-selling conditions at this DTE."
};
const REGTIP={
  "CALM CARRY":"Rule: slope ≥1.05 & SPX above 200d MA & VIX <17 — quiet contango bull. Caution: the largest historical short-vol losses started here.",
  "STEADY CARRY":"Rule: slope ≥1.0 & SPX above 200d MA — orderly carry conditions.",
  "NEUTRAL RANGE":"Rule: none of the other conditions met — no clear carry or stress signal. Historically the weakest regime for short vol.",
  "TRANSITION":"Rule: VIX 20–28 — elevated vol, direction unresolved.",
  "STRESS / BACKW":"Rule: VIX ≥28 or slope <0.97 (backwardation) — vol event underway. Premium rich; short vol historically paid, with fat risk.",
  "VOL SHOCK":"Rule: VIX ≥40 — crisis conditions. Thin sample."
};

/* ---- generic line chart ---- */
function lineChart(series, opts){
  const w=opts.w||960,h=opts.h||220,pl=40,pr=14,pt=12,pb=22,dates=opts.dates;
  let vs=[];series.forEach(s=>s.data.forEach(v=>{if(v!=null)vs.push(v)}));
  let vmin=Math.min(...vs),vmax=Math.max(...vs);const pad=(vmax-vmin)*0.06||1;vmin-=pad;vmax+=pad;
  if(opts.zero&&vmin>0)vmin=0; if(opts.zero&&vmax<0)vmax=0;
  const n=series[0].data.length;
  const X=i=>pl+i*(w-pl-pr)/(n-1), Y=v=>pt+(vmax-v)/(vmax-vmin)*(h-pt-pb);
  let g="";for(let k=0;k<4;k++){const gv=vmin+k*(vmax-vmin)/3;
    g+=`<line x1="${pl}" y1="${Y(gv).toFixed(1)}" x2="${w-pr}" y2="${Y(gv).toFixed(1)}" stroke="#141c26"/>
    <text x="6" y="${(Y(gv)+3).toFixed(1)}" fill="#4d5a6d" font-size="9">${gv.toFixed(0)}</text>`;}
  if(opts.zero&&vmin<0&&vmax>0)g+=`<line x1="${pl}" y1="${Y(0).toFixed(1)}" x2="${w-pr}" y2="${Y(0).toFixed(1)}" stroke="#3a4657" stroke-dasharray="3 3"/>`;
  let xl="";const ticks=6;for(let k=0;k<ticks;k++){const i=Math.round(k*(n-1)/(ticks-1));
    xl+=`<text x="${X(i).toFixed(1)}" y="${h-6}" fill="#4d5a6d" font-size="9" text-anchor="middle">${dates[i].slice(0,7)}</text>`;}
  let paths="";series.forEach(s=>{let d="",pen=false;
    s.data.forEach((v,i)=>{if(v==null){pen=false;return;}d+=(pen?"L":"M")+X(i).toFixed(1)+","+Y(v).toFixed(1)+" ";pen=true;});
    paths+=`<path d="${d}" fill="none" stroke="${cv(s.color)}" stroke-width="1.3"/>`;
    if(opts.area){let a=`M${X(0).toFixed(1)},${Y(0).toFixed(1)} `;s.data.forEach((v,i)=>{if(v!=null)a+="L"+X(i).toFixed(1)+","+Y(v).toFixed(1)+" ";});
      a+=`L${X(n-1).toFixed(1)},${Y(0).toFixed(1)} Z`;
      paths=`<path d="${a}" fill="${cv(s.color)}" opacity="0.14"/>`+paths;}});
  return `<svg viewBox="0 0 ${w} ${h}" class="chart" preserveAspectRatio="none" font-family="var(--mono)">${g}${xl}${paths}</svg>`;
}
function legend(series){return series.map(s=>`<span class="lg"><span class="sw" style="background:${cv(s.color)}"></span>${s.name}</span>`).join("");}
function REG_COLOR_JS(name){const o=DATA.regime.order,i=o.indexOf(name);return i<0?"--ink":DATA.regime.colors[i];}

/* ---- term-structure snapshot ---- */
function svgTermSnap(){
  const pts=DATA.term,w=520,h=180,pl=40,pr=16,pt=16,pb=26;
  const vals=pts.map(p=>p.v),vmin=Math.min(...vals)*0.96,vmax=Math.max(...vals)*1.04;
  const X=i=>pl+i*(w-pl-pr)/(pts.length-1),Y=v=>pt+(vmax-v)/(vmax-vmin)*(h-pt-pb),g=cv("--green");
  let grid="",dots="",labs="";
  for(let k=0;k<4;k++){const gv=vmin+k*(vmax-vmin)/3;grid+=`<line x1="${pl}" y1="${Y(gv).toFixed(1)}" x2="${w-pr}" y2="${Y(gv).toFixed(1)}" stroke="#141c26"/><text x="6" y="${(Y(gv)+3).toFixed(1)}" fill="#4d5a6d" font-size="9">${gv.toFixed(0)}</text>`;}
  const path=pts.map((p,i)=>(i?"L":"M")+X(i).toFixed(1)+","+Y(p.v).toFixed(1)).join(" ");
  pts.forEach((p,i)=>{dots+=`<circle cx="${X(i).toFixed(1)}" cy="${Y(p.v).toFixed(1)}" r="3" fill="${g}"/><text x="${X(i).toFixed(1)}" y="${(Y(p.v)-8).toFixed(1)}" fill="#cfe" font-size="9" text-anchor="middle">${p.v.toFixed(1)}</text>`;labs+=`<text x="${X(i).toFixed(1)}" y="${h-8}" fill="#7b8aa0" font-size="9" text-anchor="middle">${p.lbl}</text>`;});
  return `<svg viewBox="0 0 ${w} ${h}" class="chart" font-family="var(--mono)">${grid}<path d="${path}" fill="none" stroke="${g}" stroke-width="1.6"/>${dots}${labs}</svg>`;
}

/* ---- heat color ---- */
function heatColor(v,min,max){
  if(v==null)return "#0c1119";
  const mid=0; let t;
  if(v>=mid){t=max>mid?v/max:0;return `rgba(61,220,132,${(0.15+0.7*Math.min(t,1)).toFixed(2)})`;}
  t=min<mid?v/min:0;return `rgba(255,93,108,${(0.15+0.7*Math.min(t,1)).toFixed(2)})`;
}

/* ---- cards ---- */
function barRow(l,v){const x=v==null?0:Math.max(0,Math.min(100,v));return `<div class="bar" data-tip="${TIP[l]||""}"><span class="bk">${l}</span><span class="track"><span class="fill" style="width:${x.toFixed(0)}%"></span></span><span class="bv">${x.toFixed(0)}</span></div>`;}
function mRow(k,vHtml){return `<div class="m" data-tip="${TIP[k]||""}"><span class="mk">${k}</span>${vHtml}</div>`;}
function cardHTML(c,rank,color,best,weak){
  const tag=best?'<span class="tag best">★ BEST</span>':weak?'<span class="tag weak">WEAKEST</span>':"";const s=c.subs,neg=c.cur_vrp<0?"neg":"";
  return `<div class="card" style="--c:var(${color})">${tag}
    <div class="dte">${c.dte}<small>DTE</small></div><div class="composite" data-tip="${TIP.COMPOSITE}">${num(c.composite,1)}</div>
    <div class="comp-lbl" data-tip="${TIP.COMPOSITE}">COMPOSITE /100 · RANK #${rank}</div>
    <div class="metrics">
      ${mRow("CUR IV",`<span class="mv">${num(c.cur_iv,1,"%")}</span>`)}
      ${mRow("CUR VRP",`<span class="mv ${neg}">${num(c.cur_vrp,1,"%")}</span>`)}
      ${mRow("VRP PCT",`<span class="mv">${ord(c.vrp_pct)}</span>`)}
      ${mRow("HIT RATE",`<span class="mv">${num(c.hit,0,"%")}</span>`)}
      ${mRow("WORST",`<span class="mv neg">${num(c.worst,1,"%")}</span>`)}
      ${mRow("STABILITY",`<span class="mv">${num(s.stab,0)}/100</span>`)}</div>
    <div class="bars">${barRow("RICH",s.rich)}${barRow("CARRY",s.carry)}${barRow("SAFETY",s.safety)}${barRow("PATH",s.path)}${barRow("STAB",s.stab)}</div></div>`;
}

function slice(a,n){return a.slice(-n);}

/* ---- regime -> option-strategy suitability.
   Short-vol families: assignment DRIVEN BY MEASURED per-regime backtest stats
   (straddle proxy, flat pricing) with explicit thresholds shown in the evidence.
   Calendars / long vol: mechanics-based, labelled as not backtested here. ---- */
function tradeContext(c,reg){
  const sl=reg.slope,slt=reg.slope_txt,vix=reg.vix,vixp=reg.vix_pct,front=reg.front,
    r20=reg.ret20,R=reg.label,rvCalm=reg.rv_calm;
  const calm=R==='CALM CARRY'||R==='STEADY CARRY',
        stress=R==='STRESS / BACKW'||R==='VOL SHOCK';
  const contango=sl>=1.0,steep=sl>=1.08;
  const st=DATA.backtest.breakdowns["Straddle (ATM)"].regime.find(x=>x.regime===R);   // measured, this regime
  const F=[],A=[],C=[];
  const push=(a,name,desc,ev)=>a.push({name,desc,ev});
  const mEv=st?`measured ${R}: ${st.mean>=0?'+':''}${num(st.mean,2)}%/tr · win ${num(st.hit,0)}% · p5 ${num(st.p5,1)}% · worst ${num(st.worst,1)}% (n=${st.n})`:'no measured sample';
  const thin=st&&st.n<100?' · thin sample':'';
  const calmTail=calm&&st?' · note: largest historical losses started in calm regimes':'';

  // rv_composite drag: measured (2026-08-26) — in CALM/STEADY a positive rv10-rv63
  // precedes weaker short-vol P&L. Only these two regimes (TRANSITION flips sign).
  const rvcDrag=calm&&reg.rvc_accel;
  const rvcEv=rvcDrag?` · ⚠ RV accelerating (rv comp +${num(reg.rvc,2)}) — measured drag on short vol`:'';
  // Short vol, defined risk — thresholds: Favored mean≥0.3 & win≥65 · Caution mean<0 or win<60
  if(st){
    const ev=mEv+thin+calmTail+rvcEv;
    if(rvcDrag&&st.mean>=0.3&&st.hit>=65)push(A,'Iron Condors / Short Strangle','short vol — paid here, but RV accelerating (size down)',ev);
    else if(st.mean>=0.3&&st.hit>=65)push(F,'Iron Condors / Short Strangle','short vol — historically paid in this regime',ev);
    else if(st.mean<0||st.hit<60)push(C,'Iron Condors / Short Strangle','short vol — historically weak in this regime',ev);
    else push(A,'Iron Condors / Short Strangle','short vol — mixed record in this regime',ev);
  }
  // Short vol, undefined risk — stricter: also requires calm realized vol
  if(st){
    const ev=mEv+(rvCalm?' · RV accel low':' · RV accel elevated')+calmTail+rvcEv;
    if(st.mean>=0.5&&st.hit>=65&&rvCalm&&!rvcDrag)push(F,'Premium Selling (ATM)','undefined-risk short vol — paid & RV calm',ev);
    else if(st.mean<0||st.hit<60||!rvCalm||rvcDrag)push(C,'Premium Selling (ATM)','undefined risk — weak record or RV accelerating',ev);
    else push(A,'Premium Selling (ATM)','size down — mixed record',ev);
  }
  // Term-structure carry — measured: 9/30 ATM call calendar backtest, per current regime
  {const sc=DATA.backtest.breakdowns["Calendar 9/30 (ATM call)"].regime.find(x=>x.regime===R);
   const ev=sc?`measured 9/30 calendar, ${R}: ${sc.mean>=0?'+':''}${num(sc.mean,2)}%/tr · win ${num(sc.hit,0)}% · p5 ${num(sc.p5,1)}% (n=${sc.n})`+(sc.n<100?' · thin sample':'')+` · slope ${sl}`:'no measured sample';
   if(sc&&sc.mean>=0.1&&sc.hit>=55)push(F,'Calendars / Diagonals','term carry — historically paid in this regime',ev);
   else if(!sc||sc.mean<-0.05||sc.hit<45)push(C,'Calendars / Diagonals','term carry — historically weak in this regime',ev);
   else push(A,'Calendars / Diagonals','term carry — ≈ zero after costs in this regime',ev);}
  // Long vol — measured: buying the ATM straddle, per current regime. Insurance framing:
  // negative mean = cost of the hedge, not a defect; assignment reflects measured EV.
  {const sv2=DATA.backtest.breakdowns["Long Vol (ATM buy)"].regime.find(x=>x.regime===R);
   const ev=sv2?`measured long straddle, ${R}: ${sv2.mean>=0?'+':''}${num(sv2.mean,2)}%/tr · win ${num(sv2.hit,0)}% (n=${sv2.n})`+(sv2.n<100?' · thin sample':'')+` · VIX ${vix} (${ord(vixp)} pct)`:'no measured sample';
   if(sv2&&sv2.mean>=0)push(A,'Long Vol / Tail Hedge','hedge — measured EV non-negative here',ev);
   else push(C,'Long Vol / Tail Hedge',`hedge — insurance cost ≈ ${sv2?num(-sv2.mean,2):'?'}%/trade in this regime`,ev);}
  return {F,A,C};
}
function colHTML(items){return items.length? items.map(it=>
  `<div class="citem"><div class="name">${it.name} <span class="desc">— ${it.desc}</span></div><div class="ev">${it.ev}</div></div>`).join("")
  : '<div class="citem"><div class="desc">None from current evidence layer.</div></div>';}

function renderBacktest(){
  const bt=DATA.backtest,P=bt.params;
  const scols=["--blue","--amber","--green","--pink","--purple"], snames=Object.keys(bt.structures);
  const sk=state.skew?"skew":"flat", ga=state.gate?"gated":"all";
  const pick=n=>bt.structures[n][sk][ga];

  el("bt-title").textContent="▶ OPTION-STRUCTURE BACKTEST · 30 DTE · hold to expiry";
  el("bt-sub").innerHTML=`Every day: open the structure priced from the interpolated 30d IV via Black-Scholes (−${P.slip}% modeled cost), hold to expiry — ${P.dte} calendar days ≈ ${P.exp_td} trading sessions. Equity = non-overlapping trades over ${P.years} years, ${P.unit}.`;

  // toggle buttons
  el("bt-skew").innerHTML=`<button class="lb${!state.skew?" active":""}" data-tg="skew" data-v="0">off (ATM)</button><button class="lb${state.skew?" active":""}" data-tg="skew" data-v="1">approx</button>`;
  el("bt-gate").innerHTML=`<button class="lb${!state.gate?" active":""}" data-tg="gate" data-v="0">off</button><button class="lb${state.gate?" active":""}" data-tg="gate" data-v="1">CALM/STEADY</button>`;

  const be=snames.map((n,i)=>({name:n,color:scols[i],data:pick(n).eq}));
  el("bt-eq").innerHTML=lineChart(be,{dates:bt.eq_dates,zero:true});el("bt-leg").innerHTML=legend(be);

  el("bt-cmp-title").textContent=`▶ STRUCTURE COMPARISON · ${state.skew?"skew-adjusted":"ATM (flat)"} · ${state.gate?"CALM/STEADY entries only":"all entries"} · non-overlapping`;
  const metrics=[
    ["Mean P&L %/trade",s=>num(s.stats.mean,2),"Average per-trade P&L across all daily entries, in % of spot notional."],
    ["Win rate",s=>num(s.stats.hit,0)+"%","Share of daily-entry trades that ended positive."],
    ["Worst trade %",s=>num(s.stats.worst,1),"The single worst trade in the whole sample — the left tail made explicit."],
    ["5th-pctile trade %",s=>num(s.stats.p5,1),"5% of all trades were worse than this value."],
    ["Avg win / loss",s=>num(s.stats.avg_win,2)+" / "+num(s.stats.avg_loss,2),"Average size of winning vs losing trades — shows the asymmetry of the payoff."],
    ["Trades",s=>s.n,"Non-overlapping trades used for the equity curve (one position at a time)."],
    ["Total return %",s=>num(s.tot,1),"Sum of non-overlapping per-trade returns over the full period, % of spot notional."],
    ["Annualised % (simple, tot/yrs)",s=>num(s.ann,1),"Total return ÷ years — simple average, not compounded."],
    ["Max drawdown %",s=>num(s.mdd,1),"Deepest peak-to-trough decline of the non-overlapping equity curve."]];
  el("bt-cmp").querySelector("thead").innerHTML="<tr><th>Metric</th>"+snames.map((n,i)=>`<th><span style="color:${cv(scols[i])}">${n}</span></th>`).join("")+"</tr>";
  el("bt-cmp").querySelector("tbody").innerHTML=metrics.map(([lab,fn,tp])=>
    `<tr><td data-tip="${tp}">${lab}</td>`+snames.map(n=>`<td>${fn(pick(n))}</td>`).join("")+"</tr>").join("");

  // unified breakdown tables — one strategy selector drives both (flat pricing, all entries)
  if(!snames.includes(state.btStrat))state.btStrat=snames[0];
  el("bt-strat").innerHTML=snames.map((n,i)=>`<button class="lb${n===state.btStrat?" active":""}" data-si="${i}">${n}</button>`).join("");
  const bd=bt.breakdowns[state.btStrat];
  const cadence=state.btStrat.startsWith("Calendar")?`hold ${P.cal_front_td} sessions (front expiry, back leg marked with real then-current 21d IV)`:`hold ${P.exp_td} sessions`;
  el("bt-q-title").textContent=`▶ ${state.btStrat} · P&L BY ENTRY VRP-30 QUINTILE · flat, all daily entries, ${cadence}`;
  el("bt-reg-title").textContent=`▶ ${state.btStrat} · P&L BY ENTRY REGIME · flat, all daily entries, ${cadence}`;
  el("bt-q").querySelector("tbody").innerHTML=bd.quintile.map(b=>`<tr><td>Q${b.q}${b.q===1?" (cheap)":b.q===bd.quintile.length?" (rich)":""}</td><td>${num(b.sig,2)}</td><td>${num(b.pnl,2)}</td><td>${num(b.hit,0)}%</td></tr>`).join("");
  el("bt-reg").querySelector("tbody").innerHTML=bd.regime.map(b=>{const i=DATA.regime.order.indexOf(b.regime);return `<tr><td data-tip="${REGTIP[b.regime]||""}"><span style="color:${cv(DATA.regime.colors[i])}">■</span> ${b.regime}</td><td>${b.n}</td><td>${num(b.mean,2)}</td><td>${num(b.hit,0)}%</td><td>${num(b.p5,1)}</td><td>${num(b.worst,1)}</td></tr>`;}).join("");
  el("bt-note").innerHTML=`Prices are <b>reconstructed</b> with Black-Scholes from the interpolated ATM 30d implied vol (no free option-quote history) — spot ^GSPC, rate ${P.rate}, dividend q=${P.q}. <b>Skew</b> (toggle) is a parametric approximation: IV(K)=IV_atm·(1+slope·(−m)), put slope ${P.skew_dn} / call slope ${P.skew_up} of standardized moneyness — a typical SPX smirk, <b>not</b> real per-strike quotes. Strikes for the 16Δ/5Δ legs are placed off the ATM vol. Also <b>no real bid-ask</b> (flat ${P.slip}% cost), settlement at expiry from the actual SPX path, P&L in % of spot notional (not margin — real leveraged returns and risk are higher). Good for <b>relative</b> comparison, not exact fills.`;
}

function render(){
  const L=DATA.lookbacks[state.lb],reg=DATA.regime.current,ch=DATA.charts,N=L.sessions;
  el("t-spx").textContent=DATA.spx.toLocaleString("en-US",{minimumFractionDigits:2});
  el("t-iv30").textContent=DATA.iv30.toFixed(1)+"%";
  el("t-vrp30").textContent=(DATA.vrp30>=0?"+":"")+DATA.vrp30.toFixed(1)+"%";
  el("t-asof").textContent=DATA.asof;
  el("w-range").textContent=L.win_start+" – "+L.win_end; el("w-sess").textContent="("+N+" trading sessions)";
  // data-source status — computed from real per-ticker freshness, not hardcoded
  const nOk=DATA.n_ok,nSrc=DATA.sources.length,allOk=nOk===nSrc;
  const bd=el("t-badge");bd.textContent=allOk?"● REAL DATA":`● DATA ${nOk}/${nSrc}`;
  bd.style.color=cv(allOk?"--green":"--amber");bd.style.borderColor=cv(allOk?"--green-dim":"--amber");
  const lv=el("s-live");lv.textContent=allOk?"● REAL DATA ACTIVE":`● DEGRADED — ${nOk}/${nSrc} SOURCES`;
  lv.style.color=cv(allOk?"--green":"--amber");
  el("s-src").innerHTML=DATA.sources.map(s=>
    `<span class="${s.ok?"chk":"chx"}" title="last ${s.last}, lag ${s.lag}">${s.ok?"✓":"✗"}</span> ${s.name}${s.ok?"":" (stale "+s.lag+"d)"}`).join(" ");
  const sr=el("s-regime");sr.textContent=reg.label;sr.style.color=cv(reg.color);
  sr.setAttribute("data-tip",REGTIP[reg.label]||"");
  el("s-slope").textContent=`Term slope VIX3M/VIX ${reg.slope} (${reg.slope_txt}) · VIX ${reg.vix} (${ord(reg.vix_pct)} pct) · SPX ${reg.spx_state} · 20d ${reg.ret20>=0?"+":""}${reg.ret20}%`;

  /* OVERVIEW */
  const ranked=[...L.cards].sort((a,b)=>b.composite-a.composite);
  const bestD=ranked[0].dte,weakD=ranked[ranked.length-1].dte,rk=Object.fromEntries(ranked.map((c,i)=>[c.dte,i+1]));
  el("cards").innerHTML=L.cards.map((c,i)=>cardHTML(c,rk[c.dte],DATA.colors[i%DATA.colors.length],c.dte===bestD,c.dte===weakD)).join("");
  const top=L.top;
  const c30=L.cards.find(c=>c.dte===30)||L.cards[L.cards.length-1];
  const tc=tradeContext(c30,reg);
  el("ctx-sub").textContent=`This is not a trade instruction. ALL rows are assigned from MEASURED per-regime backtest stats: short vol via the straddle proxy (Favored mean≥0.3%/tr & win≥65, Caution mean<0 or win<60), calendars via the 9/30 ATM call calendar, long vol via buying the ATM straddle (negative mean = insurance cost).`;
  el("ctx-meta").innerHTML=`Current regime: <b>${reg.label}</b><br>Top DTE by lookback: <b>${top.dte} DTE</b><br>Regime sample: ${reg.reg_days} days · hit ${num(reg.reg_hit,0)}%`;
  el("col-fav").innerHTML=colHTML(tc.F); el("col-acc").innerHTML=colHTML(tc.A); el("col-cau").innerHTML=colHTML(tc.C);
  el("ctx-hist").innerHTML=`<b>${top.dte} DTE CURRENTLY RANKS HIGHEST.</b> Composite ${num(top.composite,1)}/100. Current VRP at the ${ord(top.vrp_pct)} percentile of trailing ${N}-session history. Hit rate ${num(top.hit,0)}%. Worst episode <span class="neg">${num(top.worst,1)}%</span>. Note: 7 DTE uses short-end interpolation (flat below the 9-day node) — treat it as indicative.`;

  /* VRP MATRIX */
  const mx=DATA.matrix;let mh="<tr><th class='k'>DTE \\ Month</th>"+mx.months.map(m=>`<th>${m}</th>`).join("")+"</tr>";
  mx.dtes.forEach(d=>{mh+=`<tr><td class="k">${d} DTE</td>`+mx.rows[String(d)].map(v=>`<td class="c" style="background:${heatColor(v,mx.vmin,mx.vmax)}">${v==null?"—":v.toFixed(1)}</td>`).join("")+"</tr>";});
  el("matrix-tbl").innerHTML=mh;

  /* CHARTS (respect lookback) */
  const dts=slice(ch.dates,N);
  el("c1-title").textContent=`▶ IMPLIED vs REALIZED VOL · 30-day · last ${N} sessions`;
  const s1=[{name:"IV30 (implied)",color:"--blue",data:slice(ch.iv30,N)},{name:"RV30 (realized, trailing)",color:"--amber",data:slice(ch.rv30,N)}];
  el("chart-ivrv").innerHTML=lineChart(s1,{dates:dts});el("c1-leg").innerHTML=legend(s1);
  el("c2-title").textContent=`▶ VRP BY DTE · nowcast IV−RV · last ${N} sessions`;
  const s2=[{name:"7",color:"--pink",data:slice(ch.vrp7,N)},{name:"14",color:"--yellow",data:slice(ch.vrp14,N)},{name:"21",color:"--green",data:slice(ch.vrp21,N)},{name:"30",color:"--blue",data:slice(ch.vrp30,N)}];
  el("chart-vrp").innerHTML=lineChart(s2,{dates:dts,zero:true});el("c2-leg").innerHTML=legend(s2);
  el("c3-title").textContent=`▶ VIX TERM STRUCTURE HISTORY · last ${N} sessions`;
  const s3=[{name:"VIX9D",color:"--pink",data:slice(ch.vix9d,N)},{name:"VIX (30d)",color:"--green",data:slice(ch.vix,N)},{name:"VIX3M",color:"--blue",data:slice(ch.vix3m,N)}];
  el("chart-term").innerHTML=lineChart(s3,{dates:dts});el("c3-leg").innerHTML=legend(s3);

  /* REGIME ANALYSIS */
  const rvcTxt=(reg.rvc>=0?"+":"")+reg.rvc+(reg.rvc_accel?" ⚠ accel":" (calm)");
  el("reg-grid").innerHTML=[["REGIME",reg.label],["VIX3M/VIX SLOPE",reg.slope+" ("+reg.slope_txt+")"],["VIX",reg.vix+"  ("+ord(reg.vix_pct)+" pct)"],["RV COMPOSITE (rv10−rv63)",rvcTxt],["SPX STATE",reg.spx_state],["20-DAY RETURN",(reg.ret20>=0?"+":"")+reg.ret20+"%"]].map(([k,v])=>`<div class="reg-cell" data-tip="${k.indexOf('RV COMPOSITE')===0?'rv10 − rv63 in vol points: >0 = realized vol accelerating vs its quarter norm. In CALM/STEADY this measurably drags short-vol P&L (Ticket 02).':''}"><div class="rk">${k}</div><div class="rv">${v}</div></div>`).join("");
  const per=DATA.regime.per_regime, perW=(DATA.regime.per_regime_lb||{})[state.lb]||{};
  el("reg-tbl-title").textContent=`▶ PER-REGIME EVIDENCE · forward VRP-30 · full history vs selected window (${N} sessions)`;
  el("reg-tbl").querySelector("tbody").innerHTML=DATA.regime.order.map((l,i)=>{const p=per[l],w=perW[l]||{};
    const wcell=w.ok?`${num(w.avg_vrp,2)}`:`<span class="dim" title="only ${w.n||0} obs, below the ${DATA.regime.min_reg_n}-obs minimum">insufficient (n=${w.n||0})</span>`;
    const whit=w.ok?`${num(w.hit,0)}%`:`—`;
    return `<tr><td data-tip="${REGTIP[l]||""}"><span style="color:${cv(DATA.regime.colors[i])}">■</span> ${l}</td><td>${p.days} (${num(p.share,1)}%)</td><td>${num(p.avg_vrp,2)}</td><td>${num(p.hit,0)}%</td><td>${wcell}</td><td>${whit}</td></tr>`;}).join("");
  el("reg-tbl-note").innerHTML=`"Avg VRP" = mean of IV − subsequent realized vol on days in that regime. <b>Full</b> = whole history (stable reference). <b>Window</b> = the selected lookback, gated at a hard ${DATA.regime.min_reg_n}-observation minimum — below it a cell reads <i>insufficient</i> rather than a misleading number (forward RV overlaps ~21×, so small windows have very few independent observations). Descriptive of history, not a forecast.`;

  /* VALIDATION */
  const v=DATA.validation;
  el("val-tbl").querySelector("tbody").innerHTML=v.buckets.map(b=>`<tr><td>Q${b.bucket}${b.bucket===1?" (cheapest)":b.bucket===v.buckets.length?" (richest)":""}</td><td>${b.n}</td><td>${num(b.sig,2)}</td><td>${num(b.out,2)}</td><td>${num(b.hit,0)}%</td></tr>`).join("");
  const mono=v.buckets[v.buckets.length-1].out>v.buckets[0].out;
  el("val-note").innerHTML=`Signal ranges from Q1 (IV cheapest vs realized) to Q5 (richest). ${mono?"Forward VRP rises across quintiles → the signal carries forward information.":"Pattern is not monotone → limited forward edge."}`;
  let eq=0,eqser=[],mdd=0,peak=0;slice(ch.vrp30_fwd,N).forEach(x=>{if(x!=null){eq+=x;}eqser.push(x==null?null:eq);peak=Math.max(peak,eq);mdd=Math.min(mdd,eq-peak);});
  el("eq-title").textContent=`▶ CUMULATIVE CAPTURED VRP-30 (equity proxy) · last ${N} sessions`;
  el("chart-eq").innerHTML=lineChart([{name:"cum VRP",color:"--green",data:eqser}],{dates:dts,zero:true,area:true});
  el("val-sum").innerHTML=[["Mean forward VRP",num(v.mean,2)],["Std",num(v.std,2)],["Info ratio (mean/std)",num(v.info_ratio,2)],["Hit rate",num(v.hit,0)+"%"],["Best / Worst",num(v.best,1)+" / "+num(v.worst,1)],["Sample size",v.n+" days"],["Window max drawdown",num(mdd,1)]].map(([k,x])=>`<div class="row"><span class="rk">${k}</span><span class="rv">${x}</span></div>`).join("");

  /* ANALOG ENGINE */
  const an=DATA.analog;
  el("an-title").textContent=`▶ ANALOG ENGINE · ${an.k} nearest historical days · full history (fixed — does not follow the lookback buttons)`;
  el("an-sub").textContent=`Window ${an.window_from} – ${an.window_to} · ${an.n_eligible.toLocaleString("en-US")} eligible days. Measured skill: fwd-RV Spearman ≈0.60 vs 0.55 baseline; return prediction ≈0 → context, not alpha.`;
  el("an-state").innerHTML=[["VIX",an.cur.vix],["VIX9D/VIX",an.cur.v9v],["VIX/VIX3M",an.cur.vv3m],["RV COMPOSITE",an.cur.rvc],["21d EFFICIENCY",an.cur.er21]].map(([k,x])=>`<div class="reg-cell"><div class="rk">${k}</div><div class="rv">${x}</div></div>`).join("");
  el("an-h").querySelector("tbody").innerHTML=an.per_h.map(h=>`<tr><td>${h.h}d</td><td>${h.matches}</td><td class="${h.avg>=0?'good':'bad'}">${num(h.avg,2)}%</td><td>${num(h.med,2)}%</td><td class="bad">${num(h.worst,1)}%</td><td class="good">${num(h.best,1)}%</td><td>${num(h.pos,0)}%</td></tr>`).join("");
  el("an-h-note").innerHTML=`Across the ${an.k} nearest analogs: average forward realized vol ${an.avg_fwd_rv}%, average 21-day VIX change ${an.avg_vix_chg>=0?"+":""}${an.avg_vix_chg}. Forward outcomes are historical analogs, <b>not forecasts</b>.`;
  el("an-top").querySelector("tbody").innerHTML=an.top.map(t=>`<tr><td>${t.date}</td><td><span style="color:${cv(REG_COLOR_JS(t.regime))}">${t.regime}</span></td><td>${t.dist}</td><td>${t.sim}%</td><td>${t.vix}</td><td class="${t.ret21>=0?'good':'bad'}">${num(t.ret21,1)}%</td><td>${num(t.fwd_rv,1)}%</td></tr>`).join("");

  /* BACKTEST */
  renderBacktest();

  /* TIMELINE ribbon */
  const codes=slice(ch.reg,N),rw=1200,rh=54,cw=rw/codes.length;
  let rib="";codes.forEach((c,i)=>{rib+=`<rect x="${(i*cw).toFixed(2)}" y="0" width="${(cw+0.5).toFixed(2)}" height="${rh-16}" fill="${cv(DATA.regime.colors[c])}"/>`;});
  let rxl="";for(let k=0;k<8;k++){const i=Math.round(k*(codes.length-1)/7);rxl+=`<text x="${(i*cw).toFixed(1)}" y="${rh-3}" fill="#4d5a6d" font-size="9">${dts[i].slice(0,7)}</text>`;}
  el("tl-title").textContent=`▶ REGIME TIMELINE · last ${N} sessions`;
  el("ribbon").innerHTML=`<svg viewBox="0 0 ${rw} ${rh}" class="chart" preserveAspectRatio="none" font-family="var(--mono)">${rib}${rxl}</svg>`;
  el("tl-leg").innerHTML=DATA.regime.order.map((l,i)=>`<span class="lg" data-tip="${REGTIP[l]||""}"><span class="sw" style="background:${cv(DATA.regime.colors[i])}"></span>${l}</span>`).join("");

  /* FORECAST */
  const T=DATA.regime.transition;
  let mh2="<tr><th class='k'>from \\ to</th>"+DATA.regime.order.map((l,i)=>`<th><span style="color:${cv(DATA.regime.colors[i])}">${l.split(" ")[0]}</span></th>`).join("")+"</tr>";
  T.forEach((row,i)=>{mh2+=`<tr><td class="k"><span style="color:${cv(DATA.regime.colors[i])}">■</span> ${DATA.regime.order[i]}</td>`+row.map((p,j)=>`<td class="c" style="background:rgba(61,220,132,${p==null?0:(p/100*0.8).toFixed(2)})">${p==null?"—":p}</td>`).join("")+"</tr>";});
  el("mtx").innerHTML=mh2;

  /* FORECAST (streak-adjusted persistence, 5-day) + calibration */
  const F=DATA.forecast;
  el("fc-head").textContent=`▶ REGIME FORECAST · next ${F.horizon} trading days`;
  el("fc-sub").textContent=`From today's regime (${reg.label}). Evidence-based next-regime probabilities — does not forecast SPX direction, VIX level, returns or trade profitability.`;
  const conf=F.top_prob, risk=conf>=60?"LOW":conf>=45?"MODERATE":"HIGH";
  el("fc-cards").innerHTML=[
    ["MOST LIKELY NEXT",`<span style="color:${cv(REG_COLOR_JS(F.top))}">${F.top}</span> · ${F.top_prob}%`],
    ["SECOND",`${F.second} · ${F.second_prob}%`],
    ["TRANSITION RISK",`${risk} (top ${F.top_prob}% vs 2nd ${F.second_prob}%)`]
  ].map(([k,v])=>`<div class="reg-cell"><div class="rk">${k}</div><div class="rv">${v}</div></div>`).join("");
  el("fc-title").textContent=`▶ NEXT-REGIME PROBABILITY · ${F.horizon} trading days out`;
  el("fc-bars").innerHTML=F.order.map((l,i)=>{const p=F.dist[i]||0;return `<div class="fcbar"><span class="fk" data-tip="${REGTIP[l]||""}"><span style="color:${cv(F.colors[i])}">■</span> ${l}</span><span class="track"><span class="fill" style="width:${p}%;background:${cv(F.colors[i])}"></span></span><span class="fv">${num(p,1)}%</span></div>`;}).join("");
  const cal=F.calib;
  el("fc-cal-title").textContent=`▶ FORECAST RELIABILITY · retroactive causal backtest · ${cal.n.toLocaleString("en-US")} daily forecasts since ${F.cal_from}`;
  el("fc-cal").querySelector("tbody").innerHTML=cal.buckets.map(b=>`<tr><td>${b.bucket}</td><td>${b.n}</td><td>${b.conf==null?"—":b.conf+"%"}</td><td>${b.acc==null?"—":b.acc+"%"}</td><td class="${b.err==null?"":(b.err<0?"bad":"good")}">${b.err==null?"—":(b.err>0?"+":"")+b.err+" pp"}</td></tr>`).join("");
  el("fc-cal-note").innerHTML=`Each day's forecast is re-derived from data up to that day, then marked HIT/MISS after ${F.horizon} trading days. This is a <b>simulated causal replay, not a live journal</b> — identical for everyone, regenerated each build. Overall: avg confidence ${cal.overall_conf}%, actual ${cal.overall_acc}% → <b class="${cal.bias_pp<0?'bad':'good'}">${cal.bias} (${cal.bias_pp>0?"+":""}${cal.bias_pp} pp)</b>. Quality ${cal.quality}/100. Bucket errors are shown, not hidden — that is what the curve is for.`;
  el("fc-jrn").querySelector("tbody").innerHTML=F.journal.map(j=>`<tr><td>${j.date}</td><td><span style="color:${cv(REG_COLOR_JS(j.regime))}">${j.regime}</span></td><td>${j.pred}</td><td>${j.prob}%</td><td>${j.actual||"—"}</td><td class="${j.status==='HIT'?'good':j.status==='MISS'?'bad':''}">${j.status}</td></tr>`).join("");

  /* FORWARD TEST — genuine out-of-sample, grows forward only */
  const FW=DATA.forward;
  el("fw-title").textContent=`▶ FORWARD TEST · live, out-of-sample · started ${FW.since}`;
  el("fw-sub").innerHTML=`The honest test: each build logs that day's forecast and marks it HIT/MISS ${FW.horizon} trading days later. Unlike the retroactive curve above, this <b>cannot be curve-fit</b> — it only grows forward. ${FW.n_resolved===0?"<b>No forecasts have resolved yet — building up.</b>":""}`;
  el("fw-cards").innerHTML=[
    ["RESOLVED",`${FW.n_resolved}`],
    ["HIT RATE",FW.hit_rate==null?"— (building up)":`${FW.hit_rate}% (${FW.hits}/${FW.n_resolved})`],
    ["PENDING",`${FW.n_pending}`],
  ].map(([k,v])=>`<div class="reg-cell"><div class="rk">${k}</div><div class="rv">${v}</div></div>`).join("");
  el("fw-tbl").querySelector("tbody").innerHTML=FW.recent.length?FW.recent.map(j=>`<tr><td>${j.date}</td><td><span style="color:${cv(REG_COLOR_JS(j.regime))}">${j.regime}</span></td><td>${j.pred}</td><td>${j.prob}%</td><td>${j.resolved_on||"—"}</td><td>${j.actual||"—"}</td><td class="${j.status==='HIT'?'good':j.status==='MISS'?'bad':''}">${j.status}</td></tr>`).join(""):`<tr><td colspan="7" class="dim">No entries yet — the first build logs today's forecast.</td></tr>`;

  /* SETTINGS */
  el("set-src").innerHTML=DATA.sources.map(s=>
    `<div class="row"><span class="rk">${s.name}</span><span class="rv">${s.ticker} · last ${s.last} · <span class="${s.ok?"chk":"chx"}">${s.ok?"✓ fresh":"✗ stale ("+s.lag+" sessions)"}</span>${s.filled?` · <span class="chx" title="interior gaps carried forward (ffill)">⚠ ${s.filled} filled</span>`:""}</span></div>`).join("");
  el("set-par").innerHTML=[["History start",DATA.first],["As of",DATA.asof],["Total sessions",DATA.n_total],["DTE horizons",DATA.dtes.join(" / ")],["Lookbacks",Object.values(DATA.lookbacks).map(l=>l.name).join(" · ")],["Term nodes (days)",Object.entries(DATA.node_days).map(([k,v])=>k+"="+v).join(" · ")],["Forecast horizon",DATA.forecast.horizon+" trading days (streak-adjusted persistence)"]].map(([k,x])=>`<div class="row"><span class="rk">${k}</span><span class="rv">${x}</span></div>`).join("");
  el("set-w").innerHTML=Object.entries(DATA.weights).map(([k,x])=>`<div class="row"><span class="rk">${k.toUpperCase()}</span><span class="rv">${x}%</span></div>`).join("");
  el("set-subs").innerHTML=`<b style="color:var(--ink-dim)">RICH</b> — richness: percentile of current VRP within this DTE's own trailing history (rich IV vs its norm).<br><b style="color:var(--ink-dim)">CARRY</b> — carry per day: VRP ÷ calendar days, ranked across the DTE set (which tenor pays most premium per day now, relative).<br><b style="color:var(--ink-dim)">SAFETY</b> — merged reliability (info ratio = mean ÷ std of forward VRP) + tail-safety (1 − |CVaR₅|). These two were measured to be redundant (Spearman +0.89 on the real definitions, 2026-08-26), so they are one axis to avoid double-counting.<br><b style="color:var(--ink-dim)">PATH</b> — path-safety: trailing mean Max Adverse Excursion of the SPX price path over the forward window, inverted percentile within its own window history. A genuine <i>path</i> axis (the price journey), distinct from SAFETY's VRP-outcome tail.<br><b style="color:var(--ink-dim)">STAB</b> — stability: steadiness of recent VRP vs its window dispersion.<br>Five decorrelated axes — level, cross-sectional carry, outcome safety, path safety, steadiness — so the composite doesn't double-count. <b>Note:</b> the composite ranks premium <i>quality/safety</i>, not expected P&L — measured (2026-08-26) low-vol conditions precede weaker short-vol returns (complacency).`;
  el("set-rules").innerHTML=`VIX≥40 → <code>VOL SHOCK</code> · VIX≥28 or slope&lt;0.97 → <code>STRESS / BACKW</code> · VIX≥20 → <code>TRANSITION</code> · slope≥1.05 &amp; uptrend &amp; VIX&lt;17 → <code>CALM CARRY</code> · slope≥1.0 &amp; uptrend → <code>STEADY CARRY</code> · else → <code>NEUTRAL RANGE</code>.<br>slope = VIX3M / VIX (&gt;1 contango). uptrend = SPX &gt; 200-day moving average.`;

  /* HELP — live numbers injected from the dataset */
  {const S_=DATA.backtest.structures,vv=DATA.validation;
   const wsd=S_["Straddle (ATM)"].flat.all.stats.worst;
   const wic=S_["Iron Condor (16/5Δ)"].flat.all.stats.worst;
   const lv=S_["Long Vol (ATM buy)"].flat.all.stats.mean;
   const ca=S_["Calendar 9/30 (ATM call)"].flat.all.stats.mean;
   const nr=DATA.backtest.breakdowns["Straddle (ATM)"].regime.find(x=>x.regime==="NEUTRAL RANGE");
   el("h-n").textContent=vv.n.toLocaleString("en-US");
   el("h-vrp").textContent=num(vv.mean,2);
   el("h-hitv").textContent=num(vv.hit,0);
   el("h-nr").textContent=nr?`${num(nr.mean,2)}%/trade · win ${num(nr.hit,0)}%`:"—";
   el("h-wsd").textContent=num(wsd,1);el("h-wsd2").textContent=num(wsd,1);
   el("h-wic").textContent=num(wic,1);
   el("h-lv").textContent=num(Math.abs(lv),2);
   el("h-cal").textContent=num(ca,2);
   el("h-size").textContent=wsd?Math.round(2/Math.abs(wsd)*100):"—";}

  const wt=DATA.weights;
  el("foot").innerHTML=`<b>Composite over 5 independent axes:</b> RICHNESS ${wt.rich}% · CARRY/day ${wt.carry}% · SAFETY ${wt.safety}% · PATH ${wt.path}% · STABILITY ${wt.stab}%. &nbsp;·&nbsp; <b>VRP</b> = interpolated implied vol − horizon-matched realized vol, vol points. Percentile/safety/path/validation use forward RV or the forward price path; the nowcast uses trailing RV. Free public data only (yfinance). Research tool — evidence, not advice. Not investment advice.`;
}

el("tabs").addEventListener("click",e=>{const b=e.target.closest(".tab");if(!b)return;state.tab=b.dataset.tab;
  document.querySelectorAll(".tab").forEach(t=>t.classList.toggle("active",t===b));
  document.querySelectorAll(".panel").forEach(p=>p.classList.toggle("active",p.dataset.panel===state.tab));});
const lbs=el("lbs");
lbs.innerHTML=Object.entries(DATA.lookbacks).map(([k,v])=>`<button class="lb${k===state.lb?" active":""}" data-lb="${k}">${v.name}</button>`).join("");
lbs.addEventListener("click",e=>{const b=e.target.closest(".lb");if(!b)return;state.lb=b.dataset.lb;
  document.querySelectorAll("#lbs .lb").forEach(x=>x.classList.toggle("active",x===b));render();});
el("bt-ctrls").addEventListener("click",e=>{const b=e.target.closest("[data-tg]");if(!b)return;
  state[b.dataset.tg]=b.dataset.v==="1";renderBacktest();});
el("bt-strat-box").addEventListener("click",e=>{const b=e.target.closest("[data-si]");if(!b)return;
  state.btStrat=Object.keys(DATA.backtest.structures)[+b.dataset.si];renderBacktest();});

/* tooltip engine */
const tipEl=el("tip");
document.addEventListener("mouseover",e=>{
  const t=e.target.closest("[data-tip]");
  if(!t||!t.dataset.tip){tipEl.style.display="none";return;}
  tipEl.textContent=t.dataset.tip;tipEl.style.display="block";});
document.addEventListener("mousemove",e=>{
  if(tipEl.style.display==="none")return;
  const w=tipEl.offsetWidth,h=tipEl.offsetHeight;
  let x=e.clientX+14,y=e.clientY+18;
  if(x+w>innerWidth-8)x=e.clientX-w-12;
  if(y+h>innerHeight-8)y=e.clientY-h-12;
  tipEl.style.left=x+"px";tipEl.style.top=y+"px";});

render();

/* stale-data self-check: the page compares its own data date to the viewer's clock.
   Works even when the build is frozen (a source outage stops the rebuild) — the live
   page flags itself as stale without a new deploy. Trading-day aware: weekends don't count. */
(function(){
  try{
    const asof=new Date(DATA.asof_iso+"T00:00:00");
    const now=new Date();
    let bdays=0; const d=new Date(asof);
    while(d<now){d.setDate(d.getDate()+1);const wd=d.getDay();if(wd!==0&&wd!==6)bdays++;}
    const THRESH=3;                       // >3 business days stale = source likely disrupted
    if(bdays>THRESH){
      el("stale-msg").innerHTML=`⚠ <b>DATA MAY BE STALE</b> — last successful update <b>${DATA.asof}</b> `+
        `(${bdays} business days ago). The free data source (Yahoo/Cboe) is likely disrupted; `+
        `every figure reflects that date, not today. The build refreshes automatically once the source recovers.`;
      el("stale-overlay").style.display="block";
      document.body.style.paddingTop=el("stale-overlay").offsetHeight+"px";
    }
    el("stale-x").addEventListener("click",()=>{el("stale-overlay").style.display="none";document.body.style.paddingTop="0";});
  }catch(e){}
})();
</script>
"""

if __name__ == "__main__":
    main()
