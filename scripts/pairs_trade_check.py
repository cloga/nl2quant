"""Check whether two A-share stocks are suitable for pairs trading.

Default pair:
- 300750.SZ (宁德时代)
- 002466.SZ (天齐锂业)

Outputs:
- data/pairs_<A>_<B>_<YYYYMMDD>.csv

Usage (PowerShell):
    python scripts/pairs_trade_check.py
    python scripts/pairs_trade_check.py --a 300750.SZ --b 002466.SZ --years 5

Notes:
- Uses Tushare Pro daily prices.
- Requires TUSHARE_TOKEN in .env.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import tushare as ts
from dotenv import load_dotenv
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller, coint


@dataclass
class PairResult:
    a: str
    b: str
    start: str
    end: str
    n_days: int
    beta: float
    coint_pvalue: float
    adf_pvalue_spread: float
    spread_half_life_days: float | None
    roll_window: int
    entry_z: float
    exit_z: float
    trades: int
    win_rate: float | None
    avg_holding_days: float | None
    sharpe_daily: float | None
    max_drawdown: float | None


def _get_pro():
    load_dotenv()
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("TUSHARE_TOKEN not set")
    ts.set_token(token)
    return ts.pro_api()


def fetch_daily_close(pro, ts_code: str, start: str, end: str) -> pd.Series:
    df = pro.daily(ts_code=ts_code, start_date=start, end_date=end)
    if df is None or df.empty:
        raise RuntimeError(f"No daily data for {ts_code}")
    df = df[["trade_date", "close"]].copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df = df.sort_values("trade_date")
    return df.set_index("trade_date")["close"].rename(ts_code)


def estimate_half_life(spread: pd.Series) -> float | None:
    # AR(1): dS_t = a + b*S_{t-1} + e
    s = spread.dropna()
    if len(s) < 100:
        return None
    lagged = s.shift(1).dropna()
    delta = s.diff().dropna()
    aligned = pd.concat([delta, lagged], axis=1).dropna()
    if aligned.empty:
        return None
    y = aligned.iloc[:, 0].values
    x = add_constant(aligned.iloc[:, 1].values)
    b = OLS(y, x).fit().params[1]
    if b >= 0:
        return None
    half_life = float(-np.log(2) / b)
    if not np.isfinite(half_life) or half_life <= 0:
        return None
    return half_life


def backtest_zscore(spread: pd.Series, roll: int, entry_z: float, exit_z: float) -> tuple[pd.Series, dict]:
    s = spread.copy()
    mu = s.rolling(roll).mean()
    sd = s.rolling(roll).std(ddof=0)
    z = (s - mu) / sd

    position = pd.Series(0, index=s.index, dtype=int)
    entry_price = None
    entry_idx = None
    trade_pnls = []
    holding_days = []

    pos = 0
    for i, dt in enumerate(s.index):
        if not np.isfinite(z.loc[dt]) or not np.isfinite(s.loc[dt]):
            position.loc[dt] = pos
            continue

        if pos == 0:
            if z.loc[dt] <= -entry_z:
                pos = 1  # long spread
                entry_price = s.loc[dt]
                entry_idx = dt
            elif z.loc[dt] >= entry_z:
                pos = -1  # short spread
                entry_price = s.loc[dt]
                entry_idx = dt
        else:
            if abs(z.loc[dt]) <= exit_z:
                # close
                pnl = (s.loc[dt] - entry_price) * pos
                trade_pnls.append(pnl)
                holding_days.append((dt - entry_idx).days)
                pos = 0
                entry_price = None
                entry_idx = None

        position.loc[dt] = pos

    # daily PnL proxy from spread changes while holding
    spread_ret = s.diff().fillna(0.0)
    pnl_series = (position.shift(1).fillna(0) * spread_ret).rename("pnl")

    trade_pnls = np.asarray(trade_pnls, dtype=float)
    holding_days = np.asarray(holding_days, dtype=float)

    stats: dict = {
        "trades": int(len(trade_pnls)),
        "win_rate": float((trade_pnls > 0).mean()) if len(trade_pnls) else None,
        "avg_holding_days": float(np.mean(holding_days)) if len(holding_days) else None,
    }

    daily = pnl_series
    if daily.std(ddof=0) > 0:
        stats["sharpe_daily"] = float(daily.mean() / daily.std(ddof=0) * np.sqrt(252))
    else:
        stats["sharpe_daily"] = None

    # equity curve & max drawdown (in pnl units)
    equity = daily.cumsum()
    running_max = equity.cummax()
    dd = equity - running_max
    stats["max_drawdown"] = float(dd.min()) if len(dd) else None

    out = pd.DataFrame({
        "spread": s,
        "z": z,
        "position": position,
        "pnl": pnl_series,
        "equity": equity,
    })
    return out, stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", default="300750.SZ", help="Stock A ts_code")
    parser.add_argument("--b", default="002466.SZ", help="Stock B ts_code")
    parser.add_argument("--years", type=int, default=5, help="Lookback years")
    parser.add_argument("--roll", type=int, default=60, help="Rolling window for z-score")
    parser.add_argument("--entry-z", type=float, default=2.0, help="Entry threshold")
    parser.add_argument("--exit-z", type=float, default=0.5, help="Exit threshold")
    args = parser.parse_args()

    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=int(args.years * 365.25))
    start = start_dt.strftime("%Y%m%d")
    end = end_dt.strftime("%Y%m%d")

    pro = _get_pro()
    a_close = fetch_daily_close(pro, args.a, start, end)
    b_close = fetch_daily_close(pro, args.b, start, end)

    df = pd.concat([a_close, b_close], axis=1).dropna().sort_index()
    if len(df) < 300:
        raise RuntimeError(f"Not enough overlap data: {len(df)} rows")

    loga = np.log(df[args.a])
    logb = np.log(df[args.b])

    beta = float(OLS(loga.values, add_constant(logb.values)).fit().params[1])
    spread = (loga - beta * logb).rename("spread")

    # tests
    coint_stat, coint_p, _ = coint(loga, logb)
    adf_p = float(adfuller(spread.dropna().values, autolag="AIC")[1])

    half_life = estimate_half_life(spread)

    bt_df, bt_stats = backtest_zscore(spread, roll=args.roll, entry_z=args.entry_z, exit_z=args.exit_z)

    # export
    outdir = Path("data")
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"pairs_{args.a.replace('.', '')}_{args.b.replace('.', '')}_{datetime.now().strftime('%Y%m%d')}.csv"
    export_df = pd.concat([df, loga.rename("log_a"), logb.rename("log_b"), bt_df], axis=1)
    export_df.to_csv(outpath, index=True, encoding="utf-8")

    res = PairResult(
        a=args.a,
        b=args.b,
        start=start,
        end=end,
        n_days=int(len(df)),
        beta=beta,
        coint_pvalue=float(coint_p),
        adf_pvalue_spread=adf_p,
        spread_half_life_days=half_life,
        roll_window=int(args.roll),
        entry_z=float(args.entry_z),
        exit_z=float(args.exit_z),
        trades=int(bt_stats["trades"]),
        win_rate=bt_stats["win_rate"],
        avg_holding_days=bt_stats["avg_holding_days"],
        sharpe_daily=bt_stats["sharpe_daily"],
        max_drawdown=bt_stats["max_drawdown"],
    )

    print("=== Pair Trading Check ===")
    print(f"Pair: {res.a} vs {res.b}")
    print(f"Window: {res.start} .. {res.end} | overlap days: {res.n_days}")
    print(f"Hedge ratio beta (log): {res.beta:.4f}")
    print(f"Cointegration p-value: {res.coint_pvalue:.4f} (<=0.05 preferred)")
    print(f"ADF p-value on spread: {res.adf_pvalue_spread:.4f} (<=0.05 preferred)")
    print(f"Half-life (days): {res.spread_half_life_days if res.spread_half_life_days is not None else 'N/A'}")
    print(f"Backtest: roll={res.roll_window} entry_z={res.entry_z} exit_z={res.exit_z}")
    print(f"Trades: {res.trades} | Win rate: {res.win_rate} | Avg hold days: {res.avg_holding_days}")
    print(f"Sharpe (pnl proxy): {res.sharpe_daily} | MaxDD (pnl units): {res.max_drawdown}")
    print(f"Saved: {outpath}")

    # Heuristic recommendation
    ok = (res.coint_pvalue <= 0.05) and (res.adf_pvalue_spread <= 0.05) and (res.trades >= 10)
    print("Recommendation:", "Suitable" if ok else "Not ideal")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
