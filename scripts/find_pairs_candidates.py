"""Scan A-share universe to find candidate pairs-trading symbols.

This script is designed to be *practical* under Tushare rate limits:

Typical usage (PowerShell):

If you already have a cache, rerun with:

Outputs:

Notes:
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import tushare as ts
from dotenv import load_dotenv
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller, coint


@dataclass(frozen=True)
class ScanConfig:
    years: int
    start: str
    end: str
    max_api_per_min: int
    retry_sleep_sec: int
    max_retries: int
    min_overlap_days: int
    min_industry_size: int
    per_industry_top_corr_pairs: int
    coint_pvalue_max: float
    adf_pvalue_max: float
    half_life_max_days: float
    roll_window: int
    entry_z: float
    exit_z: float


class SimpleRateLimiter:
    """A simple fixed-window limiter: up to N calls per interval seconds."""

    def __init__(self, limit: int, interval: float = 60.0):
        self.limit = int(limit)
        self.interval = float(interval)
        self._window_start = time.monotonic()
        self._count = 0

    def acquire(self) -> None:
        now = time.monotonic()
        elapsed = now - self._window_start
        if elapsed >= self.interval:
            self._window_start = now
            self._count = 0
        if self._count >= self.limit:
            sleep_s = self.interval - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)
            self._window_start = time.monotonic()
            self._count = 0
        self._count += 1


def _get_pro():
    load_dotenv()
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("TUSHARE_TOKEN not set")
    ts.set_token(token)
    return ts.pro_api()


def _ts_call(callable_, limiter: SimpleRateLimiter, *, retries: int, retry_sleep_sec: int, **kwargs):
    last_err = None
    for attempt in range(retries + 1):
        try:
            limiter.acquire()
            return callable_(**kwargs)
        except Exception as e:  # noqa: BLE001
            last_err = e
            # Tushare often embeds rate-limit messages in the exception string
            msg = str(e)
            if "最多" in msg or "500" in msg or "频率" in msg or "access" in msg.lower():
                time.sleep(max(5, retry_sleep_sec))
            else:
                time.sleep(1)
    raise last_err


def get_trade_dates(pro, limiter: SimpleRateLimiter, start: str, end: str) -> list[str]:
    df = _ts_call(
        pro.trade_cal,
        limiter,
        retries=3,
        retry_sleep_sec=5,
        exchange="SSE",
        start_date=start,
        end_date=end,
        is_open=1,
        fields="cal_date,is_open",
    )
    if df is None or df.empty:
        raise RuntimeError("trade_cal returned empty")
    dates = df["cal_date"].tolist()
    return [str(d) for d in dates]


def get_stock_basic(pro, limiter: SimpleRateLimiter) -> pd.DataFrame:
    df = _ts_call(
        pro.stock_basic,
        limiter,
        retries=3,
        retry_sleep_sec=5,
        exchange="",
        list_status="L",
        fields="ts_code,symbol,name,area,industry,market,list_date,is_hs",
    )
    if df is None or df.empty:
        raise RuntimeError("stock_basic returned empty")
    # Keep only A-share style codes (.*.SZ/.SH/.BJ)
    df = df[df["ts_code"].str.endswith((".SZ", ".SH", ".BJ"))].copy()
    return df


def _cache_paths(start: str, end: str) -> tuple[Path, Path]:
    cache_dir = Path("data") / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    gz_path = cache_dir / f"daily_close_{start}_{end}.csv.gz"
    state_path = cache_dir / f"daily_close_{start}_{end}.state.json"
    return gz_path, state_path


def _read_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_state(state_path: Path, state: dict) -> None:
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def download_daily_close_cache(
    pro,
    limiter: SimpleRateLimiter,
    trade_dates: Iterable[str],
    gz_path: Path,
    state_path: Path,
    *,
    resume: bool,
    max_retries: int,
    retry_sleep_sec: int,
) -> None:
    state = _read_state(state_path) if resume else {}
    last_done = state.get("last_trade_date") if resume else None
    started = False if last_done else True

    # Ensure gzip file exists with header.
    if not gz_path.exists() or not resume:
        if gz_path.exists() and not resume:
            gz_path.unlink()
        with gzip.open(gz_path, mode="wt", encoding="utf-8", newline="") as f:
            f.write("trade_date,ts_code,close\n")

    for td in trade_dates:
        if not started:
            if td == last_done:
                started = True
            continue

        df = _ts_call(
            pro.daily,
            limiter,
            retries=max_retries,
            retry_sleep_sec=retry_sleep_sec,
            trade_date=td,
            fields="trade_date,ts_code,close",
        )
        if df is None or df.empty:
            # still mark progress; some days might be empty due to upstream issues
            state["last_trade_date"] = td
            _write_state(state_path, state)
            continue

        # Append as csv lines (no header)
        df = df.dropna(subset=["close"])
        with gzip.open(gz_path, mode="at", encoding="utf-8", newline="") as f:
            df.to_csv(f, index=False, header=False)

        state["last_trade_date"] = td
        _write_state(state_path, state)


def load_close_long(gz_path: Path, universe: set[str]) -> pd.DataFrame:
    df = pd.read_csv(gz_path, compression="gzip", dtype={"trade_date": str, "ts_code": str})
    df = df[df["ts_code"].isin(universe)].copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df = df.dropna(subset=["close"]).sort_values(["ts_code", "trade_date"])
    return df


def compute_log_returns(df_long: pd.DataFrame) -> pd.DataFrame:
    # df_long: columns trade_date, ts_code, close
    df_long = df_long.copy()
    df_long["log_close"] = np.log(df_long["close"].astype(float))
    df_long["log_ret"] = df_long.groupby("ts_code", sort=False)["log_close"].diff()
    df_long = df_long.dropna(subset=["log_ret"])
    return df_long[["trade_date", "ts_code", "close", "log_close", "log_ret"]]


def _top_corr_pairs_for_group(returns_wide: pd.DataFrame, top_k: int) -> list[tuple[str, str, float]]:
    # returns_wide: index dates, columns ts_code
    corr = returns_wide.corr(min_periods=max(30, int(returns_wide.shape[0] * 0.6)))
    codes = corr.columns.to_list()
    pairs: list[tuple[str, str, float]] = []
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            v = corr.iat[i, j]
            if np.isfinite(v):
                pairs.append((codes[i], codes[j], float(v)))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_k]


def estimate_half_life(spread: pd.Series) -> float | None:
    s = spread.dropna()
    if len(s) < 200:
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
    hl = float(-np.log(2) / b)
    if not np.isfinite(hl) or hl <= 0:
        return None
    return hl


def backtest_zscore(spread: pd.Series, roll: int, entry_z: float, exit_z: float) -> dict:
    s = spread.dropna().copy()
    if len(s) < max(roll * 3, 200):
        return {"trades": 0, "sharpe": None, "win_rate": None, "avg_hold_days": None}

    mu = s.rolling(roll).mean()
    sd = s.rolling(roll).std(ddof=0)
    z = (s - mu) / sd

    pos = 0
    entry_price = None
    entry_dt = None
    trade_pnls: list[float] = []
    holds: list[int] = []

    for dt in s.index:
        if not np.isfinite(z.loc[dt]):
            continue
        if pos == 0:
            if z.loc[dt] <= -entry_z:
                pos = 1
                entry_price = float(s.loc[dt])
                entry_dt = dt
            elif z.loc[dt] >= entry_z:
                pos = -1
                entry_price = float(s.loc[dt])
                entry_dt = dt
        else:
            if abs(z.loc[dt]) <= exit_z:
                pnl = (float(s.loc[dt]) - entry_price) * pos
                trade_pnls.append(pnl)
                holds.append(int((dt - entry_dt).days))
                pos = 0
                entry_price = None
                entry_dt = None

    # daily pnl proxy
    pnl_daily = (np.sign(pos) * 0.0)  # unused; keep shape simple
    spread_diff = s.diff().fillna(0.0)
    # reconstruct positions for daily pnl
    position = pd.Series(0, index=s.index, dtype=int)
    pos = 0
    for dt in s.index:
        if not np.isfinite(z.loc[dt]):
            position.loc[dt] = pos
            continue
        if pos == 0:
            if z.loc[dt] <= -entry_z:
                pos = 1
            elif z.loc[dt] >= entry_z:
                pos = -1
        else:
            if abs(z.loc[dt]) <= exit_z:
                pos = 0
        position.loc[dt] = pos

    daily = (position.shift(1).fillna(0) * spread_diff).astype(float)
    sharpe = None
    if daily.std(ddof=0) > 0:
        sharpe = float(daily.mean() / daily.std(ddof=0) * np.sqrt(252))

    arr = np.asarray(trade_pnls, dtype=float)
    win_rate = float((arr > 0).mean()) if len(arr) else None
    avg_hold = float(np.mean(holds)) if len(holds) else None

    return {"trades": int(len(arr)), "sharpe": sharpe, "win_rate": win_rate, "avg_hold_days": avg_hold}


def evaluate_pair(
    prices: pd.DataFrame,
    a: str,
    b: str,
    *,
    min_overlap_days: int,
    roll_window: int,
    entry_z: float,
    exit_z: float,
) -> dict | None:
    sub = prices[[a, b]].dropna().sort_index()
    if len(sub) < min_overlap_days:
        return None

    loga = np.log(sub[a].astype(float))
    logb = np.log(sub[b].astype(float))

    beta = float(OLS(loga.values, add_constant(logb.values)).fit().params[1])
    spread = (loga - beta * logb)

    try:
        _, coint_p, _ = coint(loga, logb)
    except Exception:
        coint_p = np.nan

    try:
        adf_p = float(adfuller(spread.dropna().values, autolag="AIC")[1])
    except Exception:
        adf_p = np.nan

    hl = estimate_half_life(spread)
    bt = backtest_zscore(spread, roll=roll_window, entry_z=entry_z, exit_z=exit_z)

    return {
        "a": a,
        "b": b,
        "beta": beta,
        "coint_p": float(coint_p) if np.isfinite(coint_p) else None,
        "adf_p": float(adf_p) if np.isfinite(adf_p) else None,
        "half_life": float(hl) if hl is not None else None,
        "trades": bt["trades"],
        "sharpe": bt["sharpe"],
        "win_rate": bt["win_rate"],
        "avg_hold_days": bt["avg_hold_days"],
        "overlap_days": int(len(sub)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=2, help="Lookback years")
    parser.add_argument("--max-api-per-min", type=int, default=350, help="Conservative Tushare call limit")
    parser.add_argument("--from-cache", action="store_true", help="Do not call Tushare daily; use cached gzip")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume; overwrite cache")
    parser.add_argument("--top", type=int, default=100, help="Top N pairs to output")

    parser.add_argument(
        "--min-overlap-days",
        type=int,
        default=0,
        help="Minimum overlapping trading days required (0 = auto based on years)",
    )
    parser.add_argument("--min-industry-size", type=int, default=15)
    parser.add_argument("--per-industry-top-corr-pairs", type=int, default=60)

    parser.add_argument("--coint-p-max", type=float, default=0.05)
    parser.add_argument("--adf-p-max", type=float, default=0.05)
    parser.add_argument("--half-life-max", type=float, default=120)

    parser.add_argument("--max-eval", type=int, default=0, help="Override number of candidate pairs to run tests on")

    parser.add_argument(
        "--loose",
        action="store_true",
        help="Use looser filters (coint/adf<=0.2, half-life<=250) to get candidates for manual review",
    )

    parser.add_argument("--roll", type=int, default=60)
    parser.add_argument("--entry-z", type=float, default=2.0)
    parser.add_argument("--exit-z", type=float, default=0.5)

    args = parser.parse_args()

    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=int(args.years * 365.25))
    start = start_dt.strftime("%Y%m%d")
    end = end_dt.strftime("%Y%m%d")

    if args.loose:
        coint_p_max = 0.2
        adf_p_max = 0.2
        half_life_max = 250.0
    else:
        coint_p_max = float(args.coint_p_max)
        adf_p_max = float(args.adf_p_max)
        half_life_max = float(args.half_life_max)

    min_overlap_days = int(args.min_overlap_days)
    if min_overlap_days <= 0:
        # Roughly require ~70% of expected trading days in the lookback window.
        min_overlap_days = max(120, int(252 * args.years * 0.7))

    cfg = ScanConfig(
        years=args.years,
        start=start,
        end=end,
        max_api_per_min=int(args.max_api_per_min),
        retry_sleep_sec=20,
        max_retries=4,
        min_overlap_days=min_overlap_days,
        min_industry_size=int(args.min_industry_size),
        per_industry_top_corr_pairs=int(args.per_industry_top_corr_pairs),
        coint_pvalue_max=coint_p_max,
        adf_pvalue_max=adf_p_max,
        half_life_max_days=half_life_max,
        roll_window=int(args.roll),
        entry_z=float(args.entry_z),
        exit_z=float(args.exit_z),
    )

    pro = _get_pro()
    limiter = SimpleRateLimiter(cfg.max_api_per_min, 60.0)

    stock_basic = get_stock_basic(pro, limiter)
    universe = set(stock_basic["ts_code"].tolist())

    gz_path, state_path = _cache_paths(cfg.start, cfg.end)

    if not args.from_cache:
        trade_dates = get_trade_dates(pro, limiter, cfg.start, cfg.end)
        download_daily_close_cache(
            pro,
            limiter,
            trade_dates,
            gz_path,
            state_path,
            resume=not args.no_resume,
            max_retries=cfg.max_retries,
            retry_sleep_sec=cfg.retry_sleep_sec,
        )

    if not gz_path.exists():
        raise RuntimeError(f"Cache missing: {gz_path}")

    df_long = load_close_long(gz_path, universe)
    df_ret = compute_log_returns(df_long)

    # Build a prices pivot only for candidates evaluation later; keeping full wide can be heavy.
    # We'll build per-industry pivots from df_long.

    # Candidate generation: within each industry, pick top correlated return pairs.
    candidates: list[tuple[str, str, float, str]] = []
    for industry, group in stock_basic.groupby("industry", dropna=False):
        industry = industry if isinstance(industry, str) and industry else "(unknown)"
        codes = group["ts_code"].tolist()
        if len(codes) < cfg.min_industry_size:
            continue

        sub = df_ret[df_ret["ts_code"].isin(codes)]
        if sub.empty:
            continue

        wide = sub.pivot_table(index="trade_date", columns="ts_code", values="log_ret", aggfunc="mean")
        # drop extremely sparse columns
        min_obs = max(120, int(wide.shape[0] * 0.6))
        wide = wide.dropna(axis=1, thresh=min_obs)
        if wide.shape[1] < cfg.min_industry_size:
            continue

        top_pairs = _top_corr_pairs_for_group(wide, top_k=cfg.per_industry_top_corr_pairs)
        for a, b, corr in top_pairs:
            candidates.append((a, b, corr, industry))

    if not candidates:
        raise RuntimeError("No candidates found (check data availability / parameters)")

    # De-dup candidates (a,b) regardless of order; keep best corr record.
    best_by_key: dict[tuple[str, str], tuple[float, str]] = {}
    for a, b, corr, ind in candidates:
        key = tuple(sorted((a, b)))
        cur = best_by_key.get(key)
        if cur is None or corr > cur[0]:
            best_by_key[key] = (corr, ind)

    # Sort by corr desc; evaluate top M candidates to keep runtime bounded.
    cand_sorted = sorted(best_by_key.items(), key=lambda kv: kv[1][0], reverse=True)
    # M is intentionally larger than output top N
    max_eval = max(500, args.top * 20)
    if int(args.max_eval) > 0:
        max_eval = int(args.max_eval)
    cand_sorted = cand_sorted[:max_eval]

    # Prepare price pivot for only involved symbols
    needed = sorted({c for (c1, c2), _ in cand_sorted for c in (c1, c2)})
    df_prices = df_long[df_long["ts_code"].isin(needed)].pivot_table(
        index="trade_date", columns="ts_code", values="close", aggfunc="last"
    ).sort_index()

    rows: list[dict] = []
    for (a, b), (corr, industry) in cand_sorted:
        res = evaluate_pair(
            df_prices,
            a,
            b,
            min_overlap_days=cfg.min_overlap_days,
            roll_window=cfg.roll_window,
            entry_z=cfg.entry_z,
            exit_z=cfg.exit_z,
        )
        if not res:
            continue

        res.update({
            "corr_ret": float(corr),
            "industry": industry,
        })

        rows.append(res)

    out = pd.DataFrame(rows)
    if not out.empty:
        out["pass_filters"] = (
            out["coint_p"].notna()
            & out["adf_p"].notna()
            & (out["coint_p"] <= cfg.coint_pvalue_max)
            & (out["adf_p"] <= cfg.adf_pvalue_max)
            & out["half_life"].notna()
            & (out["half_life"] <= cfg.half_life_max_days)
        )

        out = out.sort_values(
            ["pass_filters", "coint_p", "adf_p", "half_life", "sharpe", "corr_ret"],
            ascending=[False, True, True, True, False, False],
            na_position="last",
        )

    outdir = Path("data")
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"pairs_scan_{cfg.years}y_{cfg.start}_{cfg.end}_{datetime.now().strftime('%Y%m%d')}.csv"
    out.head(int(args.top)).to_csv(outpath, index=False, encoding="utf-8")

    print("=== Pairs Scan Summary ===")
    print(f"Lookback: {cfg.start} .. {cfg.end} (years={cfg.years})")
    print(f"Candidates screened: {len(cand_sorted)} (from industries)")
    if out.empty:
        print("Pairs evaluated: 0 (no valid overlaps)")
        print(f"Saved top 0: {outpath}")
    else:
        passed = int(out["pass_filters"].sum())
        print(f"Pairs evaluated: {len(out)}")
        print(f"Pairs passed filters: {passed}")
        print(f"Saved top {min(int(args.top), len(out))}: {outpath}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
