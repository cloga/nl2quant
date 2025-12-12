#!/usr/bin/env python
"""Compute forward/dynamic PE using Tushare consensus sources.

Sources supported:
- stk_surv: 券商盈利预测明细（高权限，字段随账号而异）
- report_rc: 研报摘要（含 quarter/eps/np/pe 等结构化字段）

Usage:
    .\.venv\Scripts\python.exe scripts\forward_pe_consensus.py --code 600519.SH --year 2025 --source report_rc
    .\.venv\Scripts\python.exe scripts\forward_pe_consensus.py --code 600519.SH --year 2025 --source auto

Logic:
- Fetch latest price via `pro.daily` (close).
- Fetch total shares via `pro.daily_basic` (total_share, unit: 万股).
- Get EPS forecast via selected source:
    * stk_surv: try to locate EPS column for given year.
    * report_rc: prefer quarter == '{year}Q4', else latest non-null EPS for given year, else latest.
- Net profit forecast ≈ EPS * total_shares
- Market cap ≈ close * total_shares
- Forward PE = Market cap / Net profit forecast

Notes:
- If EPS is missing, script reports clearly and exits.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import os

import pandas as pd
import tushare as ts
from dotenv import load_dotenv

# ensure project root on sys.path (optional for shared utils)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _pro():
    load_dotenv()
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("TUSHARE_TOKEN not set in .env")
    ts.set_token(token)
    return ts.pro_api()


def fetch_price_and_shares(pro, ts_code: str) -> tuple[float | None, float | None]:
    # latest close
    df_p = pro.daily(ts_code=ts_code, limit=1, fields="ts_code,trade_date,close")
    close = float(df_p.iloc[0]["close"]) if df_p is not None and not df_p.empty else None

    # latest basic for share info
    df_b = pro.daily_basic(ts_code=ts_code, limit=1, fields="ts_code,trade_date,total_share")
    # total_share unit: 万股
    total_share_w = float(df_b.iloc[0]["total_share"]) if df_b is not None and not df_b.empty else None
    return close, total_share_w


def fetch_consensus_eps_stk_surv(pro, ts_code: str, year: int) -> float | None:
    """Try to fetch EPS from stk_surv if available (fields vary by account)."""
    try:
        df = pro.stk_surv(ts_code=ts_code)
    except Exception as e:
        print(f"stk_surv 不可用或权限不足: {e}")
        return None
    if df is None or df.empty:
        print("stk_surv 无数据")
        return None

    eps_col = None
    year_col = None
    for c in df.columns:
        lc = c.lower()
        if lc in {"eps", "forecast_eps", "eps_forecast"} or "eps" in lc:
            eps_col = c if eps_col is None else eps_col
        if lc in {"report_year", "year"}:
            year_col = c

    if eps_col is None:
        print("stk_surv 未发现 EPS 字段")
        return None

    if year_col and year in set(pd.to_numeric(df[year_col], errors="coerce").dropna().astype(int)):
        df_year = df[pd.to_numeric(df[year_col], errors="coerce").astype("Int64") == year]
        df_year = df_year.dropna(subset=[eps_col])
        if not df_year.empty:
            val = df_year.iloc[-1].get(eps_col)
            return float(val) if pd.notna(val) else None

    # Fallback: latest non-null eps
    df2 = df.dropna(subset=[eps_col])
    if df2.empty:
        return None
    val = df2.iloc[-1].get(eps_col)
    return float(val) if pd.notna(val) else None


def fetch_consensus_eps_report_rc(pro, ts_code: str, year: int) -> float | None:
    """Fetch EPS from report_rc, preferring {year}Q4, else latest non-null in the year, else latest overall."""
    try:
        df = pro.report_rc(ts_code=ts_code)
    except Exception as e:
        print(f"report_rc 不可用或权限不足: {e}")
        return None
    if df is None or df.empty:
        print("report_rc 无数据")
        return None

    if "eps" not in df.columns:
        print("report_rc 未发现 eps 字段")
        return None

    # Normalize quarter to string
    quarter = df.get("quarter")
    df_eps = df.copy()
    df_eps["eps"] = pd.to_numeric(df_eps["eps"], errors="coerce")
    df_eps = df_eps.dropna(subset=["eps"])
    if df_eps.empty:
        return None

    target_q = f"{year}Q4"
    if quarter is not None:
        mask_year = quarter.astype(str).str.startswith(str(year))
        df_year = df_eps[mask_year]
        if not df_year.empty:
            df_q4 = df_year[quarter.astype(str) == target_q]
            if not df_q4.empty:
                return float(df_q4.iloc[-1]["eps"])
            # fallback: latest in year
            return float(df_year.iloc[-1]["eps"]) if pd.notna(df_year.iloc[-1]["eps"]) else None

    # fallback: latest overall
    return float(df_eps.iloc[-1]["eps"]) if pd.notna(df_eps.iloc[-1]["eps"]) else None


def compute_forward_pe(ts_code: str, year: int, source: str = "auto") -> None:
    pro = _pro()
    close, total_share_w = fetch_price_and_shares(pro, ts_code)
    if close is None or total_share_w is None:
        print("无法获取价格或总股本数据")
        return

    eps = None
    if source == "stk_surv":
        eps = fetch_consensus_eps_stk_surv(pro, ts_code, year)
    elif source == "report_rc":
        eps = fetch_consensus_eps_report_rc(pro, ts_code, year)
    else:  # auto
        eps = fetch_consensus_eps_stk_surv(pro, ts_code, year)
        if eps is None:
            eps = fetch_consensus_eps_report_rc(pro, ts_code, year)

    if eps is None:
        print("缺少 EPS 一致预期，无法计算前瞻动态PE")
        return

    # Convert shares: 万股 -> 股
    total_shares = total_share_w * 10000.0

    # Market cap ≈ close * total_shares
    market_cap = close * total_shares

    # Net profit forecast ≈ EPS * total_shares
    net_profit_forecast = eps * total_shares

    if net_profit_forecast <= 0:
        print("预测净利润 <= 0，前瞻PE无意义")
        return

    forward_pe = market_cap / net_profit_forecast

    print(f"{ts_code} 前瞻动态PE (基于一致预期 EPS @ {year})")
    print(f"  最新收盘: {close}")
    print(f"  总股本(万股): {total_share_w}")
    print(f"  预测EPS: {eps}")
    print(f"  预测净利润: {net_profit_forecast:,.0f}")
    print(f"  前瞻PE: {forward_pe:.2f}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--code", required=True, help="A-share ts_code, e.g. 600519.SH")
    p.add_argument("--year", type=int, default=pd.Timestamp.today().year)
    p.add_argument("--source", choices=["auto", "stk_surv", "report_rc"], default="auto")
    args = p.parse_args()

    compute_forward_pe(args.code, args.year, source=args.source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
