import os
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta

# Ensure project root on path for `app` imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.config import Config

# All-weather China ETF basket (monthly rebalance)
TARGET_WEIGHTS = {
    "511260": 0.40,   # 十年国债ETF（合并30年份额）
    "513100": 0.10,   # 纳指ETF
    "518880": 0.10,   # 黄金ETF（5位补0 -> 518880）
    "510300": 0.10,   # 沪深300ETF
    "515080": 0.10,   # 中证红利ETF
    "159920": 0.10,   # 恒生ETF
    "501018": 0.04,   # 南方原油ETF（LOF）
    "159980": 0.03,   # 有色ETF
    "159985": 0.03,   # 豆粕ETF
}

INITIAL_CAPITAL = 1_000_000
START_DATE = "20000101"
END_DATE = datetime.today().strftime("%Y%m%d")
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# Optimization parameters
REBALANCE_FREQ = "Q"  # Quarterly rebalance
TARGET_VOL = None  # Set to None to disable volatility targeting
VOL_LOOKBACK = 60  # 60 trading days for vol estimation


def candidate_ts_codes(code: str) -> list[str]:
    """Generate plausible Tushare ts_code variants for an ETF code."""
    code = code.strip().upper()
    if "." in code:
        return [code]

    # Heuristic: most ETFs are 6-digit; if 5-digit, append trailing 0 (e.g., 51889 -> 518890)
    if len(code) == 5:
        code6 = code + "0"
    else:
        code6 = code

    guesses = []
    if code6[0] in {"5", "6"}:  # SH ETFs/bonds/commodities typically start with 5/6
        guesses.append(f"{code6}.SH")
        guesses.append(f"{code6}.SZ")
    elif code6[0] in {"1", "3"}:  # SZ ETFs
        guesses.append(f"{code6}.SZ")
        guesses.append(f"{code6}.SH")
    else:
        guesses.append(f"{code6}.SH")
        guesses.append(f"{code6}.SZ")
    # De-dup while preserving order
    seen = set()
    dedup = []
    for g in guesses:
        if g not in seen:
            seen.add(g)
            dedup.append(g)
    return dedup


def fetch_with_chunks(pro, api_name: str, ts_code: str, sd: str, ed: str) -> pd.DataFrame:
    frames = []
    sd_dt = datetime.strptime(sd, "%Y%m%d")
    ed_dt = datetime.strptime(ed, "%Y%m%d")
    cursor = sd_dt
    while cursor <= ed_dt:
        chunk_end = min(cursor + timedelta(days=365), ed_dt)
        chunk_sd = cursor.strftime("%Y%m%d")
        chunk_ed = chunk_end.strftime("%Y%m%d")
        api_fn = getattr(pro, api_name)
        df_chunk = api_fn(ts_code=ts_code, start_date=chunk_sd, end_date=chunk_ed)
        if df_chunk is not None and not df_chunk.empty:
            frames.append(df_chunk)
        cursor = chunk_end + timedelta(days=1)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fetch_etf_close(code: str, start_date: str, end_date: str, pro) -> pd.Series:
    """Fetch ETF close prices via Tushare fund_daily/daily with fallbacks, chunking, and retries."""
    last_exc = None
    ts_codes = candidate_ts_codes(code)
    for ts_code in ts_codes:
        for _ in range(MAX_RETRIES):
            try:
                df = fetch_with_chunks(pro, "fund_daily", ts_code, start_date, end_date)
                if df is None or df.empty:
                    df = fetch_with_chunks(pro, "daily", ts_code, start_date, end_date)
                if df is None or df.empty:
                    last_exc = ValueError(f"No data for {ts_code}")
                    time.sleep(RETRY_DELAY)
                    continue
                date_col = "trade_date" if "trade_date" in df.columns else ("nav_date" if "nav_date" in df.columns else None)
                if date_col is None:
                    last_exc = ValueError(f"No date column for {ts_code}")
                    time.sleep(RETRY_DELAY)
                    continue
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.sort_values(date_col).set_index(date_col)
                close_col = "close" if "close" in df.columns else ("nav" if "nav" in df.columns else None)
                if close_col is None:
                    last_exc = ValueError(f"No close/nav column for {ts_code}")
                    time.sleep(RETRY_DELAY)
                    continue
                df = df[[close_col]]
                df.columns = ["close"]
                return df["close"].astype(float)
            except Exception as e:  # noqa: BLE001
                last_exc = e
                time.sleep(RETRY_DELAY)
                continue
    raise last_exc if last_exc else RuntimeError(f"Failed to fetch {code}")


def build_price_frame(weights: dict[str, float], start_date: str, end_date: str, pro):
    available = {}
    dropped = []
    for code in weights:
        try:
            s = fetch_etf_close(code, start_date, end_date, pro)
            available[code] = s
            print(f"Fetched {code}: {s.index[0].date()} -> {s.index[-1].date()}, {len(s)} rows")
        except Exception as e:
            print(f"Skip {code}: {e}")
            dropped.append(code)
    if not available:
        raise RuntimeError("No ETF data available; all fetches failed.")

    # Ensure overlapping date window exists; iteratively drop the shortest-coverage asset until overlap is positive
    overlap_dropped = []
    while True:
        starts = {k: v.index[0] for k, v in available.items()}
        ends = {k: v.index[-1] for k, v in available.items()}
        max_start = max(starts.values())
        min_end = min(ends.values())
        if max_start <= min_end:
            break
        drop_code = min(ends, key=ends.get)  # drop the asset with the earliest end date
        overlap_dropped.append(drop_code)
        available.pop(drop_code)
        print(f"Dropped {drop_code} due to no overlap (ends {ends[drop_code].date()})")
        if not available:
            raise RuntimeError("No overlapping date range across ETFs.")

    # Trim all series to the common overlap window
    starts = {k: v.index[0] for k, v in available.items()}
    ends = {k: v.index[-1] for k, v in available.items()}
    max_start = max(starts.values())
    min_end = min(ends.values())
    for k in list(available.keys()):
        available[k] = available[k].loc[(available[k].index >= max_start) & (available[k].index <= min_end)]

    prices = pd.concat(available.values(), axis=1, join="inner")
    prices.columns = list(available.keys())
    prices = prices.dropna()

    if prices.empty:
        raise RuntimeError("Price frame is empty after aligning overlap; please check data availability.")

    # Renormalize weights to keep proportions among available ETFs
    weights_available = {k: v for k, v in weights.items() if k in prices.columns}
    total_w = sum(weights_available.values())
    weights_norm = {k: v / total_w for k, v in weights_available.items()}
    dropped.extend(overlap_dropped)
    return prices, weights_norm, dropped


def run_rebalance_with_voltarget(prices: pd.DataFrame, weights: dict[str, float], initial_capital: float, 
                                  rebal_freq: str = "Q", target_vol: float = 0.06, vol_lookback: int = 60) -> pd.DataFrame:
    """
    Rebalance at specified frequency with volatility targeting.
    rebal_freq: 'M' (monthly), 'Q' (quarterly), etc.
    target_vol: annualized target volatility (e.g., 0.06 = 6%)
    vol_lookback: trading days for rolling vol estimation
    """
    holdings = pd.Series(0.0, index=prices.columns)
    cash = 0.0
    records = []
    prev_period = None
    leverage = 1.0  # vol targeting leverage multiplier
    
    for i, (dt, row) in enumerate(prices.iterrows()):
        # Determine rebalance period key
        if rebal_freq == "Q":
            period_key = (dt.year, (dt.month - 1) // 3)  # 0-3 for quarters
        elif rebal_freq == "M":
            period_key = (dt.year, dt.month)
        else:
            period_key = (dt.year, dt.month)  # fallback to monthly
        
        total_value = float((holdings * row).sum() + cash)
        if prev_period is None:
            total_value = initial_capital
        
        # Rebalance if period changed
        if prev_period != period_key:
            # Estimate realized volatility from past returns (if sufficient history)
            if target_vol and i >= vol_lookback and len(records) >= vol_lookback:
                recent_rets = [records[j]["daily_return"] for j in range(len(records) - vol_lookback, len(records))]
                realized_vol = np.std(recent_rets, ddof=1) * np.sqrt(252)
                if realized_vol > 0.001:  # avoid division by zero
                    leverage = target_vol / realized_vol
                    leverage = max(0.5, min(1.0, leverage))  # clamp to [0.5, 1.0] - reduce exposure only, no borrowing
            
            # Apply leverage to target allocation (leverage <= 1 means partial cash position)
            target_value = pd.Series({k: total_value * weights[k] * leverage for k in prices.columns})
            holdings = target_value / row
            cash = total_value * (1 - leverage)  # remaining as cash buffer
            rebalanced = True
        else:
            rebalanced = False
        
        portfolio_value = float((holdings * row).sum() + cash)
        weight_now = (holdings * row) / portfolio_value if portfolio_value else pd.Series(0, index=prices.columns)
        records.append({
            "date": dt,
            "portfolio_value": portfolio_value,
            "daily_return": 0.0 if not records else portfolio_value / records[-1]["portfolio_value"] - 1,
            "rebalance": rebalanced,
            "leverage": leverage,
            **{f"weight_{k}": weight_now.get(k, 0.0) for k in prices.columns},
            **{f"price_{k}": row[k] for k in prices.columns},
            **{f"units_{k}": holdings[k] for k in prices.columns},
        })
        prev_period = period_key
    return pd.DataFrame(records).set_index("date")


def compute_metrics(equity: pd.DataFrame) -> dict:
    final_value = equity["portfolio_value"].iloc[-1]
    total_return = final_value / INITIAL_CAPITAL - 1
    days = max((equity.index[-1] - equity.index[0]).days, 1)
    cagr = (1 + total_return) ** (365.0 / days) - 1
    daily_ret = equity["daily_return"].fillna(0)
    ann_vol = daily_ret.std(ddof=0) * np.sqrt(252)
    ann_return = daily_ret.mean() * 252
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    roll_max = equity["portfolio_value"].cummax()
    drawdown = equity["portfolio_value"] / roll_max - 1
    max_dd = drawdown.min()
    return {
        "final_value": final_value,
        "total_return": total_return,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "start_date": equity.index[0].date().isoformat(),
        "end_date": equity.index[-1].date().isoformat(),
        "n_days": days,
        "n_rebalances": int(equity["rebalance"].sum()),
    }


def main():
    token = os.getenv("TUSHARE_TOKEN") or Config.TUSHARE_TOKEN
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is not set in environment; please configure it.")
    ts.set_token(token)
    pro = ts.pro_api()

    prices, weights_norm, dropped = build_price_frame(TARGET_WEIGHTS, START_DATE, END_DATE, pro)
    equity = run_rebalance_with_voltarget(prices, weights_norm, INITIAL_CAPITAL, 
                                           rebal_freq=REBALANCE_FREQ, target_vol=TARGET_VOL, vol_lookback=VOL_LOOKBACK)
    metrics = compute_metrics(equity)

    # Monthly snapshots (month-end) with positions/weights
    monthly = equity.resample("ME").last().copy()
    monthly_out = "test/rebalance_all_weather_monthly.csv"
    monthly.to_csv(monthly_out)

    # Annual returns based on year-end equity
    yearly = equity.resample("YE").last()[["portfolio_value"]].copy()
    yearly["annual_return"] = yearly["portfolio_value"].pct_change()
    yearly_out = "test/rebalance_all_weather_annual_returns.csv"
    yearly.to_csv(yearly_out)

    out_path = "test/rebalance_all_weather_equity.csv"
    equity.to_csv(out_path)

    rebal_label = {"Q": "季度", "M": "月度"}.get(REBALANCE_FREQ, REBALANCE_FREQ)
    vol_label = f" + 波动率目标({TARGET_VOL*100:.0f}%)" if TARGET_VOL else ""
    print(f"=== A股全天候组合：{rebal_label}再平衡{vol_label} ===")
    print(f"数据覆盖: {metrics['start_date']} -> {metrics['end_date']} ({metrics['n_days']} 天)")
    print(f"样本ETF: {', '.join(prices.columns)}")
    if dropped:
        print(f"未使用(无数据): {', '.join(dropped)}")
    print(f"再平衡次数: {metrics['n_rebalances']}")
    print(f"期末资产: {metrics['final_value']:.2f}")
    print(f"总收益率: {metrics['total_return']*100:.2f}%")
    print(f"CAGR: {metrics['cagr']*100:.2f}%")
    print(f"年化波动: {metrics['ann_vol']*100:.2f}%")
    print(f"Sharpe (rf=0): {metrics['sharpe']:.2f}")
    print(f"最大回撤: {metrics['max_drawdown']*100:.2f}%")
    last_weights = equity[[c for c in equity.columns if c.startswith('weight_')]].iloc[-1]
    print("期末权重:")
    for k, v in last_weights.items():
        print(f"  {k.replace('weight_', '')}: {v*100:.2f}%")
    # Print latest month-end snapshot
    latest = monthly.iloc[-1]
    print("最新月度快照:")
    print(f"  日期: {latest.name.date()}")
    print(f"  资产: {latest['portfolio_value']:.2f}")
    for k in prices.columns:
        w_col = f"weight_{k}"
        u_col = f"units_{k}"
        p_col = f"price_{k}"
        print(f"  {k}: 权重 {latest[w_col]*100:.2f}%, 价格 {latest[p_col]:.4f}, 持仓 {latest[u_col]:.4f}")

    print(f"月度数据已保存: {monthly_out}")
    print(f"年度收益表已保存: {yearly_out}")


if __name__ == "__main__":
    main()
