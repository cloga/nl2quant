import pandas as pd
import numpy as np
import akshare as ak
from datetime import datetime

TARGET_WEIGHTS = {"511090": 0.5, "515080": 0.5}
INITIAL_CAPITAL = 1_000_000  # 50W + 50W
START_DATE = "20000101"  # earliest possible
END_DATE = datetime.today().strftime("%Y%m%d")


def fetch_etf_close(code: str, start_date: str, end_date: str) -> pd.Series:
    """Fetch ETF close prices via AkShare fund_etf_hist_em."""
    df = ak.fund_etf_hist_em(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="")
    if df is None or df.empty:
        raise ValueError(f"No data returned for {code}")
    df = df.rename(columns={"日期": "date", "收盘": "close"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    return df["close"].astype(float)


def build_price_frame(codes: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    series_list = []
    for code in codes:
        s = fetch_etf_close(code, start_date, end_date)
        s.name = code
        series_list.append(s)
    prices = pd.concat(series_list, axis=1, join="inner").dropna()
    return prices


def run_monthly_rebalance(prices: pd.DataFrame, weights: dict[str, float], initial_capital: float) -> pd.DataFrame:
    holdings = pd.Series(0.0, index=prices.columns)
    cash = 0.0
    records = []
    prev_month = None
    for dt, row in prices.iterrows():
        month_key = (dt.year, dt.month)
        total_value = float((holdings * row).sum() + cash)
        if prev_month is None:
            total_value = initial_capital
        if prev_month != month_key:
            target_value = pd.Series({k: total_value * weights[k] for k in prices.columns})
            holdings = target_value / row
            cash = 0.0
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
            **{f"weight_{k}": weight_now.get(k, 0.0) for k in prices.columns},
            **{f"price_{k}": row[k] for k in prices.columns},
            **{f"units_{k}": holdings[k] for k in prices.columns},
        })
        prev_month = month_key
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
    codes = list(TARGET_WEIGHTS.keys())
    prices = build_price_frame(codes, START_DATE, END_DATE)
    equity = run_monthly_rebalance(prices, TARGET_WEIGHTS, INITIAL_CAPITAL)
    metrics = compute_metrics(equity)

    out_path = "test/rebalance_511090_515080_equity.csv"
    equity.to_csv(out_path)

    print("=== Monthly Rebalance: 50% 511090 / 50% 515080 ===")
    print(f"Data range: {metrics['start_date']} -> {metrics['end_date']} ({metrics['n_days']} days)")
    print(f"Rebalances: {metrics['n_rebalances']}")
    print(f"Final equity: {metrics['final_value']:.2f}")
    print(f"Total return: {metrics['total_return']*100:.2f}%")
    print(f"CAGR: {metrics['cagr']*100:.2f}%")
    print(f"Ann. vol: {metrics['ann_vol']*100:.2f}%")
    print(f"Sharpe (rf=0): {metrics['sharpe']:.2f}")
    print(f"Max drawdown: {metrics['max_drawdown']*100:.2f}%")
    last_weights = equity[[c for c in equity.columns if c.startswith('weight_')]].iloc[-1]
    print("Last weights:")
    for k, v in last_weights.items():
        print(f"  {k.replace('weight_', '')}: {v*100:.2f}%")
    print(f"Equity curve saved to {out_path}")


if __name__ == "__main__":
    main()
