import datetime as dt
import os
from dataclasses import dataclass
from dotenv import load_dotenv
import statistics
from typing import List, Tuple, Dict

import akshare as ak
import pandas as pd
import tushare as ts


load_dotenv()

START_DATE = "2010-01-01"
TICKER = "h00922.CSI"  # CSI Dividend Total Return Index (includes reinvested dividends)
INITIAL_CASH = 1_000_000.0
DAILY_INVEST = 10_000.0


@dataclass
class StrategyConfig:
    tp: float = 0.04  # take profit threshold
    sl: float | None = None  # stop loss threshold, negative number (e.g., -0.1)
    ma_filter: int | None = None  # e.g., 200 for MA200 filter
    trailing_tp: bool = False  # enable trailing take profit
    trailing_trigger: float | None = None  # e.g., 0.08 (8%) trigger level
    trailing_drawdown: float | None = None  # e.g., 0.03 (3%) drawdown to exit
    valuation_window: int | None = None  # lookback days for percentile (e.g., 252 for 1 year)
    valuation_low: float | None = None  # e.g., 0.3 (30th percentile) for double investment
    valuation_high: float | None = None  # e.g., 0.7 (70th percentile) for half investment
    label: str = "base"


def fetch_price(ticker: str, start: str) -> pd.Series:
    """Fetch daily close price. Try Tushare first (if token set), then AkShare."""
    start_str = start.replace("-", "")
    end_str = dt.date.today().strftime("%Y%m%d")

    # --- Try Tushare ---
    token = os.getenv("TUSHARE_TOKEN")
    if token:
        try:
            ts.set_token(token)
            pro = ts.pro_api()
            df_ts = pro.index_daily(ts_code=ticker, start_date=start_str, end_date=end_str)
            if df_ts is not None and not df_ts.empty:
                df_ts["trade_date"] = pd.to_datetime(df_ts["trade_date"])
                df_ts = df_ts.sort_values("trade_date").set_index("trade_date")
                return df_ts["close"].astype(float)
        except Exception as exc:  # noqa: BLE001
            print(f"Tushare fetch failed, will try AkShare. Reason: {exc}")

    # --- Fallback: AkShare ---
    symbol = ticker.lower().replace(".sh", "sh").replace(".sz", "sz")
    df = ak.index_zh_a_hist(symbol=symbol, period="daily", start_date=start_str, end_date=end_str)
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {ticker} via Tushare or AkShare")
    df = df.rename(columns={"日期": "date", "收盘": "Close"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    return df["Close"].astype(float)


def run_strategy(price: pd.Series, cfg: StrategyConfig) -> Tuple[pd.Series, List[dict]]:
    cash = INITIAL_CASH
    shares = 0.0
    invested = 0.0
    cycle_start = None
    trailing_peak = 0.0  # track peak return for trailing TP

    equity = []
    trades: List[dict] = []

    ma_ok = pd.Series(True, index=price.index)
    if cfg.ma_filter:
        ma = price.rolling(cfg.ma_filter).mean()
        ma_ok = price > ma

    # Valuation percentile for dynamic DCA
    valuation_multiplier = pd.Series(1.0, index=price.index)
    if cfg.valuation_window and cfg.valuation_low and cfg.valuation_high:
        rolling_min = price.rolling(cfg.valuation_window, min_periods=20).min()
        rolling_max = price.rolling(cfg.valuation_window, min_periods=20).max()
        percentile = (price - rolling_min) / (rolling_max - rolling_min + 1e-9)
        valuation_multiplier = percentile.apply(
            lambda p: 2.0 if p < cfg.valuation_low else (0.5 if p > cfg.valuation_high else 1.0)
        )

    for date, px in price.items():
        current_return = (px * shares - invested) / invested if invested > 0 else 0.0

        # Optional stop-loss on live cycle
        if invested > 0 and cfg.sl is not None and current_return <= cfg.sl:
            proceeds = shares * px
            ret = (proceeds - invested) / invested
            trades.append({
                "start": cycle_start,
                "end": date,
                "days": (date - cycle_start).days if cycle_start else 0,
                "invested": invested,
                "proceeds": proceeds,
                "return": ret,
                "reason": "stop_loss",
            })
            cash += proceeds
            shares = 0.0
            invested = 0.0
            cycle_start = None
            trailing_peak = 0.0

        # Trailing take-profit logic
        if cfg.trailing_tp and cfg.trailing_trigger and cfg.trailing_drawdown and invested > 0:
            if current_return > trailing_peak:
                trailing_peak = current_return
            # Exit if peaked above trigger and now drawn down by threshold
            if trailing_peak >= cfg.trailing_trigger and (trailing_peak - current_return) >= cfg.trailing_drawdown:
                proceeds = shares * px
                ret = (proceeds - invested) / invested
                trades.append({
                    "start": cycle_start,
                    "end": date,
                    "days": (date - cycle_start).days if cycle_start else 0,
                    "invested": invested,
                    "proceeds": proceeds,
                    "return": ret,
                    "reason": "trailing_tp",
                })
                cash += proceeds
                shares = 0.0
                invested = 0.0
                cycle_start = None
                trailing_peak = 0.0

        # Standard take-profit check (if not using trailing or not triggered yet)
        if not cfg.trailing_tp and invested > 0 and current_return >= cfg.tp:
            proceeds = shares * px
            ret = (proceeds - invested) / invested
            trades.append(
                {
                    "start": cycle_start,
                    "end": date,
                    "days": (date - cycle_start).days if cycle_start else 0,
                    "invested": invested,
                    "proceeds": proceeds,
                    "return": ret,
                    "reason": "take_profit",
                }
            )
            cash += proceeds
            shares = 0.0
            invested = 0.0
            cycle_start = None
            trailing_peak = 0.0

        # Dynamic DCA: adjust daily investment by valuation multiplier
        daily_invest = DAILY_INVEST * valuation_multiplier.loc[date]
        if cash >= daily_invest and ma_ok.loc[date]:
            if cycle_start is None:
                cycle_start = date
            buy_shares = daily_invest / px
            shares += buy_shares
            invested += daily_invest
            cash -= daily_invest

        equity.append((date, cash + shares * px))

    equity_series = pd.Series({d: v for d, v in equity}).sort_index()
    return equity_series, trades


def max_drawdown(series: pd.Series) -> float:
    roll_max = series.cummax()
    drawdown = series / roll_max - 1.0
    return drawdown.min()


def annualized_return(final_value: float, start: pd.Timestamp, end: pd.Timestamp) -> float:
    days = max((end - start).days, 1)
    return (final_value / INITIAL_CASH) ** (365.0 / days) - 1.0


def summarize(label: str, price: pd.Series, equity: pd.Series, trades: List[dict]) -> Dict[str, float]:
    final_value = equity.iloc[-1]
    total_return = final_value / INITIAL_CASH - 1.0
    cagr = annualized_return(final_value, equity.index[0], equity.index[-1])
    mdd = max_drawdown(equity)

    win_trades = [t for t in trades if t["return"] > 0]
    avg_days = statistics.mean([t["days"] for t in trades]) if trades else 0
    win_rate = len(win_trades) / len(trades) if trades else 0

    print(f"=== {label} ===")
    print(f"Final Equity: {final_value:,.0f} | Total Return: {total_return:.2%} | CAGR: {cagr:.2%}")
    print(f"Max Drawdown: {mdd:.2%}")
    print(f"Cycles closed: {len(trades)} | Win rate: {win_rate:.1%} | Avg days per cycle: {avg_days:.1f}")
    if trades:
        last = trades[-1]
        print(
            f"Last cycle: {last['start'].date()} -> {last['end'].date()} | Return {last['return']:.2%} | Days {last['days']} | Reason {last.get('reason','tp')}"
        )
    print()

    return {
        "final_equity": final_value,
        "total_return": total_return,
        "cagr": cagr,
        "mdd": mdd,
        "cycles": len(trades),
        "win_rate": win_rate,
        "avg_days": avg_days,
    }


def main():
    price = fetch_price(TICKER, START_DATE)

    configs = [
        StrategyConfig(tp=0.04, sl=None, ma_filter=None, label="TP4% (baseline)"),
        StrategyConfig(tp=0.06, sl=-0.10, ma_filter=None, label="TP6% SL-10%"),
        StrategyConfig(tp=0.06, sl=-0.10, ma_filter=200, label="TP6% SL-10% + MA200 filter"),
        StrategyConfig(
            tp=0.08, sl=None, ma_filter=None,
            trailing_tp=True, trailing_trigger=0.08, trailing_drawdown=0.03,
            label="Trailing TP (8% trigger, 3% drawdown)"
        ),
        StrategyConfig(
            tp=0.08, sl=None, ma_filter=None,
            valuation_window=252, valuation_low=0.3, valuation_high=0.7,
            label="Dynamic DCA (valuation percentile)"
        ),
        StrategyConfig(
            tp=0.08, sl=None, ma_filter=None,
            trailing_tp=True, trailing_trigger=0.08, trailing_drawdown=0.03,
            valuation_window=252, valuation_low=0.3, valuation_high=0.7,
            label="Trailing TP + Dynamic DCA"
        ),
    ]

    # Buy & hold benchmark (lump-sum at start)
    bh_shares = INITIAL_CASH / price.iloc[0]
    bh_equity = price * bh_shares
    bh_final = bh_equity.iloc[-1]
    bh_total_return = bh_final / INITIAL_CASH - 1.0
    bh_cagr = annualized_return(bh_final, price.index[0], price.index[-1])
    bh_mdd = max_drawdown(bh_equity)

    print(f"Period: {price.index[0].date()} -> {price.index[-1].date()} ({len(price)} bars)")
    print(f"Benchmark Buy&Hold Final: {bh_final:,.0f} | Total: {bh_total_return:.2%} | CAGR: {bh_cagr:.2%} | MDD: {bh_mdd:.2%}\n")

    summaries = []
    for cfg in configs:
        equity, trades = run_strategy(price, cfg)
        summary = summarize(cfg.label, price, equity, trades)
        summary["label"] = cfg.label
        summaries.append(summary)

        out = pd.DataFrame({
            "strategy": equity,
            "buy_hold": bh_equity.reindex(equity.index).fillna(method="ffill"),
        }).dropna()
        out.to_csv(f"analysis_output_dividend_dca_{cfg.label.replace(' ', '_').replace('%','p').replace('-','minus')}.csv")

    print("=== Summary vs Benchmark ===")
    for s in summaries:
        edge = s["cagr"] - bh_cagr
        print(f"{s['label']}: CAGR {s['cagr']:.2%}, MDD {s['mdd']:.2%}, Cycles {s['cycles']} | Edge vs B&H: {edge:.2%}")


if __name__ == "__main__":
    main()
