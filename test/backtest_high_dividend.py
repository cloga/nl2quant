"""
High Dividend Stock Portfolio Backtest
========================================
Strategy: Select stocks with dividend yield > 5%, hold equal-weight portfolio,
rebalance annually, compare with CSI Dividend index benchmark.

Due to Tushare data limitations for individual stock dividends, we use a proxy:
- CSI High Dividend Index (000842.CSI) as the high-dividend portfolio
- CSI Dividend Index (h00922.CSI) as benchmark
- Period: 2010-2025

For real implementation with individual stocks, would need:
1. Stock screening: dividend_yield > 5%, consecutive dividends > 3 years
2. Daily price data + annual dividend records
3. Tax consideration (10% dividend tax in China)
"""

import datetime as dt
import os
from dotenv import load_dotenv

import pandas as pd
import tushare as ts

load_dotenv()

START_DATE = "2010-01-01"
INITIAL_CASH = 1_000_000.0


def fetch_index_price(ticker: str, start: str) -> pd.Series:
    """Fetch daily close price via Tushare."""
    start_str = start.replace("-", "")
    end_str = dt.date.today().strftime("%Y%m%d")

    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("TUSHARE_TOKEN not set in .env")

    ts.set_token(token)
    pro = ts.pro_api()

    df = pro.index_daily(ts_code=ticker, start_date=start_str, end_date=end_str)
    if df is None or df.empty:
        raise RuntimeError(f"No data for {ticker}")

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date").set_index("trade_date")
    return df["close"].astype(float)


def max_drawdown(series: pd.Series) -> float:
    roll_max = series.cummax()
    drawdown = series / roll_max - 1.0
    return drawdown.min()


def annualized_return(final_value: float, initial: float, start: pd.Timestamp, end: pd.Timestamp) -> float:
    days = max((end - start).days, 1)
    return (final_value / initial) ** (365.0 / days) - 1.0


def compute_sharpe(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Compute annualized Sharpe ratio."""
    excess = returns - risk_free_rate / 252
    if excess.std() == 0:
        return 0.0
    return excess.mean() / excess.std() * (252 ** 0.5)


def main():
    # Fetch data
    print("Fetching index data...")
    high_div_price = fetch_index_price("000842.CSI", START_DATE)  # CSI High Dividend
    benchmark_price = fetch_index_price("h00922.CSI", START_DATE)  # CSI Dividend Total Return

    # Align dates
    common_dates = high_div_price.index.intersection(benchmark_price.index)
    high_div_price = high_div_price.loc[common_dates]
    benchmark_price = benchmark_price.loc[common_dates]

    # Portfolio: Buy & Hold with annual rebalance (simulated by single buy-hold here)
    # In real case with multiple stocks, would rebalance each year
    portfolio_shares = INITIAL_CASH / high_div_price.iloc[0]
    portfolio_equity = high_div_price * portfolio_shares

    # Benchmark: Buy & Hold
    benchmark_shares = INITIAL_CASH / benchmark_price.iloc[0]
    benchmark_equity = benchmark_price * benchmark_shares

    # Metrics
    portfolio_final = portfolio_equity.iloc[-1]
    portfolio_total_return = portfolio_final / INITIAL_CASH - 1.0
    portfolio_cagr = annualized_return(portfolio_final, INITIAL_CASH, portfolio_equity.index[0], portfolio_equity.index[-1])
    portfolio_mdd = max_drawdown(portfolio_equity)
    portfolio_returns = portfolio_equity.pct_change().dropna()
    portfolio_sharpe = compute_sharpe(portfolio_returns)

    benchmark_final = benchmark_equity.iloc[-1]
    benchmark_total_return = benchmark_final / INITIAL_CASH - 1.0
    benchmark_cagr = annualized_return(benchmark_final, INITIAL_CASH, benchmark_equity.index[0], benchmark_equity.index[-1])
    benchmark_mdd = max_drawdown(benchmark_equity)
    benchmark_returns = benchmark_equity.pct_change().dropna()
    benchmark_sharpe = compute_sharpe(benchmark_returns)

    # Calculate correlation
    correlation = portfolio_returns.corr(benchmark_returns)

    # Volatility
    portfolio_vol = portfolio_returns.std() * (252 ** 0.5)
    benchmark_vol = benchmark_returns.std() * (252 ** 0.5)

    print("\n" + "=" * 60)
    print("HIGH DIVIDEND STRATEGY BACKTEST (2010-2025)")
    print("=" * 60)
    print(f"Period: {portfolio_equity.index[0].date()} -> {portfolio_equity.index[-1].date()} ({len(portfolio_equity)} days)")
    print(f"Initial Capital: {INITIAL_CASH:,.0f}")
    print()

    print("--- Portfolio: CSI High Dividend (000842.CSI) ---")
    print(f"Final Equity:      {portfolio_final:>12,.0f}")
    print(f"Total Return:      {portfolio_total_return:>12.2%}")
    print(f"CAGR:              {portfolio_cagr:>12.2%}")
    print(f"Max Drawdown:      {portfolio_mdd:>12.2%}")
    print(f"Sharpe Ratio:      {portfolio_sharpe:>12.2f}")
    print(f"Volatility (Ann):  {portfolio_vol:>12.2%}")
    print()

    print("--- Benchmark: CSI Dividend Total Return (h00922.CSI) ---")
    print(f"Final Equity:      {benchmark_final:>12,.0f}")
    print(f"Total Return:      {benchmark_total_return:>12.2%}")
    print(f"CAGR:              {benchmark_cagr:>12.2%}")
    print(f"Max Drawdown:      {benchmark_mdd:>12.2%}")
    print(f"Sharpe Ratio:      {benchmark_sharpe:>12.2f}")
    print(f"Volatility (Ann):  {benchmark_vol:>12.2%}")
    print()

    print("--- Comparison ---")
    edge = portfolio_cagr - benchmark_cagr
    print(f"CAGR Edge:         {edge:>12.2%}")
    print(f"Correlation:       {correlation:>12.2f}")
    print(f"MDD Improvement:   {(portfolio_mdd - benchmark_mdd):>12.2%}")
    print()

    # Export
    out = pd.DataFrame({
        "high_dividend": portfolio_equity,
        "benchmark": benchmark_equity,
    })
    out.to_csv("analysis_output_high_dividend.csv")
    print("Saved equity curves to analysis_output_high_dividend.csv")
    print("=" * 60)

    # Analysis notes
    print("\nNOTES:")
    print("- CSI High Dividend (000842) screens for stocks with high dividend yield")
    print("- CSI Dividend Total Return (h00922) includes dividend reinvestment")
    print("- Real strategy with individual stocks would need:")
    print("  1. Screen: dividend yield > 5%, consecutive dividends > 3 years")
    print("  2. Equal-weight 10-20 stocks")
    print("  3. Annual rebalance (add/remove based on latest dividend)")
    print("  4. Consider 10% dividend tax impact")
    print("- Expected annual cash flow from dividends: 5-7% of portfolio value")


if __name__ == "__main__":
    main()
