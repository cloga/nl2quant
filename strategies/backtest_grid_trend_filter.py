"""
Grid Trading + Trend Filter Strategy (Revised)
===============================================
Enhanced Logic:
1. Base position: 50W buy-hold (always earning dividend)
2. Grid position: 50W active trading
   - ALWAYS try to buy at grid levels
   - ONLY sell if price > MA120 (avoid selling into downtrends)
   - When price < MA120, accumulate more shares without selling
   - When price > MA120, resume normal grid selling

This converts grid strategy from "range-bound" to "trend-aware":
- In downtrends: buy more, don't sell → accumulate cheap shares
- In uptrends: sell normally → lock in gains
"""

import datetime as dt
import os
from collections import defaultdict
from typing import List, Tuple

import pandas as pd
import tushare as ts
from dotenv import load_dotenv

load_dotenv()

START_DATE = "2018-01-01"
TICKER = "h00922.CSI"
INITIAL_CASH = 1_000_000.0
BASE_POSITION = 500_000.0
GRID_CASH = 500_000.0
GRID_SPACING = 0.02
GRID_UNIT = 50_000.0
MA_PERIOD = 120
COMMISSION_RATE = 0.0003
STAMP_TAX_RATE = 0.001


def fetch_price(ticker: str, start: str) -> pd.Series:
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


def calculate_cost(amount: float, is_buy: bool) -> float:
    """Calculate transaction cost."""
    commission = amount * COMMISSION_RATE
    stamp_tax = amount * STAMP_TAX_RATE if not is_buy else 0.0
    return commission + stamp_tax


def run_grid_strategy(price: pd.Series, use_trend_filter: bool = False) -> Tuple[pd.Series, pd.Series, dict]:
    """
    Run grid trading strategy.
    
    Args:
        use_trend_filter: If True, only sell when price > MA120 (avoid selling in downtrends)
    """
    # Calculate MA120
    ma120 = price.rolling(MA_PERIOD, min_periods=1).mean()

    # Base position: buy and hold
    base_shares = BASE_POSITION / price.iloc[0]
    base_equity = price * base_shares

    # Grid position state
    grid_cash = GRID_CASH
    grid_shares = 0.0
    grid_equity_list = []
    
    # Track pending sell orders by level
    pending_sells: dict = defaultdict(float)  # price_level -> shares
    trades: List[dict] = []
    
    entry_price = price.iloc[0]
    
    for idx, (date, current_price) in enumerate(price.items()):
        ma_value = ma120.iloc[idx] if idx < len(ma120) else ma120.iloc[-1]
        is_uptrend = current_price > ma_value
        
        # Process pending sell orders
        executed_levels = []
        for sell_level, sell_shares in pending_sells.items():
            if current_price >= sell_level:
                # Check trend filter
                if use_trend_filter and not is_uptrend:
                    # In downtrend, skip selling this time
                    continue
                
                # Execute sell
                proceeds = sell_shares * current_price
                cost = calculate_cost(proceeds, is_buy=False)
                grid_cash += proceeds - cost
                grid_shares -= sell_shares
                executed_levels.append(sell_level)
                
                trades.append({
                    'date': date,
                    'type': 'sell',
                    'price': current_price,
                    'shares': sell_shares,
                    'trend': 'uptrend' if is_uptrend else 'downtrend',
                })
        
        for level in executed_levels:
            del pending_sells[level]
        
        # Try to buy at grid levels
        for level_mult in range(1, 11):
            buy_level = entry_price * (1 - GRID_SPACING * level_mult)
            
            # Skip if price too far below this level
            if buy_level < current_price * 0.5:
                continue
            
            # Only buy if we're near this level and have cash
            if current_price <= buy_level * 1.001 and grid_cash >= GRID_UNIT:
                # Check if we already have a sell order for this level
                sell_level = buy_level * (1 + GRID_SPACING)
                if sell_level not in pending_sells:
                    # Execute buy
                    buy_amount = min(GRID_UNIT, grid_cash)
                    cost = calculate_cost(buy_amount, is_buy=True)
                    shares_bought = (buy_amount - cost) / current_price
                    
                    grid_cash -= buy_amount
                    grid_shares += shares_bought
                    
                    trades.append({
                        'date': date,
                        'type': 'buy',
                        'price': current_price,
                        'shares': shares_bought,
                        'trend': 'uptrend' if is_uptrend else 'downtrend',
                    })
                    
                    # Create sell order
                    pending_sells[sell_level] = shares_bought
                    break
        
        # Record equity
        grid_equity_value = grid_cash + grid_shares * current_price
        grid_equity_list.append((date, grid_equity_value))
    
    grid_equity = pd.Series({d: v for d, v in grid_equity_list})
    
    buy_trades = [t for t in trades if t['type'] == 'buy']
    sell_trades = [t for t in trades if t['type'] == 'sell']
    sell_in_uptrend = len([t for t in sell_trades if t['trend'] == 'uptrend'])
    sell_in_downtrend = len([t for t in sell_trades if t['trend'] == 'downtrend'])
    
    buy_in_uptrend = len([t for t in buy_trades if t['trend'] == 'uptrend'])
    buy_in_downtrend = len([t for t in buy_trades if t['trend'] == 'downtrend'])
    
    stats = {
        'total_trades': len(trades),
        'buy_count': len(buy_trades),
        'sell_count': len(sell_trades),
        'buy_uptrend': buy_in_uptrend,
        'buy_downtrend': buy_in_downtrend,
        'sell_uptrend': sell_in_uptrend,
        'sell_downtrend': sell_in_downtrend,
        'uptrend_days': sum(1 for p, m in zip(price, ma120) if p > m),
        'downtrend_days': sum(1 for p, m in zip(price, ma120) if p <= m),
    }
    
    return base_equity, grid_equity, stats


def max_drawdown(series: pd.Series) -> float:
    roll_max = series.cummax()
    drawdown = series / roll_max - 1.0
    return drawdown.min()


def annualized_return(final_value: float, initial: float, start: pd.Timestamp, end: pd.Timestamp) -> float:
    days = max((end - start).days, 1)
    return (final_value / initial) ** (365.0 / days) - 1.0


def main():
    print("Fetching price data...")
    price = fetch_price(TICKER, START_DATE)
    
    print("Running grid strategy WITHOUT trend filter...")
    base_eq_no_filter, grid_eq_no_filter, stats_no_filter = run_grid_strategy(price, use_trend_filter=False)
    total_eq_no_filter = base_eq_no_filter + grid_eq_no_filter
    
    print("Running grid strategy WITH MA120 trend filter...")
    base_eq_filter, grid_eq_filter, stats_filter = run_grid_strategy(price, use_trend_filter=True)
    total_eq_filter = base_eq_filter + grid_eq_filter
    
    # Pure benchmark
    bh_shares = INITIAL_CASH / price.iloc[0]
    bh_equity = price * bh_shares
    
    print("\n" + "=" * 80)
    print("GRID STRATEGY COMPARISON: WITH vs WITHOUT TREND FILTER (MA120)")
    print("=" * 80)
    print(f"Period: {price.index[0].date()} -> {price.index[-1].date()} ({len(price)} days)")
    print(f"Initial Capital: {INITIAL_CASH:,.0f}")
    print()
    
    # Strategy 1: Without filter
    final_no_filter = total_eq_no_filter.iloc[-1]
    return_no_filter = final_no_filter / INITIAL_CASH - 1.0
    cagr_no_filter = annualized_return(final_no_filter, INITIAL_CASH, total_eq_no_filter.index[0], total_eq_no_filter.index[-1])
    mdd_no_filter = max_drawdown(total_eq_no_filter)
    
    print("--- Strategy 1: Grid WITHOUT Trend Filter ---")
    print(f"Final Value:       {final_no_filter:>12,.0f}")
    print(f"Total Return:      {return_no_filter:>12.2%}")
    print(f"CAGR:              {cagr_no_filter:>12.2%}")
    print(f"Max Drawdown:      {mdd_no_filter:>12.2%}")
    print(f"Total Trades:      {stats_no_filter['total_trades']:>12}")
    print(f"  - Buys:          {stats_no_filter['buy_count']:>12}")
    print(f"  - Sells:         {stats_no_filter['sell_count']:>12}")
    print()
    
    # Strategy 2: With filter
    final_filter = total_eq_filter.iloc[-1]
    return_filter = final_filter / INITIAL_CASH - 1.0
    cagr_filter = annualized_return(final_filter, INITIAL_CASH, total_eq_filter.index[0], total_eq_filter.index[-1])
    mdd_filter = max_drawdown(total_eq_filter)
    
    print("--- Strategy 2: Grid WITH MA120 Trend Filter ---")
    print(f"Final Value:       {final_filter:>12,.0f}")
    print(f"Total Return:      {return_filter:>12.2%}")
    print(f"CAGR:              {cagr_filter:>12.2%}")
    print(f"Max Drawdown:      {mdd_filter:>12.2%}")
    print(f"Total Trades:      {stats_filter['total_trades']:>12}")
    print(f"  - Buys:          {stats_filter['buy_count']:>12}")
    print(f"  - Sells:         {stats_filter['sell_count']:>12}")
    print(f"Uptrend Days:      {stats_filter['uptrend_days']:>12}")
    print(f"Downtrend Days:    {stats_filter['downtrend_days']:>12}")
    print()
    
    # Benchmark
    bh_final = bh_equity.iloc[-1]
    bh_return = bh_final / INITIAL_CASH - 1.0
    bh_cagr = annualized_return(bh_final, INITIAL_CASH, bh_equity.index[0], bh_equity.index[-1])
    bh_mdd = max_drawdown(bh_equity)
    
    print("--- Benchmark: Pure Buy & Hold ---")
    print(f"Final Value:       {bh_final:>12,.0f}")
    print(f"Total Return:      {bh_return:>12.2%}")
    print(f"CAGR:              {bh_cagr:>12.2%}")
    print(f"Max Drawdown:      {bh_mdd:>12.2%}")
    print()
    
    # Comparison
    print("--- Comparison ---")
    print(f"CAGR vs Benchmark (without filter): {(cagr_no_filter - bh_cagr):>10.2%}")
    print(f"CAGR vs Benchmark (with filter):    {(cagr_filter - bh_cagr):>10.2%}")
    print(f"Improvement from filter:            {(cagr_filter - cagr_no_filter):>10.2%}")
    print(f"Trade reduction:                    {((stats_no_filter['total_trades'] - stats_filter['total_trades']) / stats_no_filter['total_trades'] * 100):>10.1f}%")
    print()
    
    # Export
    out = pd.DataFrame({
        'grid_no_filter': total_eq_no_filter,
        'grid_with_filter': total_eq_filter,
        'benchmark': bh_equity,
    })
    out.to_csv("analysis_output_grid_trend_filter.csv")
    print("Saved equity curves to analysis_output_grid_trend_filter.csv")
    print("=" * 80)
    
    print("\nKEY INSIGHTS:")
    print(f"1. Trend filter reduces trades by {((stats_no_filter['total_trades'] - stats_filter['total_trades']) / stats_no_filter['total_trades'] * 100):.1f}%")
    print(f"2. CAGR improvement: {(cagr_filter - cagr_no_filter):.2%}")
    print(f"3. Downtrend avoidance: During {stats_filter['downtrend_days']} downtrend days, grid paused selling")
    print(f"4. MDD improvement: {(mdd_filter - mdd_no_filter):.2%} (less negative is better)")


if __name__ == "__main__":
    main()
